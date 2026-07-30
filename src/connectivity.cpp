#include "coresg/connectivity.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>

#include "coresg/hubness.hpp"
#include "coresg/mst.hpp"
#include "coresg/thread_pool.hpp"

namespace coresg {
namespace {

using Clock = std::chrono::steady_clock;
inline double since(const Clock::time_point& t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

/// Lock-free union-find. Roots are linked by index (larger id under smaller),
/// which is what makes concurrent unions acyclic and the outcome independent
/// of the thread schedule -- so a parallel rebuild produces exactly the labels
/// a serial one would.
class AtomicUnionFind {
public:
    explicit AtomicUnionFind(idx_t n) : n_(n), parent_(new std::atomic<idx_t>[n ? n : 1]) {
        for (idx_t i = 0; i < n; ++i) parent_[i].store(i, std::memory_order_relaxed);
    }

    idx_t find(idx_t x) {
        for (;;) {
            const idx_t p = parent_[x].load(std::memory_order_relaxed);
            if (p == x) return x;
            const idx_t gp = parent_[p].load(std::memory_order_relaxed);
            if (gp != p) {
                // Path halving; a lost race just means less compression.
                idx_t expected = p;
                parent_[x].compare_exchange_weak(expected, gp, std::memory_order_relaxed,
                                                 std::memory_order_relaxed);
            }
            x = gp;
        }
    }

    bool unite(idx_t a, idx_t b) {
        for (;;) {
            a = find(a);
            b = find(b);
            if (a == b) return false;
            if (a > b) {
                const idx_t t = a;
                a = b;
                b = t;
            }
            idx_t expected = b;
            if (parent_[b].compare_exchange_strong(expected, a, std::memory_order_release,
                                                   std::memory_order_relaxed)) {
                return true;
            }
        }
    }

    /// Dense labels in [0, components).
    idx_t finalize(std::vector<idx_t>* labels) {
        labels->assign(n_, kInvalidIdx);
        std::vector<idx_t> remap(n_, kInvalidIdx);
        idx_t next = 0;
        for (idx_t i = 0; i < n_; ++i) {
            const idx_t r = find(i);
            if (remap[r] == kInvalidIdx) remap[r] = next++;
            (*labels)[i] = remap[r];
        }
        return next;
    }

private:
    idx_t n_;
    std::unique_ptr<std::atomic<idx_t>[]> parent_;
};

}  // namespace

void ConnectivityTracker::reset(idx_t n) {
    n_ = n;
    components_ = n;
    labels_.assign(n, kInvalidIdx);
    seen_.assign(n, 0);
    stamp_ = 0;
    valid_ = false;
}

void ConnectivityTracker::grow(idx_t n) {
    if (n <= n_) return;
    labels_.resize(n, kInvalidIdx);
    seen_.resize(n, 0);
    n_ = n;
}

ConnectivityStats ConnectivityTracker::rebuild(const KnnGraph& g, ThreadPool* pool) {
    const auto t0 = Clock::now();
    ConnectivityStats st;
    grow(g.size());
    const idx_t n = g.size();

    AtomicUnionFind uf(n);
    std::vector<std::size_t> counts(pool && pool->size() > 1 ? pool->size() : 1, 0);

    // Straight over the adjacency: no edge list is built, so this touches
    // n*k ids and nothing else.
    auto body = [&](std::size_t lo, std::size_t hi, int slot) {
        std::size_t c = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            const idx_t p = static_cast<idx_t>(i);
            const idx_t* I = g.ids(p);
            const std::uint32_t deg = g.degree(p);
            for (std::uint32_t j = 0; j < deg; ++j) {
                const idx_t q = I[j];
                if (q < n && q != p) {
                    uf.unite(p, q);
                    ++c;
                }
            }
        }
        counts[slot] = c;
    };
    if (pool && pool->size() > 1) {
        pool->parallel_ranges(n, body);
    } else {
        body(0, n, 0);
    }
    for (std::size_t c : counts) st.union_edges += c;

    components_ = uf.finalize(&labels_);
    valid_ = true;
    st.components = components_;
    st.incremental = false;
    st.seconds = since(t0);
    return st;
}

bool ConnectivityTracker::certify(const KnnGraph& g, idx_t u, idx_t v,
                                  const HubnessIndex* reverse, std::size_t budget,
                                  std::size_t* visited, bool* exhausted) {
    *exhausted = false;
    if (u == v) return true;
    const idx_t n = g.size();
    if (u >= n || v >= n) return true;

    // Two stamps per call: one marks the side that started at u, the other the
    // side that started at v. Meeting a node marked by the other side is a
    // replacement path.
    if (stamp_ + 2 < stamp_) {  // wrapped
        std::fill(seen_.begin(), seen_.end(), 0u);
        stamp_ = 0;
    }
    const std::uint32_t mark_a = ++stamp_;
    const std::uint32_t mark_b = ++stamp_;

    frontier_a_.clear();
    frontier_b_.clear();
    frontier_a_.push_back(u);
    frontier_b_.push_back(v);
    seen_[u] = mark_a;
    seen_[v] = mark_b;

    std::size_t used = 0;
    const bool undirected = reverse && reverse->has_reverse_lists();

    auto expand = [&](std::vector<idx_t>* frontier, std::uint32_t mine,
                      std::uint32_t theirs) -> int {
        next_.clear();
        for (idx_t x : *frontier) {
            const idx_t* I = g.ids(x);
            const std::uint32_t deg = g.degree(x);
            for (std::uint32_t j = 0; j < deg; ++j) {
                const idx_t y = I[j];
                if (y >= n) continue;
                if (seen_[y] == theirs) return 1;  // the two sides met
                if (seen_[y] == mine) continue;
                seen_[y] = mine;
                next_.push_back(y);
                if (++used > budget) return -1;
            }
            if (undirected) {
                // Without the reverse lists only forward edges are visible, so
                // the search explores a subgraph of the real undirected graph.
                // That is still sound -- it just fails more often, and a
                // failure only means a full rebuild.
                const std::vector<idx_t>& rev = reverse->reverse_neighbours(x);
                for (idx_t y : rev) {
                    if (y >= n) continue;
                    if (seen_[y] == theirs) return 1;
                    if (seen_[y] == mine) continue;
                    seen_[y] = mine;
                    next_.push_back(y);
                    if (++used > budget) return -1;
                }
            }
        }
        frontier->swap(next_);
        return 0;
    };

    while (!frontier_a_.empty() || !frontier_b_.empty()) {
        // Always grow the smaller side: keeps the explored region balanced.
        std::vector<idx_t>* side = (frontier_a_.size() <= frontier_b_.size() && !frontier_a_.empty())
                                       ? &frontier_a_
                                       : &frontier_b_;
        std::uint32_t mine = (side == &frontier_a_) ? mark_a : mark_b;
        std::uint32_t theirs = (side == &frontier_a_) ? mark_b : mark_a;
        if (side->empty()) {
            side = (side == &frontier_a_) ? &frontier_b_ : &frontier_a_;
            mine = (side == &frontier_a_) ? mark_a : mark_b;
            theirs = (side == &frontier_a_) ? mark_b : mark_a;
        }
        const int r = expand(side, mine, theirs);
        *visited += used;
        if (r == 1) return true;
        if (r == -1) return false;  // out of budget: inconclusive
        used = 0;
    }
    *exhausted = true;  // searched everything reachable: genuinely disconnected
    return false;
}

ConnectivityStats ConnectivityTracker::update(
    const KnnGraph& g, const std::vector<std::pair<idx_t, idx_t>>& removal_candidates,
    const std::vector<idx_t>& changed_nodes, idx_t first_new, const HubnessIndex* reverse,
    ThreadPool* pool, std::size_t probe_budget) {
    const auto t0 = Clock::now();
    grow(g.size());
    if (!valid_) return rebuild(g, pool);

    ConnectivityStats st;
    st.removal_candidates = removal_candidates.size();

    // A recorded eviction only destroys the *undirected* edge if the other
    // endpoint stopped listing it too. The graph is quiescent now, so this
    // filter is exact.
    auto lists = [&](idx_t a, idx_t b) {
        const idx_t* I = g.ids(a);
        const std::uint32_t deg = g.degree(a);
        for (std::uint32_t j = 0; j < deg; ++j) {
            if (I[j] == b) return true;
        }
        return false;
    };

    // An uncertified edge invalidates the seed only for the old component(s)
    // it touched -- the splicing argument still holds inside every other one.
    // So instead of discarding the whole seed, mark those components dirty and
    // re-derive just their nodes. In practice a batch produces a handful of
    // genuine migrations, so this keeps the cost at one cluster rather than
    // the whole graph.
    std::vector<std::uint8_t> dirty(components_, 0);
    std::size_t dirty_components = 0;
    auto mark_dirty = [&](idx_t x) {
        if (x >= first_new || x >= labels_.size()) return;  // new points have no old label
        const idx_t lab = labels_[x];
        if (lab >= components_) return;
        if (!dirty[lab]) {
            dirty[lab] = 1;
            ++dirty_components;
        }
    };

    for (const std::pair<idx_t, idx_t>& r : removal_candidates) {
        const idx_t u = r.first, v = r.second;
        if (u >= g.size() || v >= g.size()) continue;
        if (lists(u, v) || lists(v, u)) continue;  // survived
        ++st.destroyed_edges;
        bool exhausted = false;
        if (certify(g, u, v, reverse, probe_budget, &st.probe_nodes, &exhausted)) {
            ++st.certified_edges;
            continue;
        }
        if (st.fail_u == kInvalidIdx) {
            st.fail_u = u;
            st.fail_v = v;
            st.fail_exhausted = exhausted;
        }
        mark_dirty(u);
        mark_dirty(v);
    }
    st.dirty_components = dirty_components;

    // Past roughly half the components there is nothing left to reuse.
    if (components_ == 0 || dirty_components * 2 >= components_) {
        ConnectivityStats full = rebuild(g, pool);
        full.removal_candidates = st.removal_candidates;
        full.destroyed_edges = st.destroyed_edges;
        full.certified_edges = st.certified_edges;
        full.probe_nodes = st.probe_nodes;
        full.dirty_components = dirty_components;
        full.fail_u = st.fail_u;
        full.fail_v = st.fail_v;
        full.fail_exhausted = st.fail_exhausted;
        full.seconds = since(t0);
        return full;
    }

    const idx_t n = g.size();
    AtomicUnionFind uf(n);

    auto union_node = [&](idx_t p) {
        const idx_t* I = g.ids(p);
        const std::uint32_t deg = g.degree(p);
        for (std::uint32_t j = 0; j < deg; ++j) {
            const idx_t q = I[j];
            if (q < n && q != p) {
                uf.unite(p, q);
                ++st.union_edges;
            }
        }
    };

    // Clean components are re-established by linking every member to the first
    // one: a pass over a single array instead of over n*k neighbour ids.
    // Dirty components are rebuilt from their adjacency.
    {
        std::vector<idx_t> rep(components_, kInvalidIdx);
        const idx_t upto = std::min<idx_t>(first_new, static_cast<idx_t>(labels_.size()));
        for (idx_t v = 0; v < upto; ++v) {
            const idx_t lab = labels_[v];
            if (lab >= components_) continue;
            if (dirty[lab]) {
                union_node(v);
                ++st.rederived_nodes;
                continue;
            }
            if (rep[lab] == kInvalidIdx) {
                rep[lab] = v;
            } else {
                uf.unite(rep[lab], v);
            }
        }
    }

    for (idx_t v : changed_nodes) {
        if (v < n) union_node(v);
    }
    for (idx_t v = first_new; v < n; ++v) union_node(v);

    components_ = uf.finalize(&labels_);
    valid_ = true;
    st.components = components_;
    st.incremental = true;
    st.seconds = since(t0);
    return st;
}

std::vector<idx_t> ConnectivityTracker::labels_with(const std::vector<Edge>& extra,
                                                    idx_t* num_components) const {
    if (extra.empty()) {
        if (num_components) *num_components = components_;
        return labels_;
    }
    // Contract each k-NN component to a node and union only the extra edges:
    // O(|extra|) unions plus one pass to relabel.
    UnionFind uf(components_);
    for (const Edge& e : extra) {
        if (e.u >= n_ || e.v >= n_) continue;
        const idx_t a = labels_[e.u], b = labels_[e.v];
        if (a < components_ && b < components_) uf.unite(a, b);
    }
    const std::vector<idx_t> merged = uf.labels();
    std::vector<idx_t> out(n_);
    for (idx_t v = 0; v < n_; ++v) {
        const idx_t lab = labels_[v];
        out[v] = (lab < components_) ? merged[lab] : lab;
    }
    if (num_components) *num_components = uf.components();
    return out;
}

}  // namespace coresg
