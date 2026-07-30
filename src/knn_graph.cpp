#include "coresg/knn_graph.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_set>

#include "coresg/thread_pool.hpp"

namespace coresg {

void KnnGraph::reset(idx_t n, std::uint32_t k) {
    n_ = n;
    k_ = k;
    const std::size_t total = std::size_t(n) * k;
    ids_.assign(total, kInvalidIdx);
    dist_.assign(total, kInfDist);
    flags_.assign(total, 0);
    count_.assign(n, 0);
    locks_.resize(n == 0 ? 1 : n);
}

void KnnGraph::grow(idx_t new_n) {
    if (new_n <= n_) return;
    const std::size_t total = std::size_t(new_n) * k_;
    ids_.resize(total, kInvalidIdx);
    dist_.resize(total, kInfDist);
    flags_.resize(total, 0);
    count_.resize(new_n, 0);
    n_ = new_n;
    locks_.resize(new_n);
}

void KnnGraph::ensure_locks() {
    if (locks_.size() < n_) locks_.resize(n_ == 0 ? 1 : n_);
}

InsertResult KnnGraph::insert_unlocked(idx_t node, idx_t cand, dist_t d, bool is_new) {
    InsertResult res;
    if (cand == node || cand == kInvalidIdx) return res;

    const std::size_t base = std::size_t(node) * k_;
    idx_t* I = ids_.data() + base;
    dist_t* D = dist_.data() + base;
    std::uint8_t* F = flags_.data() + base;
    const std::uint32_t cnt = count_[node];

    if (cnt == k_ && !(d < D[k_ - 1])) return res;

    // Duplicate check. O(k) but k is small and this only runs for candidates
    // that already passed the distance test.
    for (std::uint32_t j = 0; j < cnt; ++j) {
        if (I[j] == cand) return res;
    }

    std::uint32_t start;
    if (cnt < k_) {
        start = cnt;
        count_[node] = cnt + 1;
    } else {
        res.evicted = I[k_ - 1];
        start = k_ - 1;
    }

    std::uint32_t pos = start;
    while (pos > 0 && D[pos - 1] > d) {
        D[pos] = D[pos - 1];
        I[pos] = I[pos - 1];
        F[pos] = F[pos - 1];
        --pos;
    }
    D[pos] = d;
    I[pos] = cand;
    F[pos] = is_new ? 1u : 0u;
    res.inserted = true;

    if (observer_) {
        if (res.evicted != kInvalidIdx) observer_->on_edge_removed(node, res.evicted);
        observer_->on_edge_added(node, cand);
    }
    return res;
}

InsertResult KnnGraph::try_insert(idx_t node, idx_t cand, dist_t d, bool is_new) {
    return insert_unlocked(node, cand, d, is_new);
}

InsertResult KnnGraph::try_insert_mt(idx_t node, idx_t cand, dist_t d, bool is_new) {
    if (cand == node || cand == kInvalidIdx) return InsertResult{};
    if (would_reject(node, d)) return InsertResult{};  // lock-free fast reject
    locks_.lock(node);
    InsertResult res = insert_unlocked(node, cand, d, is_new);
    locks_.unlock(node);
    return res;
}

void KnnGraph::clear_new_flags() { std::fill(flags_.begin(), flags_.end(), std::uint8_t(0)); }

void KnnGraph::set_all_new_flags() {
    for (idx_t i = 0; i < n_; ++i) {
        const std::size_t base = std::size_t(i) * k_;
        const std::uint32_t c = count_[i];
        for (std::uint32_t j = 0; j < c; ++j) flags_[base + j] = 1;
    }
}

std::vector<dist_t> KnnGraph::core_distances(std::uint32_t mpts, ThreadPool* pool) const {
    std::vector<dist_t> core(n_);
    auto body = [&](std::size_t lo, std::size_t hi, int) {
        for (std::size_t i = lo; i < hi; ++i) core[i] = core_dist(static_cast<idx_t>(i), mpts);
    };
    if (pool) {
        pool->parallel_ranges(n_, body);
    } else {
        body(0, n_, 0);
    }
    return core;
}

idx_t KnnGraph::incomplete_nodes() const {
    idx_t c = 0;
    for (idx_t i = 0; i < n_; ++i) {
        if (count_[i] < k_) ++c;
    }
    return c;
}

namespace {

// Collects the undirected edges owned by nodes in [lo, hi). An edge (p,q) is
// owned by p when p < q, or by p when q < p and p is not in q's list -- this
// keeps mutual pairs from being emitted twice without a global hash set.
template <typename Emit>
void collect_undirected(const KnnGraph& g, std::size_t lo, std::size_t hi, Emit&& emit) {
    for (std::size_t i = lo; i < hi; ++i) {
        const idx_t p = static_cast<idx_t>(i);
        const idx_t* I = g.ids(p);
        const dist_t* D = g.dists(p);
        const std::uint32_t c = g.degree(p);
        for (std::uint32_t j = 0; j < c; ++j) {
            const idx_t q = I[j];
            if (q == kInvalidIdx || q == p) continue;
            if (q < p) {
                // Emit only if the reverse edge is absent from q's list,
                // otherwise q already emitted it.
                const idx_t* J = g.ids(q);
                const std::uint32_t cq = g.degree(q);
                bool mutual = false;
                for (std::uint32_t t = 0; t < cq; ++t) {
                    if (J[t] == p) {
                        mutual = true;
                        break;
                    }
                }
                if (mutual) continue;
            }
            emit(p, q, D[j]);
        }
    }
}

}  // namespace

std::vector<Edge> KnnGraph::undirected_edges(ThreadPool* pool) const {
    std::vector<Edge> out;
    if (!pool || pool->size() <= 1) {
        out.reserve(std::size_t(n_) * k_ / 2 + 8);
        collect_undirected(*this, 0, n_, [&](idx_t p, idx_t q, dist_t w) {
            out.emplace_back(p, q, w);
        });
        return out;
    }

    const unsigned nt = pool->size();
    std::vector<std::vector<Edge>> parts(nt);
    pool->parallel_ranges(n_, [&](std::size_t lo, std::size_t hi, int slot) {
        std::vector<Edge>& local = parts[slot];
        local.reserve((hi - lo) * k_ / 2 + 8);
        collect_undirected(*this, lo, hi, [&](idx_t p, idx_t q, dist_t w) {
            local.emplace_back(p, q, w);
        });
    });
    std::size_t total = 0;
    for (const auto& p : parts) total += p.size();
    out.reserve(total);
    for (auto& p : parts) {
        out.insert(out.end(), p.begin(), p.end());
        std::vector<Edge>().swap(p);
    }
    return out;
}

std::size_t KnnGraph::undirected_edge_count(ThreadPool* pool) const {
    if (!pool || pool->size() <= 1) {
        std::size_t c = 0;
        collect_undirected(*this, 0, n_, [&](idx_t, idx_t, dist_t) { ++c; });
        return c;
    }
    std::vector<std::size_t> parts(pool->size(), 0);
    pool->parallel_ranges(n_, [&](std::size_t lo, std::size_t hi, int slot) {
        std::size_t c = 0;
        collect_undirected(*this, lo, hi, [&](idx_t, idx_t, dist_t) { ++c; });
        parts[slot] = c;
    });
    std::size_t total = 0;
    for (std::size_t c : parts) total += c;
    return total;
}

std::vector<Edge> KnnGraph::mutual_reachability_edges(const std::vector<dist_t>& core,
                                                      ThreadPool* pool) const {
    std::vector<Edge> edges = undirected_edges(pool);
    auto body = [&](std::size_t lo, std::size_t hi, int) {
        for (std::size_t i = lo; i < hi; ++i) {
            Edge& e = edges[i];
            const dist_t c = std::max(core[e.u], core[e.v]);
            e.w = std::max(e.w, c);
        }
    };
    if (pool) {
        pool->parallel_ranges(edges.size(), body);
    } else {
        body(0, edges.size(), 0);
    }
    return edges;
}

bool KnnGraph::validate(std::string* error) const {
    std::ostringstream oss;
    for (idx_t i = 0; i < n_; ++i) {
        const idx_t* I = ids(i);
        const dist_t* D = dists(i);
        const std::uint32_t c = count_[i];
        if (c > k_) {
            oss << "node " << i << ": count " << c << " > k " << k_;
            if (error) *error = oss.str();
            return false;
        }
        std::unordered_set<idx_t> seen;
        for (std::uint32_t j = 0; j < c; ++j) {
            if (I[j] == i) {
                oss << "node " << i << ": self loop at slot " << j;
                if (error) *error = oss.str();
                return false;
            }
            if (I[j] == kInvalidIdx) {
                oss << "node " << i << ": invalid id inside filled range";
                if (error) *error = oss.str();
                return false;
            }
            if (!seen.insert(I[j]).second) {
                oss << "node " << i << ": duplicate neighbour " << I[j];
                if (error) *error = oss.str();
                return false;
            }
            if (j > 0 && D[j] < D[j - 1]) {
                oss << "node " << i << ": unsorted at slot " << j;
                if (error) *error = oss.str();
                return false;
            }
        }
    }
    return true;
}

}  // namespace coresg
