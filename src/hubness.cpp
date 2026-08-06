#include "coresg/hubness.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "coresg/thread_pool.hpp"

namespace coresg {
namespace {

/// Number of anti-hubs to compare against each other. Deliberately sublinear:
/// the missing edges are only an *estimate* of the bridges, so comparing every
/// anti-hub with every other one would pay a quadratic price for accuracy the
/// method does not need.
std::size_t antihub_budget(const HubnessParams& p, std::uint32_t dim, std::size_t num_edges,
                           idx_t n, std::size_t* uncapped, bool* capped) {
    if (n == 0) return 0;
    const double logs = (num_edges > 1) ? std::log(static_cast<double>(num_edges)) : 0.0;
    double raw = p.antihub_multiplier * static_cast<double>(dim) * logs;
    if (!(raw > 0.0)) raw = 0.0;
    std::size_t count = static_cast<std::size_t>(std::llround(raw));
    if (uncapped) *uncapped = count;
    const std::size_t requested = count;

    // Cost ceiling. The MST over m anti-hubs costs ~m^2*d/2 coordinate ops, so
    // solving m^2*d/2 <= max_pair_ops for m bounds the bridge step directly
    // instead of guessing a count that happens to work at one dimensionality.
    if (p.max_pair_ops > 0.0 && dim > 0) {
        const double m_max = std::sqrt(2.0 * p.max_pair_ops / static_cast<double>(dim));
        if (m_max >= 1.0 && static_cast<double>(count) > m_max) {
            count = static_cast<std::size_t>(m_max);
        }
    }
    if (p.max_selected && count > p.max_selected) count = p.max_selected;
    if (count < p.min_selected) count = p.min_selected;
    if (count > n) count = n;
    if (capped) *capped = count < requested;
    return count;
}

}  // namespace

void HubnessIndex::build(const KnnGraph& g, ThreadPool* pool) {
    n_ = g.size();
    k_ = params_.k_for_score ? params_.k_for_score : g.k();
    rev_enabled_ = params_.store_reverse_lists && n_ <= params_.reverse_list_node_limit;

    rdeg_.reset(new std::atomic<std::int32_t>[n_ ? n_ : 1]);
    dirty_.reset(new std::atomic<std::uint8_t>[n_ ? n_ : 1]);
    nbr_changed_.reset(new std::atomic<std::uint8_t>[n_ ? n_ : 1]);
    for (idx_t i = 0; i < n_; ++i) {
        rdeg_[i].store(0, std::memory_order_relaxed);
        dirty_[i].store(0, std::memory_order_relaxed);
        nbr_changed_[i].store(0, std::memory_order_relaxed);
    }
    score_.assign(n_, 1.0);
    antihub_.assign(n_, 0);
    stamp_.assign(n_, 0);
    epoch_ = 0;
    for (std::size_t s = 0; s < kShards; ++s) {
        shards_[s].dirty.clear();
        shards_[s].nbr_changed.clear();
        shards_[s].removals.clear();
    }
    dirty_list_.clear();
    nbr_changed_list_.clear();
    removals_.clear();
    selected_.clear();

    rev_.clear();
    if (rev_enabled_) {
        rev_.resize(n_);
        rev_locks_.resize(n_ ? n_ : 1);
    }

    // Reverse degrees (and lists) from scratch. Counting is done with atomics
    // so the scan parallelises without a per-node lock.
    auto count_body = [&](std::size_t lo, std::size_t hi, int) {
        for (std::size_t i = lo; i < hi; ++i) {
            const idx_t p = static_cast<idx_t>(i);
            const idx_t* I = g.ids(p);
            const std::uint32_t c = g.degree(p);
            for (std::uint32_t j = 0; j < c; ++j) {
                const idx_t q = I[j];
                if (q >= n_) continue;
                rdeg_[q].fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    if (pool) {
        pool->parallel_ranges(n_, count_body);
    } else {
        count_body(0, n_, 0);
    }

    if (rev_enabled_) {
        // Sequential fill: push_back on n small vectors is memory bound, and
        // doing it single threaded avoids n locks for a one-off cost.
        for (idx_t p = 0; p < n_; ++p) {
            const idx_t* I = g.ids(p);
            const std::uint32_t c = g.degree(p);
            for (std::uint32_t j = 0; j < c; ++j) {
                const idx_t q = I[j];
                if (q < n_) rev_[q].push_back(p);
            }
        }
    }
}

void HubnessIndex::grow(idx_t new_n) {
    if (new_n <= n_) return;
    std::unique_ptr<std::atomic<std::int32_t>[]> rd(new std::atomic<std::int32_t>[new_n]);
    std::unique_ptr<std::atomic<std::uint8_t>[]> di(new std::atomic<std::uint8_t>[new_n]);
    std::unique_ptr<std::atomic<std::uint8_t>[]> nc(new std::atomic<std::uint8_t>[new_n]);
    for (idx_t i = 0; i < new_n; ++i) {
        rd[i].store(i < n_ ? rdeg_[i].load(std::memory_order_relaxed) : 0,
                    std::memory_order_relaxed);
        di[i].store(i < n_ ? dirty_[i].load(std::memory_order_relaxed) : std::uint8_t(0),
                    std::memory_order_relaxed);
        nc[i].store(i < n_ ? nbr_changed_[i].load(std::memory_order_relaxed) : std::uint8_t(0),
                    std::memory_order_relaxed);
    }
    rdeg_ = std::move(rd);
    dirty_ = std::move(di);
    nbr_changed_ = std::move(nc);

    score_.resize(new_n, 1.0);
    antihub_.resize(new_n, 0);
    stamp_.resize(new_n, 0);
    if (rev_enabled_) {
        rev_.resize(new_n);
        rev_locks_.resize(new_n);
    }
    n_ = new_n;
}

void HubnessIndex::attach(KnnGraph* g) {
    grow(g->size());
    g->set_observer(this);
}

void HubnessIndex::detach(KnnGraph* g) {
    if (g->observer() == this) g->set_observer(nullptr);
}

void HubnessIndex::mark_dirty(idx_t v) {
    if (v >= n_) return;
    if (dirty_[v].exchange(1, std::memory_order_relaxed) == 0) {
        ListShard& sh = shard_for(v);
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.dirty.push_back(v);
    }
}

void HubnessIndex::mark_neighbourhood_changed(idx_t v) {
    if (v >= n_) return;
    if (nbr_changed_[v].exchange(1, std::memory_order_relaxed) == 0) {
        ListShard& sh = shard_for(v);
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.nbr_changed.push_back(v);
    }
}

/// Flatten the shards into the contiguous views the public accessors return.
/// Idempotent: the shards are the authority and are only cleared by a refresh.
void HubnessIndex::gather_lists() const {
    std::size_t nd = 0, nc = 0, nr = 0;
    for (std::size_t s = 0; s < kShards; ++s) {
        nd += shards_[s].dirty.size();
        nc += shards_[s].nbr_changed.size();
        nr += shards_[s].removals.size();
    }
    if (dirty_list_.size() == nd && nbr_changed_list_.size() == nc &&
        removals_.size() == nr) {
        return;  // already current
    }
    dirty_list_.clear();
    nbr_changed_list_.clear();
    removals_.clear();
    dirty_list_.reserve(nd);
    nbr_changed_list_.reserve(nc);
    removals_.reserve(nr);
    for (std::size_t s = 0; s < kShards; ++s) {
        const ListShard& sh = shards_[s];
        dirty_list_.insert(dirty_list_.end(), sh.dirty.begin(), sh.dirty.end());
        nbr_changed_list_.insert(nbr_changed_list_.end(), sh.nbr_changed.begin(),
                                 sh.nbr_changed.end());
        removals_.insert(removals_.end(), sh.removals.begin(), sh.removals.end());
    }
}

void HubnessIndex::on_edge_added(idx_t src, idx_t dst) {
    if (dst >= n_ || src >= n_) return;
    rdeg_[dst].fetch_add(1, std::memory_order_relaxed);
    mark_dirty(dst);
    mark_neighbourhood_changed(src);
    if (rev_enabled_) {
        rev_locks_.lock(dst);
        rev_[dst].push_back(src);
        rev_locks_.unlock(dst);
    }
}

void HubnessIndex::on_edge_removed(idx_t src, idx_t dst) {
    if (dst >= n_ || src >= n_) return;
    rdeg_[dst].fetch_sub(1, std::memory_order_relaxed);
    mark_dirty(dst);
    mark_neighbourhood_changed(src);
    if (track_removals_) {
        // Recorded unfiltered: whether the *undirected* edge died depends on
        // dst's list, which is being mutated concurrently right now. The
        // consumer filters once the batch is quiescent.
        ListShard& sh = shard_for(dst);
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.removals.emplace_back(src, dst);
    }
    if (rev_enabled_) {
        rev_locks_.lock(dst);
        std::vector<idx_t>& lst = rev_[dst];
        for (std::size_t i = lst.size(); i-- > 0;) {
            if (lst[i] == src) {
                lst[i] = lst.back();
                lst.pop_back();
                break;
            }
        }
        rev_locks_.unlock(dst);
    }
}

const char* to_string(ScoreNormalization n) {
    switch (n) {
        case ScoreNormalization::ArithmeticMean: return "arithmetic";
        case ScoreNormalization::GeometricMean: return "geometric";
        case ScoreNormalization::Median: return "median";
    }
    return "?";
}

bool score_normalization_from_string(const std::string& s, ScoreNormalization* out) {
    if (s == "arithmetic" || s == "mean") { *out = ScoreNormalization::ArithmeticMean; return true; }
    if (s == "geometric" || s == "geo") { *out = ScoreNormalization::GeometricMean; return true; }
    if (s == "median") { *out = ScoreNormalization::Median; return true; }
    return false;
}

double HubnessIndex::compute_score(const KnnGraph& g, idx_t v) const {
    std::vector<std::int32_t> scratch;
    return compute_score(g, v, &scratch);
}

double HubnessIndex::compute_score(const KnnGraph& g, idx_t v,
                                   std::vector<std::int32_t>* scratch) const {
    const std::uint32_t deg = g.degree(v);
    const idx_t* I = g.ids(v);
    const std::int32_t rd = rdeg_[v].load(std::memory_order_relaxed);

    // The "typical" reverse degree around v. `sum` is kept for the degenerate
    // rules, which are defined in terms of the neighbourhood total and must
    // therefore behave identically whichever normalisation is in use.
    std::int64_t sum = 0;
    scratch->clear();
    if (params_.normalization != ScoreNormalization::ArithmeticMean) scratch->reserve(deg);
    for (std::uint32_t j = 0; j < deg; ++j) {
        const idx_t q = I[j];
        if (q >= n_) continue;
        const std::int32_t d = rdeg_[q].load(std::memory_order_relaxed);
        sum += d;
        if (params_.normalization != ScoreNormalization::ArithmeticMean) scratch->push_back(d);
    }

    if (rd > 0 && sum <= 0) return static_cast<double>(k_);
    if (rd <= 0 && sum > 0) return 0.0;
    if (rd <= 0 && sum <= 0) return 1.0;
    if (deg == 0) return 1.0;

    double denom = 0.0;
    switch (params_.normalization) {
        case ScoreNormalization::ArithmeticMean:
            denom = static_cast<double>(sum) / static_cast<double>(deg);
            break;
        case ScoreNormalization::GeometricMean: {
            // Shifted by 1 so that zero-degree neighbours are admissible;
            // undone afterwards, which keeps it comparable to the mean.
            double acc = 0.0;
            for (std::int32_t d : *scratch) acc += std::log(static_cast<double>(d) + 1.0);
            denom = std::exp(acc / static_cast<double>(scratch->size())) - 1.0;
            break;
        }
        case ScoreNormalization::Median: {
            std::sort(scratch->begin(), scratch->end());
            denom = static_cast<double>((*scratch)[(scratch->size() - 1) / 2]);
            break;
        }
    }

    // A robust denominator can legitimately be 0 while the total is not (say,
    // most of the neighbourhood has no reverse neighbours). Fall back to the
    // degenerate convention rather than dividing by zero.
    if (!(denom > 0.0)) return static_cast<double>(k_);
    return static_cast<double>(rd) / denom;
}

double HubnessIndex::antihub_fraction() const {
    if (n_ == 0) return 0.0;
    std::size_t tagged = 0;
    for (idx_t v = 0; v < n_; ++v) {
        if (antihub_[v]) ++tagged;
    }
    return double(tagged) / double(n_);
}

std::vector<idx_t> HubnessIndex::collect_rescore_set(const KnnGraph& g) {
    gather_lists();
    ++epoch_;
    if (epoch_ == 0) {  // wrapped
        std::fill(stamp_.begin(), stamp_.end(), 0u);
        epoch_ = 1;
    }
    std::vector<idx_t> out;
    out.reserve(dirty_list_.size() + nbr_changed_list_.size());

    auto push = [&](idx_t v) {
        if (v >= n_) return;
        if (stamp_[v] == epoch_) return;
        stamp_[v] = epoch_;
        out.push_back(v);
    };

    // A node's score depends on its own reverse degree and on the reverse
    // degrees of its neighbours. So: everything whose own degree moved,
    // everything whose neighbour list moved, and everything that *has* a
    // moved node as a neighbour.
    for (idx_t v : nbr_changed_list_) push(v);
    for (idx_t v : dirty_list_) {
        push(v);
        if (rev_enabled_) {
            rev_locks_.lock(v);
            const std::vector<idx_t>& lst = rev_[v];
            for (idx_t u : lst) push(u);
            rev_locks_.unlock(v);
        } else {
            // Forward approximation: propagate along the outgoing edges
            // instead. Cheaper and needs no reverse table, but it misses
            // reverse neighbours that are not also forward neighbours.
            const idx_t* I = g.ids(v);
            const std::uint32_t c = g.degree(v);
            for (std::uint32_t j = 0; j < c; ++j) push(I[j]);
        }
    }
    return out;
}

void HubnessIndex::apply_scores(const KnnGraph& g, const std::vector<idx_t>& nodes,
                                ThreadPool* pool) {
    const unsigned nt = (pool && pool->size() > 1) ? pool->size() : 1u;
    std::vector<std::vector<std::int32_t>> scratch(nt);
    auto body = [&](std::size_t lo, std::size_t hi, int slot) {
        std::vector<std::int32_t>& buf = scratch[slot];
        for (std::size_t i = lo; i < hi; ++i) {
            const idx_t v = nodes[i];
            const double s = compute_score(g, v, &buf);
            score_[v] = s;
            antihub_[v] = (s < 1.0) ? 1u : 0u;
        }
    };
    if (nt > 1) {
        pool->parallel_ranges(nodes.size(), body);
    } else {
        body(0, nodes.size(), 0);
    }
}

HubnessStats HubnessIndex::select(std::uint32_t dim, std::size_t num_edges, std::size_t rescored,
                                  ThreadPool* pool) {
    HubnessStats st;
    st.nodes = n_;
    st.rescored = rescored;
    if (n_ == 0) return st;

    const std::size_t budget =
        antihub_budget(params_, dim, num_edges, n_, &st.budget_uncapped, &st.budget_capped);

    std::vector<idx_t> order(n_);
    for (idx_t i = 0; i < n_; ++i) order[i] = i;

    // Deterministic total order: score, then reverse degree, then id, so two
    // runs on the same graph select the same anti-hubs.
    auto cmp = [&](idx_t a, idx_t b) {
        if (score_[a] != score_[b]) return score_[a] < score_[b];
        const std::int32_t ra = rdeg_[a].load(std::memory_order_relaxed);
        const std::int32_t rb = rdeg_[b].load(std::memory_order_relaxed);
        if (ra != rb) return ra < rb;
        return a < b;
    };

    if (budget < order.size()) {
        std::nth_element(order.begin(), order.begin() + budget, order.end(), cmp);
        order.resize(budget);
    }
    std::sort(order.begin(), order.end(), cmp);
    selected_ = std::move(order);
    st.budget_selected = selected_.size();

    // Per-component quota. The global budget depends only on d and |E|, so it
    // cannot see how many density-separated regions exist; once the region
    // count approaches it, some regions get no representative and no bridge
    // can reach them. Top up so every component contributes its own lowest
    // scorers, whether or not any of them fall below the neutral point.
    if (params_.min_per_component && comp_labels_ && comp_count_ > 0 &&
        comp_labels_->size() >= n_) {
        const std::size_t per = params_.min_per_component;
        std::vector<std::uint8_t> chosen(n_, 0);
        for (idx_t v : selected_) chosen[v] = 1;

        // Per-component shortlist of the lowest scorers. `per` is small (a
        // handful), so the linear replacement scan costs O(n * per).
        std::vector<std::vector<idx_t>> buckets(comp_count_);
        for (idx_t v = 0; v < n_; ++v) {
            const idx_t c = (*comp_labels_)[v];
            if (c >= comp_count_) continue;
            std::vector<idx_t>& b = buckets[c];
            if (b.size() < per) {
                b.push_back(v);
            } else {
                std::size_t worst = 0;
                for (std::size_t t = 1; t < b.size(); ++t) {
                    if (cmp(b[worst], b[t])) worst = t;
                }
                if (cmp(v, b[worst])) b[worst] = v;
            }
        }

        std::size_t added = 0, covered = 0;
        for (const std::vector<idx_t>& b : buckets) {
            if (!b.empty()) ++covered;
            for (idx_t v : b) {
                if (!chosen[v]) {
                    chosen[v] = 1;
                    selected_.push_back(v);
                    ++added;
                }
            }
        }
        st.quota_selected = added;
        st.covered_components = covered;
        if (added) std::sort(selected_.begin(), selected_.end(), cmp);
    }

    double mn = 0.0, mx = 0.0, sum = 0.0;
    idx_t tagged = 0;
    if (n_ > 0) {
        mn = score_[0];
        mx = score_[0];
        for (idx_t i = 0; i < n_; ++i) {
            const double s = score_[i];
            mn = std::min(mn, s);
            mx = std::max(mx, s);
            sum += s;
            if (antihub_[i]) ++tagged;
        }
    }
    st.min_score = mn;
    st.max_score = mx;
    st.mean_score = n_ ? sum / double(n_) : 0.0;
    st.tagged_antihubs = tagged;
    st.selected = selected_.size();
    st.selection_cutoff = selected_.empty() ? 0.0 : score_[selected_.back()];
    (void)pool;
    return st;
}

HubnessStats HubnessIndex::refresh(const KnnGraph& g, std::uint32_t dim, std::size_t num_edges,
                                   ThreadPool* pool) {
    const auto t0 = std::chrono::steady_clock::now();
    grow(g.size());

    const std::vector<idx_t> nodes = collect_rescore_set(g);
    apply_scores(g, nodes, pool);

    for (std::size_t s = 0; s < kShards; ++s) {
        ListShard& sh = shards_[s];
        for (idx_t v : sh.dirty) dirty_[v].store(0, std::memory_order_relaxed);
        for (idx_t v : sh.nbr_changed) nbr_changed_[v].store(0, std::memory_order_relaxed);
        sh.dirty.clear();
        sh.nbr_changed.clear();
    }
    dirty_list_.clear();
    nbr_changed_list_.clear();

    HubnessStats st = select(dim, num_edges, nodes.size(), pool);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

HubnessStats HubnessIndex::rescore_all(const KnnGraph& g, std::uint32_t dim,
                                       std::size_t num_edges, ThreadPool* pool) {
    const auto t0 = std::chrono::steady_clock::now();
    grow(g.size());

    std::vector<idx_t> all(n_);
    for (idx_t i = 0; i < n_; ++i) all[i] = i;
    apply_scores(g, all, pool);

    for (std::size_t s = 0; s < kShards; ++s) {
        ListShard& sh = shards_[s];
        for (idx_t v : sh.dirty) dirty_[v].store(0, std::memory_order_relaxed);
        for (idx_t v : sh.nbr_changed) nbr_changed_[v].store(0, std::memory_order_relaxed);
        sh.dirty.clear();
        sh.nbr_changed.clear();
    }
    dirty_list_.clear();
    nbr_changed_list_.clear();

    HubnessStats st = select(dim, num_edges, all.size(), pool);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

std::size_t HubnessIndex::memory_bytes() const {
    std::size_t bytes = std::size_t(n_) * (sizeof(std::atomic<std::int32_t>) + sizeof(double) +
                                           2 * sizeof(std::atomic<std::uint8_t>) +
                                           sizeof(std::uint8_t) + sizeof(std::uint32_t));
    if (rev_enabled_) {
        bytes += rev_.size() * sizeof(std::vector<idx_t>);
        for (const std::vector<idx_t>& r : rev_) bytes += r.capacity() * sizeof(idx_t);
    }
    bytes += selected_.capacity() * sizeof(idx_t);
    return bytes;
}

}  // namespace coresg
