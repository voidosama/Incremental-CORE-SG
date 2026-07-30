#include "coresg/nndescent.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "coresg/thread_pool.hpp"

namespace coresg {
namespace {

inline std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// Cheap per-node stream so that results do not depend on the thread schedule.
struct NodeRng {
    std::uint64_t s;
    explicit NodeRng(std::uint64_t seed, idx_t node, std::uint32_t round) {
        s = seed ^ (0x9E3779B97F4A7C15ull * (node + 1)) ^ (0xC2B2AE3D27D4EB4Full * (round + 1));
        splitmix64(s);
    }
    std::uint32_t next() { return static_cast<std::uint32_t>(splitmix64(s) >> 32); }
    std::uint32_t below(std::uint32_t bound) { return bound ? next() % bound : 0; }
    float unit() { return static_cast<float>(next()) * (1.0f / 4294967296.0f); }
};

template <class Op>
inline void try_pair(KnnGraph* g, idx_t u1, idx_t u2, const float* v1, const float* v2,
                     std::uint32_t stride, std::size_t* evals, std::size_t* updates) {
    if (u1 == u2) return;
    const float r = Op::rank(v1, v2, stride);
    ++(*evals);

    // Unfilled slots hold +inf, so comparing against the last slot covers both
    // "list is full" and "list is still filling" without reading the (non
    // atomic, concurrently mutated) degree counter. A stale read here only
    // costs a lock acquisition; insert_unlocked re-checks under the lock.
    const std::uint32_t k = g->k();
    const bool want1 = r < Op::to_rank(g->dists(u1)[k - 1]);
    const bool want2 = r < Op::to_rank(g->dists(u2)[k - 1]);
    if (!want1 && !want2) return;

    const dist_t d = static_cast<dist_t>(Op::finalize(r));
    if (want1 && g->try_insert_mt(u1, u2, d, true).inserted) ++(*updates);
    if (want2 && g->try_insert_mt(u2, u1, d, true).inserted) ++(*updates);
}

/// Per-thread scratch for one local join.
struct JoinScratch {
    std::vector<idx_t> nnew;
    std::vector<idx_t> nold;
    std::vector<idx_t> drained;
    std::size_t evals = 0;
    std::size_t updates = 0;
};

template <class Op>
void local_join(const DataView& data, KnnGraph* g, ReverseCache* rev, const NNDescentParams& p,
                idx_t node, std::uint32_t round, JoinScratch* sc) {
    const std::uint32_t max_new = p.sample;
    const std::uint32_t max_old = 3u * p.sample;

    sc->nnew.clear();
    sc->nold.clear();

    // --- Sampling (Algorithm 1, lines 6-14) -------------------------------
    // Done under the node's lock: the scan reads the list *and* clears the new
    // flags of the entries it took, so a concurrent insertion must not shift
    // the array in between.
    {
        SpinLocks& locks = g->locks();
        locks.lock(node);
        const std::uint32_t cnt = g->degree(node);
        idx_t* I = g->mutable_ids(node);
        std::uint8_t* F = g->mutable_flags(node);
        if (p.policy == SamplingPolicy::Closest) {
            for (std::uint32_t j = 0; j < cnt; ++j) {
                if (F[j]) {
                    if (sc->nnew.size() < max_new) {
                        sc->nnew.push_back(I[j]);
                        F[j] = 0;  // "mark sampled new items in G[i] as false"
                    }
                } else if (sc->nold.size() < max_old) {
                    sc->nold.push_back(I[j]);
                }
                if (sc->nnew.size() >= max_new && sc->nold.size() >= max_old) break;
            }
        } else {
            NodeRng rng(p.seed, node, round);
            const float pnew = cnt ? std::min(1.0f, float(max_new) / float(cnt)) : 0.f;
            const float pold = cnt ? std::min(1.0f, float(max_old) / float(cnt)) : 0.f;
            for (std::uint32_t j = 0; j < cnt; ++j) {
                if (F[j]) {
                    if (sc->nnew.size() < max_new && rng.unit() < pnew) {
                        sc->nnew.push_back(I[j]);
                        F[j] = 0;
                    }
                } else if (sc->nold.size() < max_old && rng.unit() < pold) {
                    sc->nold.push_back(I[j]);
                }
            }
        }
        locks.unlock(node);
    }

    // --- Publish into the reverse cache, then absorb our own reverse lists --
    const std::size_t sampled_new = sc->nnew.size();
    const std::size_t sampled_old = sc->nold.size();
    for (std::size_t j = 0; j < sampled_old; ++j) rev->push_old(sc->nold[j], node);
    for (std::size_t j = 0; j < sampled_new; ++j) rev->push_new(sc->nnew[j], node);

    rev->drain_old(node, &sc->drained);
    sc->nold.insert(sc->nold.end(), sc->drained.begin(), sc->drained.end());
    rev->drain_new(node, &sc->drained);
    sc->nnew.insert(sc->nnew.end(), sc->drained.begin(), sc->drained.end());

    if (sc->nnew.empty()) return;

    // --- Local join (Algorithm 1, lines 20-24) ----------------------------
    const std::uint32_t stride = data.stride;
    for (std::size_t a = 0; a < sc->nnew.size(); ++a) {
        const idx_t u1 = sc->nnew[a];
        if (u1 >= data.n) continue;
        const float* v1 = data.row(u1);
        for (std::size_t b = a + 1; b < sc->nnew.size(); ++b) {
            const idx_t u2 = sc->nnew[b];
            if (u2 >= data.n) continue;
            try_pair<Op>(g, u1, u2, v1, data.row(u2), stride, &sc->evals, &sc->updates);
        }
        for (std::size_t b = 0; b < sc->nold.size(); ++b) {
            const idx_t u2 = sc->nold[b];
            if (u2 >= data.n) continue;
            try_pair<Op>(g, u1, u2, v1, data.row(u2), stride, &sc->evals, &sc->updates);
        }
    }
}

template <class Op>
NNDescentStats run_descent(const DataView& data, KnnGraph* g, const NNDescentParams& p,
                           ThreadPool* pool, const std::vector<idx_t>* active,
                           ReverseCache* external_rev) {
    const auto t0 = std::chrono::steady_clock::now();
    NNDescentStats stats;

    ReverseCache owned;
    ReverseCache* rev = external_rev;
    if (!rev) {
        owned.reset(g->size(), p.sample);
        rev = &owned;
    } else {
        rev->grow(g->size());
    }

    g->ensure_locks();

    const std::size_t n_active = active ? active->size() : g->size();
    if (n_active == 0) return stats;

    const unsigned nt = pool ? pool->size() : 1u;
    std::vector<JoinScratch> scratch(nt ? nt : 1);

    const double stop_threshold = p.delta * double(p.k) * double(n_active);

    for (std::uint32_t round = 0; round < p.iters; ++round) {
        for (JoinScratch& s : scratch) {
            s.updates = 0;
            s.evals = 0;
        }

        auto body = [&](std::size_t i, int tid) {
            const idx_t node = active ? (*active)[i] : static_cast<idx_t>(i);
            local_join<Op>(data, g, rev, p, node, round, &scratch[tid]);
        };

        if (pool && pool->size() > 1) {
            pool->parallel_for(n_active, body, 64);
        } else {
            for (std::size_t i = 0; i < n_active; ++i) body(i, 0);
        }

        std::size_t round_updates = 0, round_evals = 0;
        for (const JoinScratch& s : scratch) {
            round_updates += s.updates;
            round_evals += s.evals;
        }
        stats.updates += round_updates;
        stats.distance_evals += round_evals;
        stats.rounds = round + 1;

        if (p.verbose) {
            std::printf("[nndescent] round %u/%u updates=%zu evals=%zu\n", round + 1, p.iters,
                        round_updates, round_evals);
        }
        if (double(round_updates) <= stop_threshold) break;
    }

    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return stats;
}

template <class Op>
void run_init_random(const DataView& data, KnnGraph* g, std::uint64_t seed, ThreadPool* pool,
                     const std::vector<idx_t>* nodes) {
    const std::size_t count = nodes ? nodes->size() : g->size();
    const std::uint32_t k = g->k();
    const std::uint32_t stride = data.stride;
    const idx_t n = static_cast<idx_t>(data.n);
    if (n < 2) return;

    auto body = [&](std::size_t i, int) {
        const idx_t node = nodes ? (*nodes)[i] : static_cast<idx_t>(i);
        NodeRng rng(seed, node, 0);
        std::uint32_t needed = k - g->degree(node);
        // A few extra draws to absorb self-hits and duplicates.
        const std::uint32_t attempts = needed + needed / 2 + 4;
        const float* v = data.row(node);
        for (std::uint32_t t = 0; t < attempts && g->degree(node) < k; ++t) {
            const idx_t cand = rng.below(n);
            if (cand == node) continue;
            const dist_t d = static_cast<dist_t>(Op::finalize(Op::rank(v, data.row(cand), stride)));
            g->try_insert_mt(node, cand, d, true);
        }
    };

    if (pool && pool->size() > 1) {
        pool->parallel_for(count, body, 256);
    } else {
        for (std::size_t i = 0; i < count; ++i) body(i, 0);
    }
}

template <class Op>
void run_brute_force(const DataView& data, KnnGraph* g, ThreadPool* pool) {
    const idx_t n = static_cast<idx_t>(data.n);
    const std::uint32_t stride = data.stride;
    auto body = [&](std::size_t i, int) {
        const idx_t node = static_cast<idx_t>(i);
        const float* v = data.row(node);
        for (idx_t j = 0; j < n; ++j) {
            if (j == node) continue;
            const float r = Op::rank(v, data.row(j), stride);
            if (g->degree(node) == g->k() && !(r < Op::to_rank(g->dists(node)[g->k() - 1]))) {
                continue;
            }
            g->try_insert(node, j, static_cast<dist_t>(Op::finalize(r)), false);
        }
    };
    if (pool && pool->size() > 1) {
        pool->parallel_for(n, body, 32);
    } else {
        for (idx_t i = 0; i < n; ++i) body(i, 0);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// ReverseCache
// ---------------------------------------------------------------------------

void ReverseCache::reset(idx_t n, std::uint32_t sample) {
    n_ = n;
    sample_ = sample;
    old_.assign(std::size_t(n) * sample, kInvalidIdx);
    new_.assign(std::size_t(n) * sample, kInvalidIdx);
    old_cnt_.reset(new std::atomic<std::uint32_t>[n ? n : 1]);
    new_cnt_.reset(new std::atomic<std::uint32_t>[n ? n : 1]);
    for (idx_t i = 0; i < n; ++i) {
        old_cnt_[i].store(0, std::memory_order_relaxed);
        new_cnt_[i].store(0, std::memory_order_relaxed);
    }
}

void ReverseCache::grow(idx_t new_n) {
    if (new_n <= n_) return;
    old_.resize(std::size_t(new_n) * sample_, kInvalidIdx);
    new_.resize(std::size_t(new_n) * sample_, kInvalidIdx);
    std::unique_ptr<std::atomic<std::uint32_t>[]> oc(new std::atomic<std::uint32_t>[new_n]);
    std::unique_ptr<std::atomic<std::uint32_t>[]> nc(new std::atomic<std::uint32_t>[new_n]);
    for (idx_t i = 0; i < new_n; ++i) {
        oc[i].store(i < n_ ? old_cnt_[i].load(std::memory_order_relaxed) : 0u,
                    std::memory_order_relaxed);
        nc[i].store(i < n_ ? new_cnt_[i].load(std::memory_order_relaxed) : 0u,
                    std::memory_order_relaxed);
    }
    old_cnt_ = std::move(oc);
    new_cnt_ = std::move(nc);
    n_ = new_n;
}

void ReverseCache::clear() {
    for (idx_t i = 0; i < n_; ++i) {
        old_cnt_[i].store(0, std::memory_order_relaxed);
        new_cnt_[i].store(0, std::memory_order_relaxed);
    }
}

void ReverseCache::push_old(idx_t node, idx_t src) {
    if (node >= n_ || sample_ == 0) return;
    std::atomic<std::uint32_t>& c = old_cnt_[node];
    if (c.load(std::memory_order_relaxed) >= sample_) return;
    const std::uint32_t pos = c.fetch_add(1, std::memory_order_relaxed);
    if (pos >= sample_) {
        c.fetch_sub(1, std::memory_order_relaxed);
        return;
    }
    old_[std::size_t(node) * sample_ + pos] = src;
}

void ReverseCache::push_new(idx_t node, idx_t src) {
    if (node >= n_ || sample_ == 0) return;
    std::atomic<std::uint32_t>& c = new_cnt_[node];
    if (c.load(std::memory_order_relaxed) >= sample_) return;
    const std::uint32_t pos = c.fetch_add(1, std::memory_order_relaxed);
    if (pos >= sample_) {
        c.fetch_sub(1, std::memory_order_relaxed);
        return;
    }
    new_[std::size_t(node) * sample_ + pos] = src;
}

void ReverseCache::drain_old(idx_t node, std::vector<idx_t>* out) {
    out->clear();
    if (node >= n_ || sample_ == 0) return;
    std::uint32_t c = old_cnt_[node].exchange(0, std::memory_order_acq_rel);
    if (c > sample_) c = sample_;
    const std::size_t base = std::size_t(node) * sample_;
    for (std::uint32_t j = 0; j < c; ++j) {
        const idx_t v = old_[base + j];
        if (v != kInvalidIdx) out->push_back(v);
    }
}

void ReverseCache::drain_new(idx_t node, std::vector<idx_t>* out) {
    out->clear();
    if (node >= n_ || sample_ == 0) return;
    std::uint32_t c = new_cnt_[node].exchange(0, std::memory_order_acq_rel);
    if (c > sample_) c = sample_;
    const std::size_t base = std::size_t(node) * sample_;
    for (std::uint32_t j = 0; j < c; ++j) {
        const idx_t v = new_[base + j];
        if (v != kInvalidIdx) out->push_back(v);
    }
}

std::size_t ReverseCache::memory_bytes() const {
    return (old_.capacity() + new_.capacity()) * sizeof(idx_t) +
           std::size_t(n_) * 2 * sizeof(std::atomic<std::uint32_t>);
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

void init_random(const DataView& data, MetricKind metric, KnnGraph* g, std::uint64_t seed,
                 ThreadPool* pool) {
    g->ensure_locks();
    dispatch_metric(metric, [&](auto op) {
        using Op = decltype(op);
        run_init_random<Op>(data, g, seed, pool, nullptr);
        return 0;
    });
}

void init_random_fill(const DataView& data, MetricKind metric, KnnGraph* g,
                      const std::vector<idx_t>& nodes, std::uint64_t seed, ThreadPool* pool) {
    g->ensure_locks();
    dispatch_metric(metric, [&](auto op) {
        using Op = decltype(op);
        run_init_random<Op>(data, g, seed, pool, &nodes);
        return 0;
    });
}

NNDescentStats nndescent(const DataView& data, MetricKind metric, KnnGraph* g,
                         const NNDescentParams& params, ThreadPool* pool,
                         const std::vector<idx_t>* active, ReverseCache* reverse) {
    return dispatch_metric(metric, [&](auto op) {
        using Op = decltype(op);
        return run_descent<Op>(data, g, params, pool, active, reverse);
    });
}

void brute_force_knn(const DataView& data, MetricKind metric, KnnGraph* g, ThreadPool* pool) {
    dispatch_metric(metric, [&](auto op) {
        using Op = decltype(op);
        run_brute_force<Op>(data, g, pool);
        return 0;
    });
}

}  // namespace coresg
