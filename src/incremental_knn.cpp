#include "coresg/incremental_knn.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>

#include "coresg/resource.hpp"
#include "coresg/thread_pool.hpp"

namespace coresg {
namespace {

using Clock = std::chrono::steady_clock;
inline double since(const Clock::time_point& t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

/// Partial Fisher-Yates: the first `count` entries of `pool` become a uniform
/// sample without replacement. O(count) per call, no allocation.
void partial_shuffle(std::vector<idx_t>& pool, std::size_t count, std::mt19937_64& rng) {
    const std::size_t n = pool.size();
    const std::size_t take = std::min(count, n);
    for (std::size_t i = 0; i < take; ++i) {
        std::uniform_int_distribution<std::size_t> d(i, n - 1);
        const std::size_t j = d(rng);
        std::swap(pool[i], pool[j]);
    }
}

/// Visited set for one beam search. Open addressing over a power-of-two table,
/// cleared by bumping a generation counter rather than by rewriting the slots.
///
/// A stamp array of size n per thread would be the obvious choice and is what
/// the first draft used; at 10 M points and 20 threads that is 800 MB of
/// scratch for a structure that never holds more than a few thousand entries.
/// The search visits at most `entries + max_expand * ef * (k + rev_cap)`
/// nodes, so a fixed table sized from that bound is both smaller and faster.
class VisitedSet {
public:
    void reset(std::size_t expected) {
        std::size_t cap = 1024;
        while (cap < expected * 2) cap <<= 1;
        if (cap != mask_ + 1) {
            slots_.assign(cap, 0);
            mask_ = cap - 1;
            epoch_ = 0;
        }
        ++epoch_;
        if (epoch_ == 0) {  // wrapped
            std::fill(slots_.begin(), slots_.end(), std::uint64_t(0));
            epoch_ = 1;
        }
        size_ = 0;
    }

    /// True when `v` was not already present.
    bool insert(idx_t v) {
        // Generation and key live in one word. Split across two arrays this
        // costs two cache misses per probe, and the probe sequence is random
        // by construction -- which measured as the single largest cost in the
        // seeding search, several times the distance computations it guards.
        // +1 so node 0 is distinguishable from an untouched slot.
        const std::uint64_t want = (std::uint64_t(epoch_) << 32) | (std::uint32_t(v) + 1u);
        std::size_t i = (std::size_t(std::uint32_t(v) + 1u) * 0x9E3779B1u) & mask_;
        for (;;) {
            const std::uint64_t s = slots_[i];
            if ((s >> 32) != epoch_) {
                slots_[i] = want;
                ++size_;
                return true;
            }
            if (s == want) return false;
            i = (i + 1) & mask_;
        }
    }

    /// The table degrades badly past ~70 % load; the search stops expanding
    /// when this trips rather than spinning in the probe loop.
    bool full() const { return size_ * 10 >= (mask_ + 1) * 7; }

private:
    std::vector<std::uint64_t> slots_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
    std::uint32_t epoch_ = 0;
};

/// Per-thread scratch for the seeding search.
struct SearchScratch {
    VisitedSet visited;
    std::vector<std::pair<float, idx_t>> frontier;  // min-heap on rank
    std::vector<std::pair<float, idx_t>> best;      // max-heap on rank
    std::size_t evals = 0;
};

/// Hint nodes scanned per query when picking a coarse entry point. Sized so
/// the scan stays a small fraction of the walk it replaces guessing for.
constexpr std::size_t kHintScan = 256;

struct FarFirst {
    bool operator()(const std::pair<float, idx_t>& a, const std::pair<float, idx_t>& b) const {
        return a.first > b.first;
    }
};
struct NearFirst {
    bool operator()(const std::pair<float, idx_t>& a, const std::pair<float, idx_t>& b) const {
        return a.first < b.first;
    }
};

/// Greedy beam search for `q` over the nodes below `limit`, ranked with `Op`'s
/// surrogate. Results land in `sc->best` as (rank, node), unsorted.
template <class Op>
void beam_search(const DataView& data, const KnnGraph& g, const ReverseNeighbours* rev, idx_t q,
                 idx_t limit, std::uint32_t ef, std::uint32_t entries, std::uint32_t max_expand,
                 std::uint32_t rev_cap, const std::vector<idx_t>* hints, std::uint64_t seed,
                 SearchScratch* sc) {
    sc->frontier.clear();
    sc->best.clear();
    if (limit == 0 || ef == 0) return;

    const std::uint32_t k = g.k();
    sc->visited.reset(std::size_t(entries) +
                      std::size_t(max_expand) * ef * (std::size_t(k) + rev_cap) / 4);

    const float* vq = data.row(q);
    const std::uint32_t stride = data.stride;

    std::uint64_t rs = seed ^ (0x9E3779B97F4A7C15ull * (std::uint64_t(q) + 1));
    auto rnd = [&rs]() {
        rs ^= rs << 13;
        rs ^= rs >> 7;
        rs ^= rs << 17;
        return rs;
    };

    auto offer = [&](idx_t u) {
        if (u >= limit || !sc->visited.insert(u)) return;
        const float r = Op::rank(vq, data.row(u), stride);
        ++sc->evals;
        if (sc->best.size() < ef || r < sc->best.front().first) {
            sc->frontier.emplace_back(r, u);
            std::push_heap(sc->frontier.begin(), sc->frontier.end(), FarFirst{});
            sc->best.emplace_back(r, u);
            std::push_heap(sc->best.begin(), sc->best.end(), NearFirst{});
            if (sc->best.size() > ef) {
                std::pop_heap(sc->best.begin(), sc->best.end(), NearFirst{});
                sc->best.pop_back();
            }
        }
    };

    // Entry points. Uniform draws alone are fine on a connected graph and bad
    // on a fragmented one: with 150 separated clusters a random entry starts
    // in the query's own component about 1/150 of the time, and greedy search
    // cannot cross between them.
    //
    // The hint set (CoreSG passes the selected anti-hubs, which carry a
    // per-component quota) fixes that only if the search enters at a hint
    // *near* the query -- drawing hints at random is barely better than
    // drawing points at random. So scan a bounded sample of the hints and
    // enter at the closest. That makes the anti-hub set a one-level coarse
    // index, which is a structure this project already maintains for the
    // bridges and is getting for free here. The scan is capped because the
    // budget is d*ln|E| -- around 2000 nodes at d=128 -- and scanning all of
    // them per query would cost more than the walk it seeds.
    const std::size_t n_hint = hints ? hints->size() : 0;
    if (n_hint) {
        const std::size_t scan = std::min<std::size_t>(n_hint, kHintScan);
        const std::size_t stride_h = std::max<std::size_t>(1, n_hint / scan);
        const std::size_t off = n_hint ? (rnd() % n_hint) : 0;
        float b0 = kInfDist, b1 = kInfDist;
        idx_t e0 = kInvalidIdx, e1 = kInvalidIdx;
        for (std::size_t t = 0; t < scan; ++t) {
            const idx_t h = (*hints)[(off + t * stride_h) % n_hint];
            if (h >= limit) continue;
            const float r = Op::rank(vq, data.row(h), stride);
            ++sc->evals;
            if (r < b0) {
                b1 = b0; e1 = e0;
                b0 = r; e0 = h;
            } else if (r < b1) {
                b1 = r; e1 = h;
            }
        }
        if (e0 != kInvalidIdx) offer(e0);
        if (e1 != kInvalidIdx) offer(e1);
    }
    for (std::uint32_t e = 0; e < entries; ++e) offer(static_cast<idx_t>(rnd() % limit));

    const std::size_t expand_budget = std::size_t(max_expand) * ef;
    std::size_t expanded = 0;
    while (!sc->frontier.empty() && expanded < expand_budget && !sc->visited.full()) {
        std::pop_heap(sc->frontier.begin(), sc->frontier.end(), FarFirst{});
        const std::pair<float, idx_t> cur = sc->frontier.back();
        sc->frontier.pop_back();
        // Everything still queued is further away than the worst kept result,
        // so no unexpanded node can improve the beam.
        if (sc->best.size() >= ef && cur.first > sc->best.front().first) break;
        ++expanded;

        const idx_t* I = g.ids(cur.second);
        const std::uint32_t c = g.degree(cur.second);
        for (std::uint32_t j = 0; j < c; ++j) offer(I[j]);
        if (rev && rev_cap) {
            const std::vector<idx_t>& r = rev->reverse_of(cur.second);
            const std::size_t lim = std::min<std::size_t>(r.size(), rev_cap);
            for (std::size_t j = 0; j < lim; ++j) offer(r[j]);
        }
    }
}

}  // namespace

const char* to_string(InsertStrategy s) {
    switch (s) {
        case InsertStrategy::Resample: return "resample";
        case InsertStrategy::Anchored: return "anchored";
    }
    return "?";
}

bool insert_strategy_from_string(const std::string& s, InsertStrategy* out) {
    if (s == "resample" || s == "uniform") { *out = InsertStrategy::Resample; return true; }
    if (s == "anchored" || s == "search") { *out = InsertStrategy::Anchored; return true; }
    return false;
}

std::vector<IncrementalKnn::Instance> IncrementalKnn::make_instances(idx_t n_base, idx_t first_new,
                                                                    idx_t n_total,
                                                                    std::uint64_t seed) const {
    std::vector<Instance> out;
    const std::size_t n_new = n_total - first_new;

    std::uint32_t s = params_.instances ? params_.instances : 1u;
    if (n_base < params_.min_base_for_split) s = 1;

    std::mt19937_64 rng(seed);
    std::vector<idx_t> base(n_base);
    std::iota(base.begin(), base.end(), idx_t(0));
    std::shuffle(base.begin(), base.end(), rng);

    if (params_.sampling == InstanceSampling::Partition) {
        const std::size_t block = (n_base + s - 1) / (s ? s : 1);
        for (std::uint32_t i = 0; i < s; ++i) {
            const std::size_t lo = std::size_t(i) * block;
            if (lo >= n_base && i > 0) break;
            const std::size_t hi = std::min<std::size_t>(lo + block, n_base);
            Instance inst;
            inst.nodes.assign(base.begin() + lo, base.begin() + hi);

            // A sprinkle of points from the other blocks so that instances
            // are not blind to each other's regions. Draws that land back in
            // this block are removed by the unique() below.
            const std::size_t extra =
                static_cast<std::size_t>(params_.overlap_fraction * double(hi - lo));
            for (std::size_t t = 0; t < extra && n_base > 0; ++t) {
                inst.nodes.push_back(base[rng() % n_base]);
            }
            std::sort(inst.nodes.begin(), inst.nodes.end());
            inst.nodes.erase(std::unique(inst.nodes.begin(), inst.nodes.end()), inst.nodes.end());

            inst.owned_begin = inst.nodes.size();
            for (std::size_t j = 0; j < n_new; ++j) {
                inst.nodes.push_back(static_cast<idx_t>(first_new + j));
            }
            out.push_back(std::move(inst));
        }
    } else {
        double frac = params_.sample_fraction;
        if (!(frac > 0.0)) frac = 1.0 / double(s);
        const std::size_t take = static_cast<std::size_t>(frac * double(n_base));
        for (std::uint32_t i = 0; i < s; ++i) {
            partial_shuffle(base, take, rng);
            Instance inst;
            inst.nodes.assign(base.begin(), base.begin() + std::min(take, base.size()));
            std::sort(inst.nodes.begin(), inst.nodes.end());
            inst.owned_begin = inst.nodes.size();
            for (std::size_t j = 0; j < n_new; ++j) {
                inst.nodes.push_back(static_cast<idx_t>(first_new + j));
            }
            out.push_back(std::move(inst));
        }
    }
    return out;
}

std::vector<IncrementalKnn::Instance> IncrementalKnn::make_build_instances(
    idx_t n_total, std::uint64_t seed) const {
    std::vector<Instance> out;
    std::uint32_t s = params_.instances ? params_.instances : 1u;
    if (n_total < params_.min_base_for_split) s = 1;

    if (s == 1) {
        Instance inst;
        inst.nodes.resize(n_total);
        std::iota(inst.nodes.begin(), inst.nodes.end(), idx_t(0));
        inst.owned_begin = 0;
        out.push_back(std::move(inst));
        return out;
    }

    std::mt19937_64 rng(seed);
    std::vector<idx_t> perm(n_total);
    std::iota(perm.begin(), perm.end(), idx_t(0));
    std::shuffle(perm.begin(), perm.end(), rng);

    const std::size_t block = (n_total + s - 1) / s;
    for (std::uint32_t i = 0; i < s; ++i) {
        const std::size_t lo = std::size_t(i) * block;
        if (lo >= n_total) break;
        const std::size_t hi = std::min<std::size_t>(lo + block, n_total);
        Instance inst;
        inst.nodes.assign(perm.begin() + lo, perm.begin() + hi);
        const std::size_t extra =
            static_cast<std::size_t>(params_.overlap_fraction * double(hi - lo));
        for (std::size_t t = 0; t < extra; ++t) inst.nodes.push_back(perm[rng() % n_total]);
        std::sort(inst.nodes.begin(), inst.nodes.end());
        inst.nodes.erase(std::unique(inst.nodes.begin(), inst.nodes.end()), inst.nodes.end());
        inst.owned_begin = 0;  // every node in a build instance is owned
        out.push_back(std::move(inst));
    }
    return out;
}

void IncrementalKnn::search_seeds(const Dataset& data, const KnnGraph& graph, idx_t first_new,
                                  idx_t n_total, const ReverseNeighbours* reverse,
                                  const std::vector<idx_t>* entry_hints, std::vector<Seed>* out,
                                  IncrementalKnnStats* stats) const {
    const std::size_t n_new = n_total - first_new;
    out->assign(n_new, Seed{});
    if (first_new == 0) return;

    const std::uint32_t ef = params_.search_ef ? params_.search_ef : 2u * graph.k();
    const std::uint32_t entries = params_.search_entries ? params_.search_entries : 1u;
    const std::uint32_t max_expand = params_.search_max_expand ? params_.search_max_expand : 1u;
    const std::uint32_t rev_cap = params_.search_reverse_cap;
    const ReverseNeighbours* rev =
        (reverse && rev_cap && reverse->reverse_available()) ? reverse : nullptr;
    if (stats) stats->used_reverse_edges = rev != nullptr;

    const unsigned nt = (pool_ && pool_->size() > 1) ? pool_->size() : 1u;
    std::vector<SearchScratch> scratch(nt);

    dispatch_metric(data.metric(), [&](auto op) {
        using Op = decltype(op);
        auto body = [&](std::size_t i, int tid) {
            SearchScratch& sc = scratch[tid];
            const idx_t q = first_new + static_cast<idx_t>(i);
            beam_search<Op>(data.view(), graph, rev, q, first_new, ef, entries, max_expand,
                            rev_cap, entry_hints, params_.seed, &sc);
            Seed& s = (*out)[i];
            s.node = q;
            s.best.reserve(sc.best.size());
            for (const std::pair<float, idx_t>& p : sc.best) {
                s.best.emplace_back(static_cast<dist_t>(Op::finalize(p.first)), p.second);
            }
            // Ascending, so applying the seed fills the sorted list in order
            // and the closure below can take a prefix as "the nearest ones".
            std::sort(s.best.begin(), s.best.end());
        };
        if (nt > 1) {
            pool_->parallel_for(n_new, body, 64);
        } else {
            for (std::size_t i = 0; i < n_new; ++i) body(i, 0);
        }
        return 0;
    });

    if (stats) {
        for (const SearchScratch& sc : scratch) stats->search_evals += sc.evals;
    }
}

std::vector<IncrementalKnn::Instance> IncrementalKnn::make_anchored_instances(
    const KnnGraph& graph, const std::vector<Seed>& seeds, idx_t first_new, idx_t n_total,
    std::uint64_t seed, IncrementalKnnStats* stats) const {
    std::vector<Instance> out;
    const std::size_t n_new = n_total - first_new;
    if (n_new == 0) return out;

    std::uint32_t s = params_.instances ? params_.instances : 1u;
    if (n_new < 2 * std::size_t(s)) s = 1;

    // ---- context: the closure of what the search found --------------------
    std::size_t budget = first_new;
    if (params_.context_per_new > 0.0) {
        budget = static_cast<std::size_t>(params_.context_per_new * double(n_new));
    }
    if (params_.context_max_nodes && params_.context_max_nodes < budget) {
        budget = params_.context_max_nodes;
    }
    if (budget > first_new) budget = first_new;

    std::vector<std::uint8_t> mark(first_new, 0);
    std::vector<idx_t> context;
    context.reserve(std::min<std::size_t>(budget, n_new * (graph.k() + 1)) + 1);

    // Ring 0: the seeds themselves, nearest first, one pass per rank so that
    // every new point contributes its closest anchor before any point
    // contributes its second. Otherwise a truncated budget is spent entirely
    // on the first few new points.
    std::size_t max_len = 0;
    for (const Seed& sd : seeds) max_len = std::max(max_len, sd.best.size());
    for (std::size_t r = 0; r < max_len && context.size() < budget; ++r) {
        for (const Seed& sd : seeds) {
            if (context.size() >= budget) break;
            if (r >= sd.best.size()) continue;
            const idx_t v = sd.best[r].second;
            if (v < first_new && !mark[v]) {
                mark[v] = 1;
                context.push_back(v);
            }
        }
    }
    const std::size_t ring0 = context.size();

    // Ring 1: the anchors' own neighbours. This is what makes the context a
    // coherent region instead of a set of isolated points -- and it is why a
    // context point keeps most of its real neighbours inside the instance,
    // which the uniform strategy cannot do.
    for (std::size_t t = 0; t < ring0 && context.size() < budget; ++t) {
        const idx_t v = context[t];
        const idx_t* I = graph.ids(v);
        const std::uint32_t c = graph.degree(v);
        for (std::uint32_t j = 0; j < c && context.size() < budget; ++j) {
            const idx_t u = I[j];
            if (u < first_new && !mark[u]) {
                mark[u] = 1;
                context.push_back(u);
            }
        }
    }

    if (stats) {
        stats->context_nodes = context.size();
        stats->context_capped = context.size() >= budget && budget < std::size_t(first_new);
    }

    // ---- split: the *context* is divided, the batch is not -----------------
    //
    // Every instance carries the whole batch. Splitting the new points across
    // instances looks like the obvious way to parallelise and is wrong: the
    // batch is its own k-NN problem, and an instance holding 1/s of it can
    // only ever discover 1/s of the intra-batch edges. Measured on a 20 %
    // batch, splitting cost 8 points of new-point recall at s=2 and 15 at s=6
    // -- and made the instance count an accuracy knob, which it must not be.
    //
    // Keeping the batch whole costs s * |batch| instance nodes rather than
    // |batch|. That is still O(|batch|): the term this strategy exists to
    // remove is n_base, and it is the *context* that used to carry it.
    std::mt19937_64 rng(seed);
    std::shuffle(context.begin(), context.end(), rng);

    const std::size_t cblock = (context.size() + s - 1) / s;

    for (std::uint32_t i = 0; i < s; ++i) {
        const std::size_t clo = std::min(std::size_t(i) * cblock, context.size());
        const std::size_t chi = std::min(clo + cblock, context.size());
        if (clo >= chi && i > 0) break;

        Instance inst;
        inst.nodes.assign(context.begin() + clo, context.begin() + chi);

        // A sprinkle from the rest of the context so the blocks are not blind
        // to each other's regions during the merge.
        const std::size_t extra =
            static_cast<std::size_t>(params_.overlap_fraction * double(chi - clo));
        for (std::size_t t = 0; t < extra && !context.empty(); ++t) {
            inst.nodes.push_back(context[rng() % context.size()]);
        }
        std::sort(inst.nodes.begin(), inst.nodes.end());
        inst.nodes.erase(std::unique(inst.nodes.begin(), inst.nodes.end()), inst.nodes.end());
        inst.owned_begin = inst.nodes.size();

        for (std::size_t j = 0; j < n_new; ++j) {
            inst.nodes.push_back(static_cast<idx_t>(first_new + j));
        }
        out.push_back(std::move(inst));
    }
    return out;
}

void IncrementalKnn::stage_job(const Dataset& data, const KnnGraph& graph, const Instance& inst,
                               const NNDescentParams& nnd, std::vector<idx_t>* g2l,
                               KnnJob* job) const {
    const std::size_t m = inst.nodes.size();
    const std::uint32_t stride = data.stride();
    job->dim = data.dim();
    job->stride = stride;
    job->local_n = m;
    job->metric = data.metric();
    job->params = nnd;
    job->params.k = graph.k();

    // The descent stops when a round produces fewer than `delta * k * m`
    // updates. That rule assumes every node still has work to do, which is
    // true of a cold build and false of an anchored instance: its context
    // arrives already converged from the global graph and contributes almost
    // no updates, so a threshold scaled to `m` fires while the owned nodes --
    // the only reason the instance exists -- are still moving. Measured on
    // 8000 base + 2000 new: new-point recall 0.76 against 0.92 for the same
    // work split differently, purely from stopping early.
    //
    // Rescaling delta by the owned share expresses the threshold in terms of
    // the nodes that can actually improve, and it does so as a scalar, so the
    // CUDA path inherits it without any extra plumbing.
    if (inst.owned_begin > 0 && inst.owned_begin < m) {
        const double owned = double(m - inst.owned_begin);
        job->params.delta = nnd.delta * owned / double(m);
    }
    job->graph.reset(static_cast<idx_t>(m), graph.k());

    // The gather is a straight memcpy per row and the warm start touches m*k
    // scattered neighbourhoods; both were serial and outside every timer,
    // which on a large instance is a sizeable chunk of an update spent on one
    // core. Rows are disjoint, so both parallelise directly.
    job->vectors.assign(m * stride, 0.0f);
    auto gather = [&](std::size_t lo, std::size_t hi, int) {
        for (std::size_t i = lo; i < hi; ++i) {
            std::memcpy(job->vectors.data() + i * stride, data.row(inst.nodes[i]),
                        stride * sizeof(float));
        }
    };
    auto fill = [&](std::size_t lo, std::size_t hi, int) {
        for (std::size_t i = lo; i < hi; ++i) (*g2l)[inst.nodes[i]] = static_cast<idx_t>(i);
    };
    auto clear = [&](std::size_t lo, std::size_t hi, int) {
        for (std::size_t i = lo; i < hi; ++i) (*g2l)[inst.nodes[i]] = kInvalidIdx;
    };
    // Inherit the edges whose other endpoint also made it into this instance.
    // Flagged "new" so the very first local join uses them. Row i is written
    // by exactly one worker, so the unlocked insert stays correct.
    // Owned nodes keep only a prefix of their inherited list, so that the
    // backend's random fill has somewhere to put the other new points. See
    // `owned_warm_fraction`.
    const std::uint32_t k = graph.k();
    std::uint32_t owned_keep = k;
    if (inst.owned_begin > 0 && inst.owned_begin < m) {
        const double f = params_.owned_warm_fraction;
        owned_keep = (f >= 1.0) ? k : static_cast<std::uint32_t>(f * double(k));
    }
    auto warm = [&](std::size_t lo, std::size_t hi, int) {
        for (std::size_t i = lo; i < hi; ++i) {
            const idx_t gid = inst.nodes[i];
            const idx_t* I = graph.ids(gid);
            const dist_t* D = graph.dists(gid);
            const std::uint32_t c = graph.degree(gid);
            const std::uint32_t budget = (i >= inst.owned_begin) ? owned_keep : k;
            std::uint32_t taken = 0;
            for (std::uint32_t j = 0; j < c && taken < budget; ++j) {
                const idx_t lj = (*g2l)[I[j]];
                if (lj != kInvalidIdx) {
                    job->graph.try_insert(static_cast<idx_t>(i), lj, D[j], true);
                    ++taken;
                }
            }
        }
    };

    const bool par = pool_ && pool_->size() > 1 && m >= 4096;
    if (par) {
        pool_->parallel_ranges(m, gather);
        pool_->parallel_ranges(m, fill);
        if (params_.warm_start) pool_->parallel_ranges(m, warm);
        pool_->parallel_ranges(m, clear);
    } else {
        gather(0, m, 0);
        fill(0, m, 0);
        if (params_.warm_start) warm(0, m, 0);
        clear(0, m, 0);
    }
}

void IncrementalKnn::stage_all(const Dataset& data, const KnnGraph& graph,
                               const std::vector<Instance>& instances, const NNDescentParams& nnd,
                               idx_t n_total, std::vector<KnnJob>* jobs,
                               IncrementalKnnStats* stats) const {
    const auto t0 = Clock::now();
    // One shared scratch map, not one per worker: at 10 M points a per-thread
    // buffer would be tens of gigabytes. Instances are staged one at a time and
    // each leaves the map all-invalid, so sharing it is safe and the
    // parallelism lives inside stage_job instead.
    std::vector<idx_t> g2l(n_total, kInvalidIdx);
    for (std::size_t i = 0; i < instances.size(); ++i) {
        stage_job(data, graph, instances[i], nnd, &g2l, &(*jobs)[i]);
        (*jobs)[i].job_id = static_cast<int>(i);
        stats->instance_nodes += instances[i].nodes.size();
        // The biggest instance is what has to fit on a device; record it so a
        // capacity claim rests on a measured figure rather than an estimate.
        if (instances[i].nodes.size() > stats->largest_instance) {
            stats->largest_instance = instances[i].nodes.size();
            stats->largest_instance_bytes = instance_device_bytes(
                instances[i].nodes.size(), data.stride(), graph.k(), nnd.sample);
        }
    }
    stats->stage_seconds = since(t0);
}

std::size_t IncrementalKnn::merge_job(const KnnJob& job, const Instance& inst, idx_t first_new,
                                      KnnGraph* graph, std::size_t* candidates) const {
    const std::size_t m = inst.nodes.size();
    const unsigned nt = (pool_ && pool_->size() > 1) ? pool_->size() : 1u;
    std::vector<std::size_t> applied(nt, 0);
    std::vector<std::size_t> offered(nt, 0);

    auto body = [&](std::size_t lo, std::size_t hi, int slot) {
        std::size_t a = 0, o = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            const idx_t gid = inst.nodes[i];
            const idx_t* I = job.graph.ids(static_cast<idx_t>(i));
            const dist_t* D = job.graph.dists(static_cast<idx_t>(i));
            const std::uint32_t c = job.graph.degree(static_cast<idx_t>(i));
            const bool gid_is_new = gid >= first_new;
            for (std::uint32_t j = 0; j < c; ++j) {
                const idx_t gj = inst.nodes[I[j]];
                const bool touches_new = gid_is_new || gj >= first_new;
                if (!touches_new && !params_.apply_base_base) continue;
                ++o;
                // Symmetric: an instance edge improves both endpoints' lists.
                if (graph->try_insert_mt(gid, gj, D[j], true).inserted) ++a;
                if (graph->try_insert_mt(gj, gid, D[j], true).inserted) ++a;
            }
        }
        applied[slot] += a;
        offered[slot] += o;
    };

    if (nt > 1) {
        pool_->parallel_ranges(m, body);
    } else {
        body(0, m, 0);
    }

    std::size_t a_total = 0, o_total = 0;
    for (unsigned t = 0; t < nt; ++t) {
        a_total += applied[t];
        o_total += offered[t];
    }
    if (candidates) *candidates += o_total;
    return a_total;
}

std::vector<idx_t> IncrementalKnn::refinement_set(const KnnGraph& graph, idx_t first_new,
                                                  idx_t n_total,
                                                  IncrementalKnnStats* stats) const {
    // Ceiling on the work a batch may do. Rings are appended in order, so
    // truncating the tail drops the least relevant nodes first and the new
    // points always survive.
    std::size_t cap = n_total;
    if (params_.refine_max_fraction > 0.0 && params_.refine_max_fraction < 1.0) {
        cap = static_cast<std::size_t>(params_.refine_max_fraction * double(n_total));
    }
    if (params_.refine_max_nodes && params_.refine_max_nodes < cap) {
        cap = params_.refine_max_nodes;
    }
    const std::size_t n_new = n_total - first_new;
    // The per-new-point form is the only one that does not grow with n. It is
    // applied under Anchored because a Theta(n) refinement would put back the
    // very term the strategy exists to remove.
    if (params_.strategy == InsertStrategy::Anchored && params_.refine_per_new > 0.0) {
        const std::size_t per = static_cast<std::size_t>(params_.refine_per_new * double(n_new));
        if (per < cap) cap = per;
    }
    if (cap < n_new) cap = n_new;  // never drop a new point

    std::vector<std::uint8_t> mark(n_total, 0);
    std::vector<idx_t> active;
    active.reserve(std::min<std::size_t>(cap, std::size_t(n_new) * (graph.k() + 1)) + 1);

    for (idx_t v = first_new; v < n_total; ++v) {
        if (!mark[v]) {
            mark[v] = 1;
            active.push_back(v);
        }
    }
    const std::size_t ring0 = active.size();

    auto expand = [&](std::size_t lo, std::size_t hi) {
        for (std::size_t t = lo; t < hi && active.size() < cap; ++t) {
            const idx_t v = active[t];
            const idx_t* I = graph.ids(v);
            const std::uint32_t c = graph.degree(v);
            for (std::uint32_t j = 0; j < c && active.size() < cap; ++j) {
                const idx_t q = I[j];
                if (q < n_total && !mark[q]) {
                    mark[q] = 1;
                    active.push_back(q);
                }
            }
        }
    };

    expand(0, ring0);
    const std::size_t ring1 = active.size();
    if (params_.refine_second_ring) expand(ring0, ring1);

    if (stats) {
        stats->refine_ring0 = ring0;
        stats->refine_ring1 = ring1 - ring0;
        stats->refine_capped = active.size() >= cap && cap < std::size_t(n_total);
    }
    return active;
}

IncrementalKnnStats IncrementalKnn::build(const Dataset& data, KnnGraph* graph) {
    IncrementalKnnStats st;
    const idx_t n = static_cast<idx_t>(data.size());
    st.new_points = n;
    if (n == 0) return st;

    graph->grow(n);
    graph->ensure_locks();

    if (params_.brute_force_limit && n <= params_.brute_force_limit) {
        const auto t0 = Clock::now();
        brute_force_knn(data.view(), data.metric(), graph, pool_);
        st.instances = 1;
        st.instance_nodes = n;
        st.instance_seconds = since(t0);
        for (idx_t v = 0; v < n; ++v) st.applied_edges += graph->degree(v);
        st.candidate_edges = st.applied_edges;
        return st;
    }

    std::vector<Instance> instances = make_build_instances(n, params_.seed);
    st.instances = instances.size();

    if (instances.size() == 1) {
        // Nothing to partition: one descent over the whole set. Still goes
        // through the backend so the GPU path is used when available.
        std::vector<KnnJob> jobs(1);
        KnnJob& job = jobs[0];
        job.vectors.assign(data.data(), data.data() + std::size_t(n) * data.stride());
        job.dim = data.dim();
        job.stride = data.stride();
        job.local_n = n;
        job.metric = data.metric();
        job.params = params_.instance_nnd;
        job.params.k = graph->k();
        job.graph.reset(n, graph->k());
        job.job_id = 0;

        const auto t0 = Clock::now();
        JobScheduler sched(backend_);
        sched.run_all(&jobs);
        st.instance_seconds = since(t0);
        st.per_instance.push_back(job.stats);
        // The single-instance fast path bypasses the multi-instance loop, so
        // it has to account for itself or every un-partitioned run reports
        // "0 jobs on device" and looks like a silent host demotion.
        if (job.ran_on == DeviceKind::Cuda) ++st.jobs_on_device; else ++st.jobs_on_host;
        if (job.out_of_memory) ++st.jobs_out_of_memory;
        st.largest_instance = n;
        st.largest_instance_bytes = instance_device_bytes(
            n, data.stride(), graph->k(), params_.instance_nnd.sample);
        st.instance_nodes = n;

        // Adopt the result wholesale: same node ids, same k.
        graph->ids_storage() = job.graph.ids_storage();
        graph->dist_storage() = job.graph.dist_storage();
        graph->flag_storage() = job.graph.flag_storage();
        graph->count_storage() = job.graph.count_storage();
        for (idx_t v = 0; v < n; ++v) st.applied_edges += graph->degree(v);
        st.candidate_edges = st.applied_edges;
        return st;
    }

    // Partitioned build: independent blocks, merge, then a global refinement.
    std::vector<KnnJob> jobs(instances.size());
    stage_all(data, *graph, instances, params_.instance_nnd, n, &jobs, &st);

    const auto t_inst = Clock::now();
    {
        JobScheduler sched(backend_);
        sched.run_all(&jobs);
    }
    st.instance_seconds = since(t_inst);
    for (const KnnJob& j : jobs) {
        st.per_instance.push_back(j.stats);
        if (j.ran_on == DeviceKind::Cuda) ++st.jobs_on_device; else ++st.jobs_on_host;
        if (j.out_of_memory) ++st.jobs_out_of_memory;
    }

    const auto t_merge = Clock::now();
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        st.applied_edges += merge_job(jobs[i], instances[i], /*first_new=*/0, graph,
                                      &st.candidate_edges);
        std::vector<float>().swap(jobs[i].vectors);
    }
    st.merge_seconds = since(t_merge);

    const auto t_ref = Clock::now();
    NNDescentParams rp = params_.refine_nnd;
    rp.k = graph->k();
    st.refine_stats = nndescent(data.view(), data.metric(), graph, rp, pool_);
    st.refine_seconds = since(t_ref);
    st.refine_nodes = n;
    return st;
}

IncrementalKnnStats IncrementalKnn::insert(const Dataset& data, KnnGraph* graph, idx_t first_new,
                                           const ReverseNeighbours* reverse,
                                           const std::vector<idx_t>* entry_hints) {
    IncrementalKnnStats st;
    const idx_t n_total = static_cast<idx_t>(data.size());
    if (n_total <= first_new) return st;

    st.base_points = first_new;
    st.new_points = n_total - first_new;

    graph->grow(n_total);
    graph->ensure_locks();

    if (params_.brute_force_limit && n_total <= params_.brute_force_limit) {
        // Exact reference path: recompute every neighbourhood. The existing
        // lists are kept and simply improved, so the observer still sees the
        // individual edge changes and the hubness index stays incremental.
        const auto t0 = Clock::now();
        brute_force_knn(data.view(), data.metric(), graph, pool_);
        st.instances = 1;
        st.instance_nodes = n_total;
        st.instance_seconds = since(t0);
        st.refine_nodes = n_total;
        return st;
    }

    std::vector<Instance> instances;
    if (params_.strategy == InsertStrategy::Anchored && first_new > 0) {
        // Phase 1: read only. Nothing may be written to the graph until every
        // search has finished -- the reverse table being navigated is mutated
        // by the observer on each insertion, and a search that saw a partially
        // updated graph would also make the result depend on scheduling.
        const auto t_search = Clock::now();
        std::vector<Seed> seeds;
        search_seeds(data, *graph, first_new, n_total, reverse, entry_hints, &seeds, &st);
        instances = make_anchored_instances(*graph, seeds, first_new, n_total, params_.seed, &st);

        // Phase 2: apply. The seed neighbourhood is the new point's starting
        // list, and the symmetric insert is what lets an existing point learn
        // about an arrival that belongs in its own list.
        for (const Seed& sd : seeds) {
            for (const std::pair<dist_t, idx_t>& p : sd.best) {
                graph->try_insert_mt(sd.node, p.second, p.first, true);
                graph->try_insert_mt(p.second, sd.node, p.first, true);
            }
        }
        st.search_seconds = since(t_search);
    } else {
        instances = make_instances(first_new, first_new, n_total, params_.seed);
    }
    st.instances = instances.size();

    std::vector<KnnJob> jobs(instances.size());
    stage_all(data, *graph, instances, params_.instance_nnd, n_total, &jobs, &st);

    const auto t_inst = Clock::now();
    {
        JobScheduler sched(backend_);
        sched.run_all(&jobs);
    }
    st.instance_seconds = since(t_inst);
    for (const KnnJob& j : jobs) {
        st.per_instance.push_back(j.stats);
        if (j.ran_on == DeviceKind::Cuda) ++st.jobs_on_device; else ++st.jobs_on_host;
        if (j.out_of_memory) ++st.jobs_out_of_memory;
    }

    // Merge: each instance offers the neighbourhoods it found. Because the
    // global lists are sorted top-k, "collect the neighbourhoods of each new
    // point and keep the best k" is exactly repeated insertion.
    const auto t_merge = Clock::now();
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        st.applied_edges += merge_job(jobs[i], instances[i], first_new, graph,
                                      &st.candidate_edges);
        std::vector<float>().swap(jobs[i].vectors);
    }
    st.merge_seconds = since(t_merge);

    // Restricted refinement over the new points and their neighbourhoods.
    const auto t_ref = Clock::now();
    std::vector<idx_t> active = refinement_set(*graph, first_new, n_total, &st);
    st.refine_nodes = active.size();
    NNDescentParams rp = params_.refine_nnd;
    rp.k = graph->k();
    st.refine_stats = nndescent(data.view(), data.metric(), graph, rp, pool_, &active);
    st.refine_seconds = since(t_ref);

    if (params_.verbose) {
        std::printf(
            "[incremental] +%zu points over %zu instances (%s, %zu instance nodes): search "
            "%.3fs, stage %.3fs, instances %.3fs, merge %.3fs, refine %.3fs (%zu active "
            "nodes)\n",
            st.new_points, st.instances, to_string(params_.strategy), st.instance_nodes,
            st.search_seconds, st.stage_seconds, st.instance_seconds, st.merge_seconds,
            st.refine_seconds, st.refine_nodes);
    }
    return st;
}

}  // namespace coresg
