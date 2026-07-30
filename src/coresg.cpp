#include "coresg/coresg.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "coresg/resource.hpp"

namespace coresg {
namespace {

using Clock = std::chrono::steady_clock;
inline double since(const Clock::time_point& t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

}  // namespace

const char* to_string(MissingEdgeMode m) {
    switch (m) {
        case MissingEdgeMode::AntiHub: return "antihub";
        case MissingEdgeMode::ExactMst: return "exact";
        case MissingEdgeMode::None: return "none";
    }
    return "?";
}

bool missing_edge_mode_from_string(const std::string& s, MissingEdgeMode* out) {
    if (s == "antihub") { *out = MissingEdgeMode::AntiHub; return true; }
    if (s == "exact") { *out = MissingEdgeMode::ExactMst; return true; }
    if (s == "none") { *out = MissingEdgeMode::None; return true; }
    return false;
}

CoreSG::CoreSG(const CoreSGParams& params) : params_(params) {
    if (!metric_is_dissimilarity(params_.metric)) {
        throw std::runtime_error(
            "CoreSG: mutual reachability requires a non-negative dissimilarity; "
            "inner product is not supported");
    }
    pool_.reset(new ThreadPool(params_.threads));
    BackendOptions bo = params_.backend;
    if (bo.cpu_threads == 0) bo.cpu_threads = pool_->size();
    backend_ = make_backend(bo, pool_.get());

    params_.incr.instance_nnd.k = params_.k_max;
    params_.incr.refine_nnd.k = params_.k_max;
    params_.hub.k_for_score = params_.k_max;

    incr_.reset(new IncrementalKnn(params_.incr, backend_.get(), pool_.get()));

    if (params_.verbose) {
        std::printf("[coresg] backend=%s devices:\n", backend_->name().c_str());
        for (const DeviceInfo& d : backend_->devices()) {
            std::printf("[coresg]   %s\n", describe(d).c_str());
        }
    }
}

CoreSG::~CoreSG() {
    // Detach first: the observer must not outlive the index it writes into.
    hub_.detach(&knn_);
}

std::vector<dist_t> CoreSG::core_distances(std::uint32_t mpts) const {
    return knn_.core_distances(std::min(mpts, params_.k_max), pool_.get());
}

IntrinsicDimStats CoreSG::intrinsic_dimension() const {
    return estimate_intrinsic_dimension(knn_, params_.intrinsic_dim_k, pool_.get(),
                                        data_.dim());
}

std::uint32_t CoreSG::budget_dim(CoreSGStats* stats) const {
    if (params_.budget_dimension == BudgetDimension::Ambient) {
        if (stats) stats->effective_dim = double(data_.dim());
        return data_.dim();
    }
    const IntrinsicDimStats id = intrinsic_dimension();
    // Fall back to ambient when the neighbourhoods carry no usable scale --
    // duplicate-heavy sparse data leaves most points without an estimate, and
    // a number derived from the surviving minority is not worth trusting.
    double d = id.reliable() ? id.value() : double(data_.dim());
    if (!(d > 0.0)) d = double(data_.dim());
    d = std::min(d, double(data_.dim()));
    d = std::max(d, 1.0);
    if (stats) {
        stats->id_stats = id;
        stats->effective_dim = d;
    }
    return static_cast<std::uint32_t>(std::llround(d));
}

std::vector<Edge> CoreSG::mst_over_subset(const std::vector<idx_t>& ids) const {
    const std::size_t m = ids.size();
    std::vector<Edge> out;
    if (m < 2) return out;

    const std::vector<float> pts = data_.gather(ids);
    std::vector<float> core(m);
    for (std::size_t i = 0; i < m; ++i) core[i] = knn_.core_dist(ids[i], params_.k_max);

    JobScheduler sched(backend_.get());
    MstResult local;

    if (m <= params_.dense_matrix_limit) {
        // Small enough to materialise: one multi-device pass over the rows,
        // then a dense Prim on the host.
        std::vector<float> matrix(m * m);
        sched.mrd_matrix(pts.data(), m, data_.dim(), data_.stride(), data_.metric(), core.data(),
                         matrix.data());
        local = mst_dense(matrix.data(), m);
    } else {
        // Stream one row per Prim step: O(m) memory instead of O(m^2).
        std::vector<float> row(m);
        Backend* be = backend_.get();
        const float* p = pts.data();
        const float* c = core.data();
        const std::uint32_t dim = data_.dim();
        const std::uint32_t stride = data_.stride();
        const MetricKind metric = data_.metric();
        local = mst_prim_virtual(static_cast<idx_t>(m), [&](idx_t i, float* dst) {
            be->mrd_block(0, p, m, dim, stride, metric, c, i, i + 1, dst);
        });
    }

    // The tree is minimal in mutual reachability space (that is what an
    // MST_kmax means), but the edges are *stored* with their base distance so
    // that edges(mpts) can reweight them for any smaller mpts exactly the way
    // it reweights the k-NN half.
    out.reserve(local.edges.size());
    for (const Edge& e : local.edges) {
        const dist_t base = metric_distance(data_.metric(), pts.data() + e.u * data_.stride(),
                                            pts.data() + e.v * data_.stride(), data_.stride());
        out.emplace_back(ids[e.u], ids[e.v], base);
    }
    return out;
}

void CoreSG::recompute_missing_edges(CoreSGStats* stats) {
    const auto t0 = Clock::now();
    missing_.clear();
    repair_count_ = 0;

    switch (params_.missing_mode) {
        case MissingEdgeMode::None:
            break;
        case MissingEdgeMode::AntiHub: {
            const std::vector<idx_t>& sel = hub_.selected();
            stats->antihubs_selected = sel.size();
            missing_ = mst_over_subset(sel);
            break;
        }
        case MissingEdgeMode::ExactMst: {
            MstResult r = exact_mst(params_.k_max);
            missing_ = std::move(r.edges);
            for (Edge& e : missing_) {
                e.w = metric_distance(data_.metric(), data_.row(e.u), data_.row(e.v),
                                      data_.stride());
            }
            break;
        }
    }
    stats->missing_edges = missing_.size();
    stats->missing_seconds = since(t0);
}

void CoreSG::repair_connectivity(CoreSGStats* stats) {
    const auto t0 = Clock::now();
    const idx_t n = knn_.size();
    if (n == 0) return;

    // Components of the k-NN half come from the tracker (maintained across
    // batches); the bridges are folded in on top, which is O(|bridges|)
    // rather than another sweep over n*k edges.
    if (!conn_.valid()) stats->conn_stats = conn_.rebuild(knn_, pool_.get());
    idx_t ncomp = 0;
    std::vector<idx_t> comp = conn_.labels_with(missing_, &ncomp);

    stats->components_before_repair = ncomp;
    stats->components_after_repair = ncomp;
    if (ncomp <= 1 || !params_.ensure_connected) {
        stats->repair_seconds = since(t0);
        return;
    }

    // Representatives: the lowest-scoring (most anti-hub) members of each
    // component, which are the points a bridge is most likely to leave from.
    std::size_t per = params_.reps_per_component ? params_.reps_per_component : 1;
    if (params_.max_reps && std::size_t(ncomp) * per > params_.max_reps) {
        per = std::max<std::size_t>(1, params_.max_reps / std::size_t(ncomp));
    }

    std::vector<std::vector<idx_t>> buckets(ncomp);
    for (idx_t v = 0; v < n; ++v) {
        std::vector<idx_t>& b = buckets[comp[v]];
        if (b.size() < per) {
            b.push_back(v);
        } else {
            // Replace the current worst (highest score) if this one is lower.
            std::size_t worst = 0;
            for (std::size_t t = 1; t < b.size(); ++t) {
                if (hub_.reverse_score(b[t]) > hub_.reverse_score(b[worst])) worst = t;
            }
            if (hub_.reverse_score(v) < hub_.reverse_score(b[worst])) b[worst] = v;
        }
    }

    std::vector<idx_t> reps;
    for (const std::vector<idx_t>& b : buckets) reps.insert(reps.end(), b.begin(), b.end());
    if (reps.size() < 2) {
        stats->repair_seconds = since(t0);
        return;
    }

    // A spanning tree over the representatives, contracted onto components:
    // keeping only its inter-component edges is enough to connect everything,
    // because contracting a connected spanning subgraph stays connected.
    const std::vector<Edge> rep_mst = mst_over_subset(reps);
    UnionFind uf(ncomp);
    std::size_t added = 0;
    for (const Edge& e : rep_mst) {
        if (uf.unite(comp[e.u], comp[e.v])) {
            missing_.push_back(e);
            ++added;
        }
    }
    repair_count_ = added;
    stats->repair_edges = added;
    stats->missing_edges = missing_.size();
    stats->components_after_repair = uf.components();
    stats->repair_seconds = since(t0);
    // The repair edges are bridges, not k-NN edges, so the tracker's own
    // labelling stays correct and does not need invalidating.

    if (params_.verbose) {
        std::printf("[coresg] repair: %u components -> %u using %zu reps (%zu edges)\n",
                    static_cast<unsigned>(ncomp),
                    static_cast<unsigned>(stats->components_after_repair), reps.size(), added);
    }
}

CoreSGStats CoreSG::build(const float* rows, std::size_t count, std::uint32_t dim) {
    const auto t_total = Clock::now();
    CoreSGStats stats;
    if (built_) throw std::runtime_error("CoreSG::build called twice; use insert()");

    data_ = Dataset(dim, params_.metric);
    data_.reserve(count);
    data_.append(rows, count);

    knn_.reset(static_cast<idx_t>(count), params_.k_max);

    const auto t_knn = Clock::now();
    stats.knn_stats = incr_->build(data_, &knn_);
    stats.knn_seconds = since(t_knn);

    // The hubness index is attached only after the initial build: during the
    // build the graph churns through millions of edges that nobody needs to
    // observe, and a single O(n*k) count at the end is far cheaper.
    // Components first: the per-component anti-hub quota needs them, and they
    // are reused unchanged by the repair step further down.
    conn_.reset(static_cast<idx_t>(count));
    stats.conn_stats = conn_.rebuild(knn_, pool_.get());

    const auto t_hub = Clock::now();
    hub_.set_params(params_.hub);
    hub_.build(knn_, pool_.get());
    hub_.set_components(&conn_.labels(), conn_.components());
    stats.knn_edges = knn_.undirected_edge_count(pool_.get());
    stats.hub_stats =
        hub_.rescore_all(knn_, budget_dim(&stats), stats.knn_edges, pool_.get());
    hub_.set_track_removals(params_.incremental_connectivity);
    hub_.attach(&knn_);
    stats.hubness_seconds = since(t_hub);

    recompute_missing_edges(&stats);
    repair_connectivity(&stats);

    stats.n = count;
    stats.new_points = count;
    stats.total_seconds = since(t_total);
    stats.peak_host_bytes = host_memory().peak_bytes;
    stats.largest_instance_bytes = stats.knn_stats.largest_instance_bytes;
    built_ = true;

    if (params_.verbose) {
        std::printf(
            "[coresg] build n=%zu k_max=%u: knn %.3fs (%zu edges), hubness %.3fs "
            "(%zu selected), missing %.3fs (%zu edges), repair %.3fs (%zu edges), total %.3fs\n",
            count, params_.k_max, stats.knn_seconds, stats.knn_edges, stats.hubness_seconds,
            stats.hub_stats.selected, stats.missing_seconds,
            stats.missing_edges - stats.repair_edges, stats.repair_seconds, stats.repair_edges,
            stats.total_seconds);
    }
    return stats;
}

CoreSGStats CoreSG::insert(const float* rows, std::size_t count) {
    const auto t_total = Clock::now();
    CoreSGStats stats;
    if (!built_) throw std::runtime_error("CoreSG::insert before build()");
    if (count == 0) return stats;

    const idx_t first_new = static_cast<idx_t>(data_.size());
    data_.append(rows, count);
    const idx_t n_total = static_cast<idx_t>(data_.size());

    knn_.grow(n_total);
    hub_.grow(n_total);

    // The observer stays attached: every insertion and eviction performed by
    // the merge and the restricted refinement updates a reverse degree in
    // O(1) and marks the affected nodes, so the refresh below only has to
    // rescore the dirty region instead of the whole graph.
    const auto t_knn = Clock::now();
    stats.knn_stats = incr_->insert(data_, &knn_, first_new);
    stats.knn_seconds = since(t_knn);

    // Before hub_.refresh(), which clears the change log this consumes.
    conn_.grow(n_total);
    if (params_.incremental_connectivity) {
        stats.conn_stats =
            conn_.update(knn_, hub_.removal_candidates(), hub_.changed_nodes(), first_new, &hub_,
                         pool_.get(), params_.connectivity_probe_budget);
    } else {
        stats.conn_stats = conn_.rebuild(knn_, pool_.get());
    }
    hub_.clear_removals();

    const auto t_hub = Clock::now();
    hub_.set_components(&conn_.labels(), conn_.components());
    stats.knn_edges = knn_.undirected_edge_count(pool_.get());
    stats.hub_stats = hub_.refresh(knn_, budget_dim(&stats), stats.knn_edges, pool_.get());
    stats.hubness_seconds = since(t_hub);

    recompute_missing_edges(&stats);
    repair_connectivity(&stats);

    stats.n = n_total;
    stats.new_points = count;
    stats.total_seconds = since(t_total);
    stats.peak_host_bytes = host_memory().peak_bytes;
    stats.largest_instance_bytes = stats.knn_stats.largest_instance_bytes;

    if (params_.verbose) {
        std::printf(
            "[coresg] insert +%zu (n=%zu): knn %.3fs, hubness %.3fs (%zu rescored, %zu "
            "selected), missing %.3fs (%zu edges), repair %.3fs (%zu edges), total %.3fs\n",
            count, std::size_t(n_total), stats.knn_seconds, stats.hubness_seconds,
            stats.hub_stats.rescored, stats.hub_stats.selected, stats.missing_seconds,
            stats.missing_edges - stats.repair_edges, stats.repair_seconds, stats.repair_edges,
            stats.total_seconds);
    }
    return stats;
}

std::vector<Edge> CoreSG::edges(std::uint32_t mpts) const {
    if (mpts == 0) mpts = 1;
    if (mpts > params_.k_max) {
        throw std::runtime_error("CoreSG::edges: mpts exceeds k_max");
    }
    const std::vector<dist_t> core = core_distances(mpts);
    std::vector<Edge> out = knn_.mutual_reachability_edges(core, pool_.get());
    // Bridge edges carry their *base* distance, so they reweight exactly the
    // same way as the k-NN half: mrd_mpts(p,q) = max(d(p,q), c(p), c(q)).
    out.reserve(out.size() + missing_.size());
    for (const Edge& e : missing_) {
        const dist_t c = std::max(core[e.u], core[e.v]);
        out.emplace_back(e.u, e.v, std::max(e.w, c));
    }
    return out;
}

MstResult CoreSG::mst(std::uint32_t mpts) const {
    return mst_from_edges(edges(mpts), knn_.size(), pool_.get());
}

std::vector<MstResult> CoreSG::all_msts(const std::vector<std::uint32_t>& mpts_values) const {
    std::vector<MstResult> out;
    out.reserve(mpts_values.size());
    for (std::uint32_t m : mpts_values) out.push_back(mst(m));
    return out;
}

MstResult CoreSG::exact_mst(std::uint32_t mpts) const {
    const idx_t n = knn_.size();
    const std::vector<dist_t> core = core_distances(mpts);
    Backend* be = backend_.get();
    const float* pts = data_.data();
    const std::uint32_t dim = data_.dim();
    const std::uint32_t stride = data_.stride();
    const MetricKind metric = data_.metric();
    return mst_prim_virtual(n, [&](idx_t i, float* dst) {
        be->mrd_block(0, pts, n, dim, stride, metric, core.data(), i, i + 1, dst);
    });
}

}  // namespace coresg
