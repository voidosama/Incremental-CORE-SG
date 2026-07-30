#pragma once
// CORE-SG: the k_max-NN graph unioned with a spanning structure that supplies
// the edges the neighbourhoods cannot express.
//
// The original result (Neto, Naldi, Campello & Sander, ICDE 2022) is
//
//     CORE-SG_kmax = kmax-NNG  union  MST_kmax
//
// and, for every mpts <= kmax, some MST_mpts of the complete mutual
// reachability graph is contained in it. Reweighting the same edge set with
// mrd_mpts and running an MST therefore replaces a fresh O(n^2) run per mpts.
//
// This implementation keeps the k-NN half exact-in-structure (an approximate
// k-NN graph built and maintained incrementally by Dynamic NN-Descent) and
// replaces the MST half with one of:
//
//   MissingEdgeMode::AntiHub  -- the estimate this project proposes. Only the
//     MST edges whose weight is *not* determined by a core distance matter,
//     i.e. edges bridging regions the neighbourhoods cannot reach. Those
//     bridges leave from the border of a cluster, so an MST computed over the
//     d*ln(|E|) strongest local anti-hubs approximates them at a cost that is
//     independent of n. This is what makes the whole structure maintainable
//     incrementally: no MST has to be repaired when points arrive, only the
//     anti-hub set has to be re-selected.
//
//   MissingEdgeMode::ExactMst -- the real MST_kmax over the complete mutual
//     reachability graph, O(n^2 d). Not scalable; it is here so that the
//     approximation can be measured against the thing it approximates.
//
//   MissingEdgeMode::None     -- k-NN graph only, for ablation.
//
// Because an approximate bridge set can leave the graph disconnected (and a
// disconnected replacement graph yields a forest, not a spanning tree), an
// optional repair step spans whatever components remain using representatives
// drawn from each component's lowest-scoring nodes.

#include <memory>
#include <string>
#include <vector>

#include "coresg/backend.hpp"
#include "coresg/connectivity.hpp"
#include "coresg/dataset.hpp"
#include "coresg/hubness.hpp"
#include "coresg/incremental_knn.hpp"
#include "coresg/intrinsic_dim.hpp"
#include "coresg/knn_graph.hpp"
#include "coresg/mst.hpp"
#include "coresg/thread_pool.hpp"
#include "coresg/types.hpp"

namespace coresg {

enum class MissingEdgeMode : std::uint8_t {
    AntiHub = 0,
    ExactMst = 1,
    None = 2,
};

const char* to_string(MissingEdgeMode m);
bool missing_edge_mode_from_string(const std::string& s, MissingEdgeMode* out);

struct CoreSGParams {
    std::uint32_t k_max = 32;  ///< also the largest supported mpts
    MetricKind metric = MetricKind::L2;
    MissingEdgeMode missing_mode = MissingEdgeMode::AntiHub;

    /// Span the remaining components after the missing edges are added.
    bool ensure_connected = true;
    std::size_t reps_per_component = 4;
    std::size_t max_reps = 4096;
    /// Maintain components across batches instead of recomputing them.
    ///
    /// OFF by default, and that is a measured decision, not an oversight. The
    /// certificate is correct and the path is exact (see test_connectivity),
    /// but NN-Descent destroys a very large number of undirected edges per
    /// batch -- 51k at a 1 % batch on 100k points, 783k at 20 % -- and each one
    /// needs a bidirectional search. That costs 19-209 ms against a parallel
    /// full rebuild's 1-2 ms. Worth revisiting only if the k-NN half is ever
    /// changed to freeze converged regions, which is what would make the
    /// destroyed set small.
    bool incremental_connectivity = false;
    /// Nodes a single certificate search may visit before giving up and
    /// falling back to re-deriving that component.
    std::size_t connectivity_probe_budget = 512;
    /// Above this many representatives the dense matrix is streamed row by row
    /// instead of being materialised.
    std::size_t dense_matrix_limit = 8192;

    /// Which dimensionality feeds the anti-hub budget d * ln(|E|). Ambient is
    /// the specified rule; Intrinsic substitutes the estimated manifold
    /// dimensionality, which is what stops the budget exceeding n on
    /// high-dimensional data. Estimation is free -- it reads the k-NN
    /// distances that are already stored.
    BudgetDimension budget_dimension = BudgetDimension::Ambient;
    /// k for the estimator; 0 uses min(k_max, 20).
    std::uint32_t intrinsic_dim_k = 0;

    IncrementalKnnParams incr;
    HubnessParams hub;
    BackendOptions backend;

    unsigned threads = 0;  ///< 0 = hardware concurrency
    std::uint64_t seed = 42;
    bool verbose = false;
};

struct CoreSGStats {
    std::size_t n = 0;
    std::size_t new_points = 0;
    std::size_t knn_edges = 0;       ///< undirected edges of the k-NN half
    std::size_t missing_edges = 0;   ///< anti-hub (or exact MST) bridge edges
    std::size_t repair_edges = 0;    ///< extra edges added to force connectivity
    std::size_t antihubs_selected = 0;
    idx_t components_before_repair = 0;
    idx_t components_after_repair = 0;

    double knn_seconds = 0.0;
    double hubness_seconds = 0.0;
    double missing_seconds = 0.0;
    double repair_seconds = 0.0;
    double total_seconds = 0.0;

    /// Dimensionality actually used for the budget, and where it came from.
    double effective_dim = 0.0;
    IntrinsicDimStats id_stats;

    /// Memory. `peak_host_bytes` is the process high-water mark;
    /// `largest_instance_bytes` is what the biggest single instance needed
    /// resident on a device, which is the figure that decides whether a
    /// configuration fits on a given card.
    std::size_t peak_host_bytes = 0;
    std::size_t largest_instance_bytes = 0;

    IncrementalKnnStats knn_stats;
    HubnessStats hub_stats;
    ConnectivityStats conn_stats;
};

class CoreSG {
public:
    explicit CoreSG(const CoreSGParams& params);
    ~CoreSG();

    CoreSG(const CoreSG&) = delete;
    CoreSG& operator=(const CoreSG&) = delete;

    const CoreSGParams& params() const { return params_; }
    const Dataset& data() const { return data_; }
    const KnnGraph& knn() const { return knn_; }
    const HubnessIndex& hubness() const { return hub_; }
    const ConnectivityTracker& connectivity() const { return conn_; }
    /// The bridge edges: MST over the selected anti-hubs (plus, when
    /// `ensure_connected`, the component-repair edges appended after them).
    const std::vector<Edge>& missing_edges() const { return missing_; }
    std::size_t repair_edge_count() const { return repair_count_; }
    Backend* backend() const { return backend_.get(); }
    ThreadPool* pool() const { return pool_.get(); }

    /// First build. `rows` is `count * dim` floats, row major.
    CoreSGStats build(const float* rows, std::size_t count, std::uint32_t dim);

    /// Incremental insertion of `count` further points.
    CoreSGStats insert(const float* rows, std::size_t count);

    /// Full CORE-SG edge list reweighted with mrd_mpts.
    std::vector<Edge> edges(std::uint32_t mpts) const;

    /// MST_mpts extracted from the CORE-SG. This is the O(|E| log n) step that
    /// replaces a fresh O(n^2) density-based MST per mpts value.
    MstResult mst(std::uint32_t mpts) const;

    /// Every MST in [1, k_max], sharing one materialisation of the edge list.
    std::vector<MstResult> all_msts(const std::vector<std::uint32_t>& mpts_values) const;

    /// Reference: exact MST over the *complete* mutual reachability graph.
    /// O(n^2 d) -- evaluation only.
    MstResult exact_mst(std::uint32_t mpts) const;

    std::vector<dist_t> core_distances(std::uint32_t mpts) const;

    /// Estimated intrinsic dimensionality of the current graph. Cheap: one
    /// pass over the stored k-NN distances, no distance computations.
    IntrinsicDimStats intrinsic_dimension() const;

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    /// Dimensionality term for the budget, per `budget_dimension`.
    std::uint32_t budget_dim(CoreSGStats* stats) const;
    void recompute_missing_edges(CoreSGStats* stats);
    void repair_connectivity(CoreSGStats* stats);
    /// MST over an arbitrary subset, in mutual reachability space w.r.t.
    /// k_max. Returns edges in *global* ids.
    std::vector<Edge> mst_over_subset(const std::vector<idx_t>& ids) const;

    CoreSGParams params_;
    Dataset data_;
    KnnGraph knn_;
    HubnessIndex hub_;
    ConnectivityTracker conn_;
    std::vector<Edge> missing_;
    std::size_t repair_count_ = 0;

    std::unique_ptr<ThreadPool> pool_;
    std::unique_ptr<Backend> backend_;
    std::unique_ptr<IncrementalKnn> incr_;
    bool built_ = false;
};

}  // namespace coresg
