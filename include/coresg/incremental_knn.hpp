#pragma once
// Incremental k-NN graph construction on top of Dynamic NN-Descent.
//
// NN-Descent is not incremental: it refines a whole graph in place, and
// dropping a batch of new points into an existing graph leaves them without
// the neighbourhood structure the descent needs to converge. Re-running it
// over everything costs O(n) per batch, which defeats the purpose.
//
// The scheme implemented here instead builds several *instances*. An instance
// is a random sample of the existing points (unique within the instance)
// merged with the whole batch of new points. Each instance is a completely
// self-contained NN-Descent problem: it can be queued on any device, several
// can run at once, and none of them needs the full data set resident. The
// instances are warm-started from the edges the sampled base points already
// have, so they converge in far fewer rounds than a cold build.
//
// Once the instances finish, every neighbourhood they produced is merged back
// into the global graph -- which, because neighbourhoods are sorted top-k
// lists, is just repeated insertion. A final NN-Descent restricted to the new
// points and their neighbourhoods (optionally a second ring) cleans up the
// edges the sampling missed.
//
// Sampling modes (Resample strategy only):
//   Partition   -- one random permutation of the base points cut into
//                  `instances` disjoint blocks. Every base point is seen by
//                  exactly one instance.
//   Independent -- each instance draws its own uniform sample without
//                  replacement. Blocks overlap; coverage is probabilistic.
//
// Insert strategies
// -----------------
// Resample (the original) draws the instance context uniformly from the base.
// Under Partition the blocks are disjoint, so their union is *every* base
// point and an update costs Theta(n_base) however small the batch is; under
// Independent with the default fraction the total lands in the same place.
// That is the cost the incremental scheme exists to avoid, and it is why an
// update only beats a rebuild once several devices divide it.
//
// Anchored (the default) keeps the instance decomposition -- self-contained,
// warm started, queueable onto any device -- and changes only *where the
// sample comes from*. A uniform sample of the base is a poor picture of the
// region a batch actually lands in: at `instances = s` only 1/s of a context
// point's neighbours land in the same instance, so the warm start is mostly
// discarded and the descent re-derives what the global graph already knew.
// Instead:
//
//   1. seed  -- every new point beam-searches the existing graph for its own
//               neighbourhood. This is the step that consults the *whole*
//               base, and it is O(ef * k * d) per point, not O(n_base);
//   2. anchor - the instance context is the neighbourhood closure of those
//               seeds, capped by an absolute budget. The context is therefore
//               a coherent local region rather than a scatter, and a context
//               point keeps most of its real neighbours;
//   3. descend/merge/refine -- unchanged.
//
// The *context* is split across instances; the batch is not. Every instance
// carries the whole batch, because the batch is its own k-NN problem and an
// instance holding 1/s of it can only discover 1/s of the intra-batch edges.
// Total instance work is s*|batch| + |context|, both O(|batch|) and independent
// of n_base. Instances stay the unit of device parallelism.
//
// The seeding search runs on the host and is a read-only pass over the graph:
// it must complete before anything is inserted, both because the reverse lists
// it navigates are mutated by the observer and so that the result does not
// depend on the order new points happen to be processed in.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "coresg/backend.hpp"
#include "coresg/dataset.hpp"
#include "coresg/knn_graph.hpp"
#include "coresg/nndescent.hpp"
#include "coresg/types.hpp"

namespace coresg {

enum class InstanceSampling : std::uint8_t { Partition = 0, Independent = 1 };

enum class InsertStrategy : std::uint8_t {
    /// Uniform sample of the base per instance. Theta(n_base) per update.
    /// Kept because every result before 2026-08 was measured with it and it is
    /// the arm the anchored strategy is scored against.
    Resample = 0,
    /// Seed by searching the graph, anchor the context on what the search
    /// found. O(|batch|) per update.
    Anchored = 1,
};

const char* to_string(InsertStrategy s);
bool insert_strategy_from_string(const std::string& s, InsertStrategy* out);

struct IncrementalKnnParams {
    /// Number of NN-Descent instances per update. This is the knob that feeds
    /// the device queue: `instances >= #devices` keeps every GPU busy.
    std::uint32_t instances = 4;
    InsertStrategy strategy = InsertStrategy::Anchored;
    InstanceSampling sampling = InstanceSampling::Partition;
    /// Independent mode only: |sample| / |base|. 0 means 1 / instances.
    double sample_fraction = 0.0;
    /// Fraction of extra points mixed into each instance, on top of its own
    /// block. Helps the blocks agree with each other during the merge. Under
    /// Anchored it sprinkles context points from other blocks into each
    /// instance, so no block is blind to the regions the others hold. The new
    /// points are never split, so this does not apply to them.
    double overlap_fraction = 0.05;

    // ---- Anchored strategy ------------------------------------------------
    /// Beam width of the seeding search. 0 means 2 * k. This is the knob that
    /// trades update cost against how good the seed neighbourhood is.
    std::uint32_t search_ef = 0;
    /// Random entry points per new point. A k-NN graph is not guaranteed
    /// connected, so a single entry can be trapped in the wrong component.
    std::uint32_t search_entries = 8;
    /// Ceiling on nodes a single search may expand, as a multiple of `ef`.
    /// Bounds the worst case on adversarial data.
    std::uint32_t search_max_expand = 4;
    /// Reverse neighbours consulted per expanded node. Back edges are what
    /// make a directed k-NN graph navigable, but they are also unbounded in
    /// degree -- a hub can appear in thousands of lists -- and each one costs
    /// a distance evaluation. 0 searches forward edges only.
    std::uint32_t search_reverse_cap = 16;
    /// Base points pulled into the instance context per new point. The context
    /// is the closure of the seeds, so this is a budget, not a target.
    /// Swept on sift1m at n=200k: 4.0 and 2.0 differ by 0.5 points of recall
    /// and 25 k instance nodes, 1.0 loses 2 points.
    double context_per_new = 2.0;
    /// Absolute ceiling on the context, on top of `context_per_new`.
    /// 0 disables it.
    std::size_t context_max_nodes = 0;
    /// How many of its seed edges an owned (new) node inherits into its
    /// instance list, as a fraction of k. The remaining slots are left empty
    /// so the backend's random fill can put *other new points* there.
    ///
    /// This is load bearing, not a tuning knob. The seeding search only sees
    /// the base, so it cannot discover that two points arriving in the same
    /// batch are each other's neighbours -- and a fully warm-started owned
    /// node has no free slot for a random candidate to land in, so NN-Descent
    /// never compares the new points with each other either. The batch's own
    /// k-NN structure then simply never gets built. It costs nothing on a
    /// small batch and 20 points of recall on a batch worth 20 % of the graph,
    /// which is inside the range the batch-size study recommends operating in.
    double owned_warm_fraction = 0.25;
    /// Below this many base points, fall back to a single instance.
    std::size_t min_base_for_split = 5000;
    /// At or below this many total points, build the k-NN graph exactly by
    /// brute force instead of descending. 0 disables it. Exists so that the
    /// CORE-SG layer can be validated against the theory without the
    /// approximation of the k-NN half getting in the way.
    std::size_t brute_force_limit = 0;

    NNDescentParams instance_nnd;  ///< per-instance descent (warm started, few rounds)
    NNDescentParams refine_nnd;    ///< restricted refinement after the merge

    /// Seed instances from the edges the sampled base points already have.
    bool warm_start = true;
    /// Also push instance-discovered base-base edges into the global graph.
    /// Free quality; set false to keep the dirty set (and the hubness refresh)
    /// as small as possible.
    bool apply_base_base = true;
    /// Refine over the neighbours of the new points' neighbours too.
    bool refine_second_ring = true;
    /// Ceiling on the refinement set, as a fraction of the whole graph. The
    /// set is built new points -> first ring -> second ring, so the cap drops
    /// the outermost nodes first and the new points are never sacrificed.
    /// 1.0 disables the cap.
    ///
    /// Note the *form*: a fraction of n is Theta(n) per batch whatever its
    /// constant, so it cannot by itself keep an update sublinear. Under
    /// Anchored the absolute form is what binds -- see `refine_per_new`.
    double refine_max_fraction = 0.25;
    /// Absolute ceiling, applied on top of the fraction. 0 disables it.
    std::size_t refine_max_nodes = 0;
    /// Anchored only: refinement budget expressed per new point, which is the
    /// form that stays sublinear. Applied as an absolute cap of
    /// `refine_per_new * |batch|` nodes. 0 falls back to the fraction.
    ///
    /// Measured 2 / 4 / 8 / 16 over eight batches: 4 buys +1.6 points of recall
    /// on sift1m and +2.3 on gist1m for no measurable time (1.195 s -> 1.200 s
    /// per update). 8 and 16 buy exactly nothing.
    ///
    /// They buy nothing because `refine_max_fraction` is a ceiling too and the
    /// two combine with a min: at n=200k the fraction allows 50,000 nodes,
    /// which is what 4 already asks for, so every setting above 4 is measuring
    /// the fraction rather than this knob. Raise `refine_max_fraction` as well
    /// before concluding that a larger repair budget does not help.
    double refine_per_new = 4.0;

    std::uint64_t seed = 7;
    bool verbose = false;
};

struct IncrementalKnnStats {
    std::size_t instances = 0;
    std::size_t new_points = 0;
    std::size_t base_points = 0;
    std::size_t instance_nodes = 0;   ///< total nodes across instances
    std::size_t jobs_on_device = 0;   ///< instances that really ran on a GPU
    std::size_t jobs_on_host = 0;     ///< instances the backend handed to the CPU
    std::size_t jobs_out_of_memory = 0;
    std::size_t largest_instance = 0; ///< nodes in the biggest instance
    std::size_t largest_instance_bytes = 0;  ///< its device footprint
    std::size_t candidate_edges = 0;  ///< directed candidates offered to the merge
    std::size_t applied_edges = 0;    ///< candidates that actually improved a list
    std::size_t refine_nodes = 0;
    std::size_t refine_ring0 = 0;  ///< the new points themselves
    std::size_t refine_ring1 = 0;  ///< their neighbours
    bool refine_capped = false;
    /// Anchored only. `search_evals` is the honest cost measure of the seeding
    /// step: it is what replaces descending the whole base.
    std::size_t search_evals = 0;
    std::size_t context_nodes = 0;   ///< distinct base points pulled in
    bool context_capped = false;
    bool used_reverse_edges = false; ///< search navigated the undirected graph
    double search_seconds = 0.0;
    double stage_seconds = 0.0;      ///< gather + warm start, previously untimed
    double instance_seconds = 0.0;
    double merge_seconds = 0.0;
    double refine_seconds = 0.0;
    NNDescentStats refine_stats;
    std::vector<NNDescentStats> per_instance;
};

class IncrementalKnn {
public:
    IncrementalKnn(const IncrementalKnnParams& params, Backend* backend, ThreadPool* pool)
        : params_(params), backend_(backend), pool_(pool) {}

    const IncrementalKnnParams& params() const { return params_; }
    void set_params(const IncrementalKnnParams& p) { params_ = p; }

    /// Build a graph over the whole data set from scratch. With
    /// `instances == 1` this is a plain Dynamic NN-Descent; with more, the
    /// data set is partitioned, each block is built independently (one per
    /// device), merged, and then globally refined.
    IncrementalKnnStats build(const Dataset& data, KnnGraph* graph);

    /// Insert the points [first_new, data.size()) into an existing graph.
    ///
    /// `reverse` is optional and only consulted by the Anchored strategy: with
    /// it the seeding search walks the undirected graph, which is markedly
    /// more reliable than following forward edges alone. CoreSG passes its
    /// hubness index, which maintains the table anyway.
    /// `entry_hints` is optional and likewise Anchored-only: node ids the
    /// seeding search should start from in preference to uniform random ones.
    /// CoreSG passes the selected anti-hubs, which carry a per-component
    /// quota -- and a k-NN graph over well-separated clusters is disconnected,
    /// so a uniformly drawn entry point usually starts in the wrong component
    /// and greedy search cannot cross. Worth 8 points of recall on the
    /// 150-cluster manifold set and nothing at all on connected data.
    IncrementalKnnStats insert(const Dataset& data, KnnGraph* graph, idx_t first_new,
                               const ReverseNeighbours* reverse = nullptr,
                               const std::vector<idx_t>* entry_hints = nullptr);

private:
    struct Instance {
        std::vector<idx_t> nodes;    ///< global ids; context first, then owned
        /// Index of the first owned node: everything before it is context the
        /// instance was given to descend against, everything from it on is
        /// what the instance is responsible for. Always set; only the anchored
        /// constructor needs to read it back, because that is the only path
        /// where context and owned nodes are assembled separately.
        std::size_t owned_begin = 0;
    };

    /// One new point's seed neighbourhood, produced by the read-only search
    /// pass and applied afterwards.
    struct Seed {
        idx_t node = kInvalidIdx;
        std::vector<std::pair<dist_t, idx_t>> best;
    };

    std::vector<Instance> make_instances(idx_t n_base, idx_t first_new, idx_t n_total,
                                         std::uint64_t seed) const;
    std::vector<Instance> make_build_instances(idx_t n_total, std::uint64_t seed) const;

    /// Read-only beam search for every new point. Nothing is written to the
    /// graph, so the reverse table stays stable for the duration.
    void search_seeds(const Dataset& data, const KnnGraph& graph, idx_t first_new, idx_t n_total,
                      const ReverseNeighbours* reverse, const std::vector<idx_t>* entry_hints,
                      std::vector<Seed>* out, IncrementalKnnStats* stats) const;

    /// Instances whose context is the neighbourhood closure of `seeds`, and
    /// whose owned set is a slice of the new batch.
    std::vector<Instance> make_anchored_instances(const KnnGraph& graph,
                                                  const std::vector<Seed>& seeds, idx_t first_new,
                                                  idx_t n_total, std::uint64_t seed,
                                                  IncrementalKnnStats* stats) const;

    /// `g2l` is scratch of size n_total, left all-kInvalidIdx on return so the
    /// next instance can reuse it. Staging several instances at once therefore
    /// needs one buffer per worker, not one shared buffer.
    void stage_job(const Dataset& data, const KnnGraph& graph, const Instance& inst,
                   const NNDescentParams& nnd, std::vector<idx_t>* g2l, KnnJob* job) const;

    /// Stage every instance, in parallel when there is more than one.
    void stage_all(const Dataset& data, const KnnGraph& graph,
                   const std::vector<Instance>& instances, const NNDescentParams& nnd,
                   idx_t n_total, std::vector<KnnJob>* jobs, IncrementalKnnStats* stats) const;

    std::size_t merge_job(const KnnJob& job, const Instance& inst, idx_t first_new,
                          KnnGraph* graph, std::size_t* candidates) const;

    std::vector<idx_t> refinement_set(const KnnGraph& graph, idx_t first_new, idx_t n_total,
                                      IncrementalKnnStats* stats) const;

    IncrementalKnnParams params_;
    Backend* backend_;
    ThreadPool* pool_;
};

}  // namespace coresg
