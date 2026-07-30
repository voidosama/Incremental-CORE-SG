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
// Sampling modes:
//   Partition   -- one random permutation of the base points cut into
//                  `instances` disjoint blocks. Every base point is seen by
//                  exactly one instance, which gives the best coverage per
//                  unit of work and is the default.
//   Independent -- each instance draws its own uniform sample without
//                  replacement. Blocks overlap; coverage is probabilistic.

#include <cstdint>
#include <vector>

#include "coresg/backend.hpp"
#include "coresg/dataset.hpp"
#include "coresg/knn_graph.hpp"
#include "coresg/nndescent.hpp"
#include "coresg/types.hpp"

namespace coresg {

enum class InstanceSampling : std::uint8_t { Partition = 0, Independent = 1 };

struct IncrementalKnnParams {
    /// Number of NN-Descent instances per update. This is the knob that feeds
    /// the device queue: `instances >= #devices` keeps every GPU busy.
    std::uint32_t instances = 4;
    InstanceSampling sampling = InstanceSampling::Partition;
    /// Independent mode only: |sample| / |base|. 0 means 1 / instances.
    double sample_fraction = 0.0;
    /// Fraction of extra random points mixed into each instance, on top of its
    /// own block. Helps the blocks agree with each other during the merge.
    double overlap_fraction = 0.05;
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
    /// This is what keeps a batch sublinear: with k=20 and a second ring, an
    /// unbounded set reaches k^2 nodes per new point and saturates a small
    /// graph. 1.0 disables the cap.
    double refine_max_fraction = 0.25;
    /// Absolute ceiling, applied on top of the fraction. 0 disables it.
    std::size_t refine_max_nodes = 0;

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
    IncrementalKnnStats insert(const Dataset& data, KnnGraph* graph, idx_t first_new);

private:
    struct Instance {
        std::vector<idx_t> nodes;    ///< global ids; context first, then owned
        std::size_t owned_begin = 0; ///< index of the first owned node
    };

    std::vector<Instance> make_instances(idx_t n_base, idx_t first_new, idx_t n_total,
                                         std::uint64_t seed) const;
    std::vector<Instance> make_build_instances(idx_t n_total, std::uint64_t seed) const;

    void stage_job(const Dataset& data, const KnnGraph& graph, const Instance& inst,
                   const NNDescentParams& nnd, std::vector<idx_t>* g2l, KnnJob* job) const;

    std::size_t merge_job(const KnnJob& job, const Instance& inst, idx_t first_new,
                          KnnGraph* graph, std::size_t* candidates) const;

    std::vector<idx_t> refinement_set(const KnnGraph& graph, idx_t first_new, idx_t n_total,
                                      IncrementalKnnStats* stats) const;

    IncrementalKnnParams params_;
    Backend* backend_;
    ThreadPool* pool_;
};

}  // namespace coresg
