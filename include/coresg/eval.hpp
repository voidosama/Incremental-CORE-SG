#pragma once
// Evaluation helpers.
//
// Two questions have to be answerable for this project to mean anything:
//
//   1. how good is the approximate k-NN graph (recall against brute force),
//      and does the incremental path degrade it relative to a fresh build;
//   2. how good is the CORE-SG built from anti-hub bridges, compared with the
//      one built from the true MST_kmax -- measured on the MSTs that come out
//      of it, since that is what every downstream density-based method
//      consumes.

#include <cstdint>
#include <string>
#include <vector>

#include "coresg/dataset.hpp"
#include "coresg/knn_graph.hpp"
#include "coresg/mst.hpp"
#include "coresg/types.hpp"

namespace coresg {

class ThreadPool;

struct RecallResult {
    double recall = 0.0;         ///< fraction of true k-NN present, averaged over nodes
    double distance_ratio = 1.0; ///< mean(approx k-th dist) / mean(true k-th dist)
    std::size_t k = 0;
    std::size_t nodes = 0;
};

/// Exact k-NN graph by brute force. O(n^2 d) -- reference only.
KnnGraph brute_force_graph(const Dataset& data, std::uint32_t k, ThreadPool* pool = nullptr);

RecallResult knn_recall(const KnnGraph& approx, const KnnGraph& exact, std::uint32_t k);

/// Recall estimated from a sample of `queries` nodes, each scored exactly
/// against every point. O(queries * n * d) instead of O(n^2 d), which is what
/// makes it usable at the sizes this project builds at. `queries >= n` is the
/// exhaustive case. Pass a seed so a sweep compares the same query set.
RecallResult sampled_recall(const Dataset& data, const KnnGraph& approx, std::uint32_t k,
                            std::size_t queries, std::uint64_t seed = 4242,
                            ThreadPool* pool = nullptr);

/// Recall restricted to a node subset (e.g. only the incrementally added
/// points, which is where the incremental scheme can lose quality).
RecallResult knn_recall_subset(const KnnGraph& approx, const KnnGraph& exact, std::uint32_t k,
                               const std::vector<idx_t>& nodes);

struct MstComparison {
    double approx_weight = 0.0;
    double exact_weight = 0.0;
    double weight_ratio = 1.0;  ///< approx / exact; 1.0 means identical cost
    std::size_t approx_edges = 0;
    std::size_t exact_edges = 0;
    std::size_t shared_edges = 0;
    double edge_jaccard = 0.0;
    idx_t approx_components = 0;
    idx_t exact_components = 0;
    /// Largest single-edge weight difference, i.e. the worst distortion of the
    /// density level at which two regions merge.
    double max_edge_weight_gap = 0.0;
};

MstComparison compare_msts(const MstResult& approx, const MstResult& exact);

/// Adjusted Rand Index between two labellings. Used to compare the flat
/// clusterings obtained by cutting two MSTs at the same level.
double adjusted_rand_index(const std::vector<idx_t>& a, const std::vector<idx_t>& b);

/// Cut every MST edge with weight > `threshold` and label the components. This
/// is the DBSCAN* clustering at eps = threshold w.r.t. the mpts the MST was
/// built for.
std::vector<idx_t> dbscan_labels_from_mst(const MstResult& mst, idx_t n, double threshold,
                                          idx_t* num_clusters = nullptr);

std::string format_recall(const RecallResult& r);
std::string format_mst_comparison(const MstComparison& c);

}  // namespace coresg
