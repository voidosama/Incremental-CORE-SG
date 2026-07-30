#pragma once
// Connected components of the k-NN graph, maintained across batches.
//
// Why this exists. The repair step needs to know whether the replacement graph
// fell apart, and if so it needs one representative per component. Computing
// that from scratch costs O(n*k) edge traversals per batch -- which, next to a
// refinement that only touches a batch-sized slice of the graph, is the single
// most expensive non-incremental step left in an update.
//
// The exactness floor. Connectivity cannot be decided sublinearly in n: one
// isolated point is undetectable unless something touches it, and sublinear
// property testers only separate "connected" from "far from connected", so
// they are blind to exactly the small distant cluster that matters here. What
// *is* achievable is making the common case proportional to the batch, and
// making the fallback fully parallel.
//
// The certificate. Insertions can only merge components, so seeding a
// union-find with the previous labels and unioning over the edges of nodes
// that changed is exact for that direction. Deletions are the hard direction,
// but they admit a cheap sufficient condition:
//
//     If every undirected edge destroyed this batch still has its two
//     endpoints connected in the resulting graph, the components are
//     unchanged.
//
// Proof: take any path in the old graph and replace each destroyed edge by a
// path between its endpoints in the new graph; splicing them gives a path in
// the new graph. So each destroyed edge only needs a local replacement, which
// a budgeted bidirectional search either finds (certificate) or fails to find
// (inconclusive -> full rebuild). Being a *sufficient* condition, a failure
// costs time and never correctness.
//
// The fallback is a lock-free union-find run straight over the adjacency
// arrays -- no edge list is materialised, and the work is spread over the
// thread pool.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "coresg/knn_graph.hpp"
#include "coresg/types.hpp"

namespace coresg {

class ThreadPool;
class HubnessIndex;

struct ConnectivityStats {
    idx_t components = 0;
    bool incremental = false;      ///< false means the full rebuild ran
    std::size_t removal_candidates = 0;
    std::size_t destroyed_edges = 0;  ///< candidates that really died
    std::size_t certified_edges = 0;  ///< destroyed edges proven non-critical
    std::size_t probe_nodes = 0;      ///< nodes visited by the searches
    std::size_t union_edges = 0;      ///< edges fed to the union-find
    std::size_t dirty_components = 0; ///< old components whose seed was discarded
    std::size_t rederived_nodes = 0;  ///< nodes rebuilt from their adjacency
    /// The destroyed edge that forced a fallback, if any. Diagnostic: lets a
    /// caller tell "the budget ran out" from "this really was a cut".
    idx_t fail_u = kInvalidIdx;
    idx_t fail_v = kInvalidIdx;
    bool fail_exhausted = false;  ///< search exhausted the graph, not the budget
    double seconds = 0.0;
};

class ConnectivityTracker {
public:
    void reset(idx_t n);
    void grow(idx_t n);

    idx_t size() const { return n_; }
    idx_t components() const { return components_; }
    const std::vector<idx_t>& labels() const { return labels_; }
    bool valid() const { return valid_; }
    void invalidate() { valid_ = false; }

    /// Full recompute over the k-NN adjacency. Parallel, allocation-free in
    /// the edge dimension.
    ConnectivityStats rebuild(const KnnGraph& g, ThreadPool* pool = nullptr);

    /// Batch update. Falls back to `rebuild` (and says so in the stats) when
    /// a destroyed edge cannot be certified within `probe_budget` nodes.
    /// `reverse` is optional; when it carries reverse neighbour lists the
    /// searches traverse the true undirected graph instead of the forward
    /// edges alone, which makes certificates far more likely to succeed.
    ConnectivityStats update(const KnnGraph& g,
                             const std::vector<std::pair<idx_t, idx_t>>& removal_candidates,
                             const std::vector<idx_t>& changed_nodes, idx_t first_new,
                             const HubnessIndex* reverse = nullptr, ThreadPool* pool = nullptr,
                             std::size_t probe_budget = 512);

    /// Components of the k-NN graph *plus* a handful of extra edges (the
    /// bridges). Cheap: it works over the cached labels, so it costs
    /// O(n + |extra|) rather than another O(n*k) sweep.
    std::vector<idx_t> labels_with(const std::vector<Edge>& extra, idx_t* num_components) const;

private:
    bool certify(const KnnGraph& g, idx_t u, idx_t v, const HubnessIndex* reverse,
                 std::size_t budget, std::size_t* visited, bool* exhausted);

    idx_t n_ = 0;
    idx_t components_ = 0;
    std::vector<idx_t> labels_;
    bool valid_ = false;

    // Scratch for the bidirectional searches, reused across calls so that a
    // batch with many destroyed edges does not reallocate per edge.
    std::vector<std::uint32_t> seen_;
    std::uint32_t stamp_ = 0;
    std::vector<idx_t> frontier_a_, frontier_b_, next_;
};

}  // namespace coresg
