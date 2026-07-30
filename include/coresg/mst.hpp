#pragma once
// Minimum spanning tree / forest primitives.
//
// Three algorithms, because three very different inputs occur in this project:
//
//   * Kruskal over an explicit edge list -- the CORE-SG replacement graph has
//     O(n * k_max) edges, so a sort dominates and this is the simplest exact
//     answer. Single threaded, deterministic, used as the reference.
//   * Boruvka over the same edge list -- same result, but the rounds are data
//     parallel, which is what the multi-threaded and (later) device paths use.
//   * Prim over a *virtual* complete graph -- O(n^2) time, O(n) memory, rows
//     produced on demand by a callback. This is how MST_kmax over the full
//     mutual-reachability graph is computed exactly for evaluation, and how
//     the (small) anti-hub set is spanned.
//
// All of them return a forest when the input graph is disconnected, together
// with the component labelling, because "how many pieces did the approximate
// CORE-SG fall into" is a number this project has to report.

#include <cstdint>
#include <functional>
#include <vector>

#include "coresg/types.hpp"

namespace coresg {

class ThreadPool;

class UnionFind {
public:
    UnionFind() = default;
    explicit UnionFind(idx_t n) { reset(n); }

    void reset(idx_t n) {
        parent_.resize(n);
        rank_.assign(n, 0);
        for (idx_t i = 0; i < n; ++i) parent_[i] = i;
        components_ = n;
    }

    idx_t find(idx_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];  // path halving
            x = parent_[x];
        }
        return x;
    }

    bool unite(idx_t a, idx_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) {
            idx_t t = a;
            a = b;
            b = t;
        }
        parent_[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
        --components_;
        return true;
    }

    idx_t components() const { return components_; }
    idx_t size() const { return static_cast<idx_t>(parent_.size()); }

    /// Dense labels in [0, components()).
    std::vector<idx_t> labels();

private:
    std::vector<idx_t> parent_;
    std::vector<std::uint8_t> rank_;
    idx_t components_ = 0;
};

struct MstResult {
    std::vector<Edge> edges;
    std::vector<idx_t> component;  ///< dense component label per node
    idx_t num_components = 0;
    double weight = 0.0;
    double seconds = 0.0;

    bool is_tree() const { return num_components <= 1; }
};

/// Exact MST/MSF of the given edge list. `edges` is taken by value and sorted.
MstResult mst_kruskal(std::vector<Edge> edges, idx_t n);

/// Same result, parallel rounds. Falls back to Kruskal for tiny inputs.
MstResult mst_boruvka(const std::vector<Edge>& edges, idx_t n, ThreadPool* pool = nullptr);

/// Picks Boruvka when a multi-threaded pool is available, Kruskal otherwise.
MstResult mst_from_edges(const std::vector<Edge>& edges, idx_t n, ThreadPool* pool = nullptr);

/// Prim over a materialised m x m symmetric weight matrix (row-major).
MstResult mst_dense(const float* matrix, std::size_t m);

/// Prim over a virtual complete graph. `row_fn(i, out)` must fill `out[0..n)`
/// with the weights from node i to every node (out[i] is ignored). Exactly n
/// rows are requested, one per node added to the tree, so the callback may be
/// as expensive as a full distance sweep.
using RowFn = std::function<void(idx_t, float*)>;
MstResult mst_prim_virtual(idx_t n, const RowFn& row_fn);

/// Connected components of an edge list.
std::vector<idx_t> connected_components(const std::vector<Edge>& edges, idx_t n,
                                        idx_t* num_components);

}  // namespace coresg
