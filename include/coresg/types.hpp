#pragma once
// Fundamental scalar types and the edge record shared by every layer of the
// pipeline (k-NN graph, hubness index, MST, CORE-SG).

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace coresg {

using idx_t = std::uint32_t;   // node identifier; caps the data set at ~4.29e9
using dist_t = float;          // all distances, including mutual reachability

inline constexpr idx_t kInvalidIdx = std::numeric_limits<idx_t>::max();
inline constexpr dist_t kInfDist = std::numeric_limits<dist_t>::infinity();

/// Undirected weighted edge. `u < v` is the canonical orientation; the MST and
/// CORE-SG layers rely on it to deduplicate mutual k-NN pairs.
struct Edge {
    idx_t u = kInvalidIdx;
    idx_t v = kInvalidIdx;
    dist_t w = kInfDist;

    Edge() = default;
    Edge(idx_t a, idx_t b, dist_t weight) : u(a), v(b), w(weight) {
        if (u > v) {
            idx_t t = u;
            u = v;
            v = t;
        }
    }
};

/// Total order used everywhere an MST is built. Ties are broken on the node
/// ids so that Kruskal, Boruvka and Prim all agree on the same tree, which is
/// what makes "MST_a == MST_b" a meaningful assertion in the tests.
inline bool edge_less(const Edge& a, const Edge& b) {
    if (a.w != b.w) return a.w < b.w;
    if (a.u != b.u) return a.u < b.u;
    return a.v < b.v;
}

struct EdgeLess {
    bool operator()(const Edge& a, const Edge& b) const { return edge_less(a, b); }
};

inline std::uint64_t edge_key(idx_t u, idx_t v) {
    if (u > v) {
        idx_t t = u;
        u = v;
        v = t;
    }
    return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint64_t>(v);
}

/// Total weight of an edge set (reported for MST-quality comparisons).
inline double total_weight(const std::vector<Edge>& edges) {
    double s = 0.0;
    for (const Edge& e : edges) s += static_cast<double>(e.w);
    return s;
}

}  // namespace coresg
