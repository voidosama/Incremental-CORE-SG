#include "coresg/mst.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>

#include "coresg/thread_pool.hpp"

namespace coresg {
namespace {

/// Packs (weight, edge index) into one 64-bit word so that a plain integer
/// minimum is also the correct edge minimum. Non-negative IEEE-754 floats
/// compare identically to their bit patterns, and mutual reachability
/// distances of a proper dissimilarity are non-negative, so this is exact.
inline std::uint64_t pack_edge(float w, std::uint32_t index) {
    std::uint32_t bits;
    std::memcpy(&bits, &w, sizeof(bits));
    if (w < 0.0f) bits = ~bits;  // defensive: keeps the order sane if it ever happens
    return (static_cast<std::uint64_t>(bits) << 32) | index;
}

inline std::uint32_t unpack_index(std::uint64_t packed) {
    return static_cast<std::uint32_t>(packed & 0xFFFFFFFFull);
}

inline void atomic_min_u64(std::atomic<std::uint64_t>& target, std::uint64_t value) {
    std::uint64_t cur = target.load(std::memory_order_relaxed);
    while (value < cur) {
        if (target.compare_exchange_weak(cur, value, std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
            return;
        }
    }
}

void finish(MstResult* res, UnionFind* uf) {
    res->component = uf->labels();
    res->num_components = uf->components();
    res->weight = total_weight(res->edges);
}

}  // namespace

std::vector<idx_t> UnionFind::labels() {
    const idx_t n = size();
    std::vector<idx_t> out(n, kInvalidIdx);
    std::vector<idx_t> remap(n, kInvalidIdx);
    idx_t next = 0;
    for (idx_t i = 0; i < n; ++i) {
        const idx_t r = find(i);
        if (remap[r] == kInvalidIdx) remap[r] = next++;
        out[i] = remap[r];
    }
    return out;
}

MstResult mst_kruskal(std::vector<Edge> edges, idx_t n) {
    const auto t0 = std::chrono::steady_clock::now();
    MstResult res;
    UnionFind uf(n);
    std::sort(edges.begin(), edges.end(), EdgeLess());
    res.edges.reserve(n ? n - 1 : 0);
    for (const Edge& e : edges) {
        if (e.u >= n || e.v >= n) continue;
        if (uf.unite(e.u, e.v)) {
            res.edges.push_back(e);
            if (res.edges.size() + 1 == std::size_t(n)) break;
        }
    }
    finish(&res, &uf);
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

MstResult mst_boruvka(const std::vector<Edge>& edges, idx_t n, ThreadPool* pool) {
    const auto t0 = std::chrono::steady_clock::now();
    if (n == 0) return MstResult{};
    if (edges.size() < 4096 || !pool || pool->size() <= 1) {
        MstResult r = mst_kruskal(edges, n);
        r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return r;
    }

    MstResult res;
    UnionFind uf(n);
    res.edges.reserve(n - 1);

    std::vector<idx_t> comp(n);
    std::unique_ptr<std::atomic<std::uint64_t>[]> best(new std::atomic<std::uint64_t>[n]);
    const std::uint64_t kNone = std::numeric_limits<std::uint64_t>::max();

    // Edges are filtered in place each round: once both endpoints share a
    // component the edge can never be chosen again.
    std::vector<std::uint32_t> live(edges.size());
    for (std::size_t i = 0; i < live.size(); ++i) live[i] = static_cast<std::uint32_t>(i);

    while (uf.components() > 1 && !live.empty()) {
        for (idx_t i = 0; i < n; ++i) comp[i] = uf.find(i);
        for (idx_t i = 0; i < n; ++i) best[i].store(kNone, std::memory_order_relaxed);

        std::vector<std::vector<std::uint32_t>> kept(pool->size());
        pool->parallel_ranges(live.size(), [&](std::size_t lo, std::size_t hi, int slot) {
            std::vector<std::uint32_t>& keep = kept[slot];
            keep.reserve((hi - lo) / 2 + 1);
            for (std::size_t t = lo; t < hi; ++t) {
                const std::uint32_t ei = live[t];
                const Edge& e = edges[ei];
                if (e.u >= n || e.v >= n) continue;
                const idx_t cu = comp[e.u];
                const idx_t cv = comp[e.v];
                if (cu == cv) continue;
                keep.push_back(ei);
                const std::uint64_t packed = pack_edge(e.w, ei);
                atomic_min_u64(best[cu], packed);
                atomic_min_u64(best[cv], packed);
            }
        });

        std::size_t total = 0;
        for (const auto& kp : kept) total += kp.size();
        std::vector<std::uint32_t> next_live;
        next_live.reserve(total);
        for (auto& kp : kept) {
            next_live.insert(next_live.end(), kp.begin(), kp.end());
            std::vector<std::uint32_t>().swap(kp);
        }
        live.swap(next_live);

        // Serial merge phase. Components are at most n, and each round at
        // least halves them, so this costs O(n) over O(log n) rounds.
        bool progress = false;
        for (idx_t c = 0; c < n; ++c) {
            const std::uint64_t packed = best[c].load(std::memory_order_relaxed);
            if (packed == kNone) continue;
            const Edge& e = edges[unpack_index(packed)];
            if (uf.unite(e.u, e.v)) {
                res.edges.push_back(e);
                progress = true;
            }
        }
        if (!progress) break;
    }

    finish(&res, &uf);
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

MstResult mst_from_edges(const std::vector<Edge>& edges, idx_t n, ThreadPool* pool) {
    if (pool && pool->size() > 1 && edges.size() >= 4096) return mst_boruvka(edges, n, pool);
    return mst_kruskal(edges, n);
}

MstResult mst_dense(const float* matrix, std::size_t m) {
    return mst_prim_virtual(static_cast<idx_t>(m), [matrix, m](idx_t i, float* out) {
        std::memcpy(out, matrix + std::size_t(i) * m, m * sizeof(float));
    });
}

MstResult mst_prim_virtual(idx_t n, const RowFn& row_fn) {
    const auto t0 = std::chrono::steady_clock::now();
    MstResult res;
    if (n == 0) {
        res.num_components = 0;
        return res;
    }
    if (n == 1) {
        res.component.assign(1, 0);
        res.num_components = 1;
        return res;
    }

    std::vector<float> best(n, std::numeric_limits<float>::infinity());
    std::vector<idx_t> from(n, kInvalidIdx);
    std::vector<std::uint8_t> in_tree(n, 0);
    std::vector<float> row(n);

    UnionFind uf(n);
    res.edges.reserve(n - 1);

    idx_t current = 0;
    in_tree[0] = 1;
    best[0] = 0.0f;

    for (idx_t step = 1; step < n; ++step) {
        row_fn(current, row.data());
        for (idx_t j = 0; j < n; ++j) {
            if (in_tree[j]) continue;
            const float w = row[j];
            if (w < best[j]) {
                best[j] = w;
                from[j] = current;
            }
        }
        idx_t pick = kInvalidIdx;
        float pick_w = std::numeric_limits<float>::infinity();
        for (idx_t j = 0; j < n; ++j) {
            if (in_tree[j]) continue;
            if (best[j] < pick_w || (best[j] == pick_w && pick == kInvalidIdx)) {
                pick_w = best[j];
                pick = j;
            }
        }
        if (pick == kInvalidIdx) break;  // graph is disconnected
        in_tree[pick] = 1;
        if (from[pick] != kInvalidIdx && pick_w < std::numeric_limits<float>::infinity()) {
            res.edges.emplace_back(from[pick], pick, pick_w);
            uf.unite(from[pick], pick);
        }
        current = pick;
    }

    finish(&res, &uf);
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

std::vector<idx_t> connected_components(const std::vector<Edge>& edges, idx_t n,
                                        idx_t* num_components) {
    UnionFind uf(n);
    for (const Edge& e : edges) {
        if (e.u < n && e.v < n) uf.unite(e.u, e.v);
    }
    std::vector<idx_t> labels = uf.labels();
    if (num_components) *num_components = uf.components();
    return labels;
}

}  // namespace coresg
