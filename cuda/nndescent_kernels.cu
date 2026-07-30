// GPU Dynamic NN-Descent.
//
// Algorithm 1 of the paper is written as a single kernel: one block per node
// does sampling, reverse-cache maintenance and the local join, and the graph
// is updated in place under per-node spin locks. There is no barrier between
// nodes inside a round -- which is the property the paper's incomplete reverse
// graph R buys, and the reason the GPU version does not need the host to
// orchestrate anything per node.
//
// The local join itself is the matrix-multiplication-shaped part: the sampled
// new neighbours form the rows, the sampled new+old neighbours the columns,
// and the distance block is accumulated through shared-memory tiles (Fig. 2 of
// the paper). Both operand tiles are stored transposed, [d][i], so that the
// 16 lanes covering a tile row read 16 consecutive banks.
//
// Everything runs in *rank* space (squared L2). Host distances are squared on
// upload and square-rooted on download, so a graph can be handed back and
// forth between the CPU and GPU backends mid-pipeline.

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

#include "coresg/cuda_api.hpp"
#include "cuda_common.cuh"

namespace coresg {
namespace cuda {
namespace {

struct GraphView {
    std::uint32_t* ids;
    float* dist;
    unsigned char* flags;
    std::uint32_t* count;
    int* locks;
    std::uint32_t n;
    std::uint32_t k;
};

struct ReverseView {
    std::uint32_t* old_ids;
    std::uint32_t* new_ids;
    std::uint32_t* old_cnt;
    std::uint32_t* new_cnt;
    std::uint32_t smp;
};

constexpr std::uint32_t kNoId = 0xFFFFFFFFu;

enum ScaleOp { kToRankL2 = 0, kFinalizeL2 = 1, kToRankCos = 2, kFinalizeCos = 3 };

__global__ void scale_kernel(float* dist, std::size_t total, int op) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float v = dist[i];
    if (!isfinite(v)) return;
    switch (op) {
        case kToRankL2: dist[i] = v * v; break;
        case kFinalizeL2: dist[i] = sqrtf(fmaxf(v, 0.0f)); break;
        case kToRankCos: dist[i] = 2.0f * v; break;
        case kFinalizeCos: dist[i] = 0.5f * v; break;
        default: break;
    }
}

__global__ void fill_kernel(float* buf, std::size_t total, float value) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) buf[i] = value;
}

__global__ void fill_u32_kernel(std::uint32_t* buf, std::size_t total, std::uint32_t value) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) buf[i] = value;
}

// ---------------------------------------------------------------------------
// Random top-up of empty neighbour slots.
// ---------------------------------------------------------------------------

constexpr int kInitThreads = 128;

__global__ void init_fill_kernel(const float* vec, std::uint32_t n, std::uint32_t stride,
                                 GraphView g, std::uint64_t seed) {
    const std::uint32_t node = blockIdx.x;
    if (node >= n || n < 2) return;

    __shared__ float red[kInitThreads];
    const int tid = threadIdx.x;
    const float* a = vec + static_cast<std::size_t>(node) * stride;

    // Every thread runs the same stream, so the candidate is block-uniform and
    // all the __syncthreads() below are reached by the whole block.
    std::uint32_t state = hash_u32(static_cast<std::uint32_t>(seed) ^ (node * 2654435761u));

    const std::uint32_t missing = g.k - g.count[node];
    const std::uint32_t attempts = missing + missing / 2u + 4u;
    for (std::uint32_t t = 0; t < attempts; ++t) {
        if (g.count[node] >= g.k) break;
        const std::uint32_t cand = rng_next(&state) % n;
        if (cand == node) continue;

        const float* b = vec + static_cast<std::size_t>(cand) * stride;
        float part = 0.0f;
        for (std::uint32_t d = tid; d < stride; d += kInitThreads) {
            const float df = a[d] - b[d];
            part += df * df;
        }
        red[tid] = part;
        __syncthreads();
        for (int s = kInitThreads / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            device_insert(node, cand, red[0], 1, g.ids, g.dist, g.flags, g.count, g.locks, g.k);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Local join (Algorithm 1). One block per node.
// ---------------------------------------------------------------------------

__global__ void local_join_kernel(const float* vec, std::uint32_t n, std::uint32_t stride,
                                  GraphView g, ReverseView r, std::uint32_t smp,
                                  const std::uint32_t* active, std::uint32_t active_n,
                                  unsigned long long* updates) {
    const std::uint32_t slot = blockIdx.x;
    if (slot >= active_n) return;
    const std::uint32_t node = active ? active[slot] : slot;
    if (node >= n) return;

    __shared__ std::uint32_t s_new[kMaxNew];
    __shared__ std::uint32_t s_old[kMaxOld];
    __shared__ int s_nnew;
    __shared__ int s_nold;
    __shared__ unsigned long long s_updates;
    __shared__ float As[kTileD][kTileR];
    __shared__ float Bs[kTileD][kTileC];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kTileC + tx;

    // --- Sampling (lines 6-14): one short critical section on this node -----
    if (tid == 0) {
        s_updates = 0;
        int nnew = 0, nold = 0;
        const std::size_t base = static_cast<std::size_t>(node) * g.k;
        const int max_new = static_cast<int>(smp) < kMaxSample ? static_cast<int>(smp)
                                                              : kMaxSample;
        const int max_old = 3 * max_new;
        lock_acquire(&g.locks[node]);
        const std::uint32_t cnt = g.count[node];
        for (std::uint32_t j = 0; j < cnt; ++j) {
            if (nnew >= max_new && nold >= max_old) break;
            const std::uint32_t id = g.ids[base + j];
            if (g.flags[base + j]) {
                if (nnew < max_new) {
                    s_new[nnew++] = id;
                    g.flags[base + j] = 0;  // "mark sampled new items as false"
                }
            } else if (nold < max_old) {
                s_old[nold++] = id;
            }
        }
        lock_release(&g.locks[node]);
        s_nnew = nnew;
        s_nold = nold;
    }
    __syncthreads();

    // --- Publish this node into its neighbours' reverse lists ---------------
    for (int t = tid; t < s_nold; t += kJoinThreads) {
        const std::uint32_t u = s_old[t];
        if (u >= n) continue;
        const std::uint32_t pos = atomicAdd(&r.old_cnt[u], 1u);
        if (pos < r.smp) {
            r.old_ids[static_cast<std::size_t>(u) * r.smp + pos] = node;
        } else {
            atomicSub(&r.old_cnt[u], 1u);
        }
    }
    for (int t = tid; t < s_nnew; t += kJoinThreads) {
        const std::uint32_t u = s_new[t];
        if (u >= n) continue;
        const std::uint32_t pos = atomicAdd(&r.new_cnt[u], 1u);
        if (pos < r.smp) {
            r.new_ids[static_cast<std::size_t>(u) * r.smp + pos] = node;
        } else {
            atomicSub(&r.new_cnt[u], 1u);
        }
    }
    __syncthreads();

    // --- Absorb our own reverse lists and clear them (lines 15-18) ----------
    if (tid == 0) {
        std::uint32_t c = atomicExch(&r.old_cnt[node], 0u);
        if (c > r.smp) c = r.smp;
        int nold = s_nold;
        for (std::uint32_t j = 0; j < c && nold < kMaxOld; ++j) {
            const std::uint32_t v = r.old_ids[static_cast<std::size_t>(node) * r.smp + j];
            if (v != kNoId && v < n) s_old[nold++] = v;
        }
        s_nold = nold;

        std::uint32_t cn = atomicExch(&r.new_cnt[node], 0u);
        if (cn > r.smp) cn = r.smp;
        int nnew = s_nnew;
        for (std::uint32_t j = 0; j < cn && nnew < kMaxNew; ++j) {
            const std::uint32_t v = r.new_ids[static_cast<std::size_t>(node) * r.smp + j];
            if (v != kNoId && v < n) s_new[nnew++] = v;
        }
        s_nnew = nnew;
    }
    __syncthreads();

    const int nA = s_nnew;
    const int nT = s_nnew + s_nold;
    if (nA == 0) return;  // block-uniform

    // --- Tiled local join --------------------------------------------------
    const int row_tiles = (nA + kTileR - 1) / kTileR;
    const int col_tiles = (nT + kTileC - 1) / kTileC;

    for (int rt = 0; rt < row_tiles; ++rt) {
        for (int ct = 0; ct < col_tiles; ++ct) {
            // Whole tile below the diagonal: every pair would be a duplicate.
            if ((ct + 1) * kTileC <= rt * kTileR) continue;

            float acc = 0.0f;
            for (std::uint32_t d0 = 0; d0 < stride; d0 += kTileD) {
                __syncthreads();
                for (int e = tid; e < kTileR * kTileD; e += kJoinThreads) {
                    const int rr = e / kTileD;
                    const int dd = e % kTileD;
                    const int idx = rt * kTileR + rr;
                    float v = 0.0f;
                    if (idx < nA && d0 + dd < stride) {
                        const std::uint32_t id = s_new[idx];
                        if (id < n) v = vec[static_cast<std::size_t>(id) * stride + d0 + dd];
                    }
                    As[dd][rr] = v;
                }
                for (int e = tid; e < kTileC * kTileD; e += kJoinThreads) {
                    const int cc = e / kTileD;
                    const int dd = e % kTileD;
                    const int idx = ct * kTileC + cc;
                    float v = 0.0f;
                    if (idx < nT && d0 + dd < stride) {
                        const std::uint32_t id =
                            (idx < nA) ? s_new[idx] : s_old[idx - nA];
                        if (id < n) v = vec[static_cast<std::size_t>(id) * stride + d0 + dd];
                    }
                    Bs[dd][cc] = v;
                }
                __syncthreads();
#pragma unroll
                for (int t = 0; t < kTileD; ++t) {
                    const float df = As[t][ty] - Bs[t][tx];
                    acc += df * df;
                }
            }

            const int gr = rt * kTileR + ty;
            const int gc = ct * kTileC + tx;
            if (gr < nA && gc < nT && gc > gr) {
                const std::uint32_t u1 = s_new[gr];
                const std::uint32_t u2 = (gc < nA) ? s_new[gc] : s_old[gc - nA];
                if (u1 != u2 && u1 < n && u2 < n) {
                    int upd = device_insert(u1, u2, acc, 1, g.ids, g.dist, g.flags, g.count,
                                            g.locks, g.k);
                    upd += device_insert(u2, u1, acc, 1, g.ids, g.dist, g.flags, g.count,
                                         g.locks, g.k);
                    if (upd) atomicAdd(&s_updates, static_cast<unsigned long long>(upd));
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0 && s_updates) atomicAdd(updates, s_updates);
}

inline int blocks_for(std::size_t total, int threads) {
    return static_cast<int>((total + threads - 1) / threads);
}

}  // namespace

bool nndescent_gpu(int device, const NndConfig& cfg, const float* host_vectors,
                   std::uint32_t* host_ids, float* host_dists, std::uint8_t* host_flags,
                   std::uint32_t* host_counts, std::uint64_t* out_updates) {
    if (cfg.n == 0 || cfg.k == 0) return true;
    if (cfg.metric != kMetricL2 && cfg.metric != kMetricCosine) {
        set_last_error("nndescent_gpu: unsupported metric for the device path");
        return false;
    }
    CORESG_CUDA_TRY(cudaSetDevice(device));

    const std::size_t n = cfg.n;
    const std::size_t k = cfg.k;
    const std::size_t vec_floats = n * cfg.stride;
    const std::size_t graph_slots = n * k;
    const std::uint32_t smp = cfg.sample > static_cast<std::uint32_t>(kMaxSample)
                                  ? static_cast<std::uint32_t>(kMaxSample)
                                  : cfg.sample;

    float* d_vec = nullptr;
    std::uint32_t* d_ids = nullptr;
    float* d_dist = nullptr;
    unsigned char* d_flags = nullptr;
    std::uint32_t* d_count = nullptr;
    int* d_locks = nullptr;
    std::uint32_t* d_rold = nullptr;
    std::uint32_t* d_rnew = nullptr;
    std::uint32_t* d_rold_cnt = nullptr;
    std::uint32_t* d_rnew_cnt = nullptr;
    unsigned long long* d_updates = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_vec);
        cudaFree(d_ids);
        cudaFree(d_dist);
        cudaFree(d_flags);
        cudaFree(d_count);
        cudaFree(d_locks);
        cudaFree(d_rold);
        cudaFree(d_rnew);
        cudaFree(d_rold_cnt);
        cudaFree(d_rnew_cnt);
        cudaFree(d_updates);
    };

#define CORESG_ALLOC(ptr, bytes)                                                     \
    do {                                                                             \
        cudaError_t _e = cudaMalloc((void**)&(ptr), (bytes));                        \
        if (_e != cudaSuccess) {                                                     \
            set_last_error(std::string("cudaMalloc " #ptr " -> ") +                  \
                           cudaGetErrorString(_e));                                  \
            cleanup();                                                               \
            return false;                                                            \
        }                                                                            \
    } while (0)

    CORESG_ALLOC(d_vec, vec_floats * sizeof(float));
    CORESG_ALLOC(d_ids, graph_slots * sizeof(std::uint32_t));
    CORESG_ALLOC(d_dist, graph_slots * sizeof(float));
    CORESG_ALLOC(d_flags, graph_slots * sizeof(unsigned char));
    CORESG_ALLOC(d_count, n * sizeof(std::uint32_t));
    CORESG_ALLOC(d_locks, n * sizeof(int));
    CORESG_ALLOC(d_rold, n * smp * sizeof(std::uint32_t));
    CORESG_ALLOC(d_rnew, n * smp * sizeof(std::uint32_t));
    CORESG_ALLOC(d_rold_cnt, n * sizeof(std::uint32_t));
    CORESG_ALLOC(d_rnew_cnt, n * sizeof(std::uint32_t));
    CORESG_ALLOC(d_updates, sizeof(unsigned long long));
#undef CORESG_ALLOC

#define CORESG_TRY_CLEAN(expr)                                                       \
    do {                                                                             \
        cudaError_t _e = (expr);                                                     \
        if (_e != cudaSuccess) {                                                     \
            set_last_error(std::string(#expr) + " -> " + cudaGetErrorString(_e));    \
            cleanup();                                                               \
            return false;                                                            \
        }                                                                            \
    } while (0)

    CORESG_TRY_CLEAN(cudaMemcpy(d_vec, host_vectors, vec_floats * sizeof(float),
                                cudaMemcpyHostToDevice));
    CORESG_TRY_CLEAN(cudaMemcpy(d_ids, host_ids, graph_slots * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice));
    CORESG_TRY_CLEAN(cudaMemcpy(d_dist, host_dists, graph_slots * sizeof(float),
                                cudaMemcpyHostToDevice));
    CORESG_TRY_CLEAN(cudaMemcpy(d_flags, host_flags, graph_slots * sizeof(unsigned char),
                                cudaMemcpyHostToDevice));
    CORESG_TRY_CLEAN(cudaMemcpy(d_count, host_counts, n * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice));
    CORESG_TRY_CLEAN(cudaMemset(d_locks, 0, n * sizeof(int)));
    CORESG_TRY_CLEAN(cudaMemset(d_rold_cnt, 0, n * sizeof(std::uint32_t)));
    CORESG_TRY_CLEAN(cudaMemset(d_rnew_cnt, 0, n * sizeof(std::uint32_t)));

    fill_u32_kernel<<<blocks_for(n * smp, 256), 256>>>(d_rold, n * smp, kNoId);
    fill_u32_kernel<<<blocks_for(n * smp, 256), 256>>>(d_rnew, n * smp, kNoId);

    // Host distances are metric values; the descent works in rank space.
    scale_kernel<<<blocks_for(graph_slots, 256), 256>>>(
        d_dist, graph_slots, cfg.metric == kMetricCosine ? kToRankCos : kToRankL2);

    GraphView g{d_ids, d_dist, d_flags, d_count, d_locks, cfg.n, cfg.k};
    ReverseView r{d_rold, d_rnew, d_rold_cnt, d_rnew_cnt, smp};

    if (cfg.random_fill) {
        init_fill_kernel<<<static_cast<int>(n), kInitThreads>>>(d_vec, cfg.n, cfg.stride, g,
                                                               cfg.seed);
        CORESG_TRY_CLEAN(cudaGetLastError());
    }

    const dim3 join_block(kTileC, kTileR);
    const double stop_threshold = double(cfg.delta) * double(cfg.k) * double(cfg.n);
    std::uint64_t total_updates = 0;

    for (std::uint32_t iter = 0; iter < cfg.iters; ++iter) {
        CORESG_TRY_CLEAN(cudaMemset(d_updates, 0, sizeof(unsigned long long)));
        local_join_kernel<<<static_cast<int>(n), join_block>>>(d_vec, cfg.n, cfg.stride, g, r,
                                                              smp, nullptr, cfg.n, d_updates);
        CORESG_TRY_CLEAN(cudaGetLastError());

        unsigned long long round_updates = 0;
        CORESG_TRY_CLEAN(cudaMemcpy(&round_updates, d_updates, sizeof(round_updates),
                                    cudaMemcpyDeviceToHost));
        total_updates += round_updates;
        if (double(round_updates) <= stop_threshold) break;
    }

    scale_kernel<<<blocks_for(graph_slots, 256), 256>>>(
        d_dist, graph_slots, cfg.metric == kMetricCosine ? kFinalizeCos : kFinalizeL2);
    CORESG_TRY_CLEAN(cudaGetLastError());

    CORESG_TRY_CLEAN(cudaMemcpy(host_ids, d_ids, graph_slots * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost));
    CORESG_TRY_CLEAN(cudaMemcpy(host_dists, d_dist, graph_slots * sizeof(float),
                                cudaMemcpyDeviceToHost));
    CORESG_TRY_CLEAN(cudaMemcpy(host_flags, d_flags, graph_slots * sizeof(unsigned char),
                                cudaMemcpyDeviceToHost));
    CORESG_TRY_CLEAN(cudaMemcpy(host_counts, d_count, n * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost));
#undef CORESG_TRY_CLEAN

    cleanup();
    if (out_updates) *out_updates = total_updates;
    return true;
}

}  // namespace cuda
}  // namespace coresg
