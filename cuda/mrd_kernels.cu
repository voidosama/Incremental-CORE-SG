// Dense mutual-reachability blocks on device.
//
// Used twice in the pipeline, both times with the same shape: a row band of an
// m x m matrix.
//
//   * the anti-hub MST -- m is d*ln(|E|), so the band is small and several
//     GPUs each take a slice of the rows;
//   * the exact reference MST -- Prim asks for one row at a time over the full
//     data set, which is O(n) memory instead of O(n^2).

#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include "coresg/cuda_api.hpp"
#include "cuda_common.cuh"

namespace coresg {
namespace cuda {
namespace {

constexpr int kMrdTile = 16;
constexpr int kMrdDepth = 32;

__device__ __forceinline__ float finalize_rank(float acc, int metric) {
    return (metric == kMetricCosine) ? 0.5f * acc : sqrtf(fmaxf(acc, 0.0f));
}

__global__ void mrd_block_kernel(const float* pts, std::uint32_t m, std::uint32_t stride,
                                 const float* core, std::uint32_t r0, std::uint32_t rows,
                                 int metric, float* out) {
    __shared__ float As[kMrdDepth][kMrdTile];
    __shared__ float Bs[kMrdDepth][kMrdTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kMrdTile + tx;

    const int row_base = blockIdx.y * kMrdTile;
    const int col_base = blockIdx.x * kMrdTile;

    float acc = 0.0f;
    for (std::uint32_t d0 = 0; d0 < stride; d0 += kMrdDepth) {
        __syncthreads();
        for (int e = tid; e < kMrdTile * kMrdDepth; e += kMrdTile * kMrdTile) {
            const int rr = e / kMrdDepth;
            const int dd = e % kMrdDepth;
            const std::uint32_t gi = r0 + row_base + rr;
            float v = 0.0f;
            if (row_base + rr < static_cast<int>(rows) && gi < m && d0 + dd < stride) {
                v = pts[static_cast<std::size_t>(gi) * stride + d0 + dd];
            }
            As[dd][rr] = v;
        }
        for (int e = tid; e < kMrdTile * kMrdDepth; e += kMrdTile * kMrdTile) {
            const int cc = e / kMrdDepth;
            const int dd = e % kMrdDepth;
            const int gj = col_base + cc;
            float v = 0.0f;
            if (gj < static_cast<int>(m) && d0 + dd < stride) {
                v = pts[static_cast<std::size_t>(gj) * stride + d0 + dd];
            }
            Bs[dd][cc] = v;
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < kMrdDepth; ++t) {
            const float df = As[t][ty] - Bs[t][tx];
            acc += df * df;
        }
    }

    const int lr = row_base + ty;
    const int gc = col_base + tx;
    if (lr >= static_cast<int>(rows) || gc >= static_cast<int>(m)) return;
    const std::uint32_t gr = r0 + lr;

    float value;
    if (gr == static_cast<std::uint32_t>(gc)) {
        value = 0.0f;
    } else {
        const float d = finalize_rank(acc, metric);
        value = fmaxf(d, fmaxf(core[gr], core[gc]));
    }
    out[static_cast<std::size_t>(lr) * m + gc] = value;
}

/// One block per row; reduces to (min, argmin) over columns in a different
/// component. `component == nullptr` means "every column except the diagonal".
__global__ void mrd_row_min_kernel(const float* pts, std::uint32_t m, std::uint32_t stride,
                                   const float* core, const std::uint32_t* component,
                                   std::uint32_t r0, std::uint32_t rows, int metric,
                                   float* out_min, std::uint32_t* out_arg) {
    const int lr = blockIdx.x;
    if (lr >= static_cast<int>(rows)) return;
    const std::uint32_t gr = r0 + lr;
    if (gr >= m) return;

    __shared__ float s_val[256];
    __shared__ std::uint32_t s_arg[256];

    const int tid = threadIdx.x;
    const float* a = pts + static_cast<std::size_t>(gr) * stride;
    const std::uint32_t my_comp = component ? component[gr] : 0u;

    float best = INFINITY;
    std::uint32_t best_j = 0xFFFFFFFFu;

    for (std::uint32_t j = tid; j < m; j += blockDim.x) {
        if (j == gr) continue;
        if (component && component[j] == my_comp) continue;
        const float* b = pts + static_cast<std::size_t>(j) * stride;
        float acc = 0.0f;
        for (std::uint32_t d = 0; d < stride; ++d) {
            const float df = a[d] - b[d];
            acc += df * df;
        }
        const float w = fmaxf(finalize_rank(acc, metric), fmaxf(core[gr], core[j]));
        if (w < best) {
            best = w;
            best_j = j;
        }
    }
    s_val[tid] = best;
    s_arg[tid] = best_j;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] < s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_arg[tid] = s_arg[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_min[lr] = s_val[0];
        out_arg[lr] = s_arg[0];
    }
}

/// Per-thread residency cache.
///
/// Prim over a virtual complete graph asks for one row at a time, n times in a
/// row, always over the same points. Re-uploading the data set for each row
/// would turn an O(n^2 d) computation into an O(n^2 stride) transfer, so the
/// operands stay resident and are only replaced when the caller's pointers
/// change. Each backend slot is pinned to one host thread, so thread_local is
/// exactly the right scope.
struct MrdCache {
    int device = -1;
    const float* pts_key = nullptr;
    const float* core_key = nullptr;
    std::size_t m = 0;
    std::uint32_t stride = 0;
    float* d_pts = nullptr;
    float* d_core = nullptr;
    float* d_out = nullptr;
    std::size_t out_capacity = 0;

    void release() {
        if (d_pts) cudaFree(d_pts);
        if (d_core) cudaFree(d_core);
        if (d_out) cudaFree(d_out);
        d_pts = d_core = d_out = nullptr;
        out_capacity = 0;
        pts_key = core_key = nullptr;
        m = 0;
        stride = 0;
    }

    ~MrdCache() {
        // Best effort: at process teardown the context may already be gone.
        release();
        cudaGetLastError();
    }
};

thread_local MrdCache t_cache;

bool ensure_resident(int device, const float* host_pts, std::size_t m, std::uint32_t stride,
                     const float* host_core) {
    const bool same = t_cache.device == device && t_cache.pts_key == host_pts &&
                      t_cache.core_key == host_core && t_cache.m == m &&
                      t_cache.stride == stride && t_cache.d_pts && t_cache.d_core;
    if (same) return true;

    t_cache.release();
    t_cache.device = device;
    CORESG_CUDA_TRY(cudaMalloc((void**)&t_cache.d_pts, m * stride * sizeof(float)));
    CORESG_CUDA_TRY(cudaMalloc((void**)&t_cache.d_core, m * sizeof(float)));
    CORESG_CUDA_TRY(cudaMemcpy(t_cache.d_pts, host_pts, m * stride * sizeof(float),
                               cudaMemcpyHostToDevice));
    CORESG_CUDA_TRY(
        cudaMemcpy(t_cache.d_core, host_core, m * sizeof(float), cudaMemcpyHostToDevice));
    t_cache.pts_key = host_pts;
    t_cache.core_key = host_core;
    t_cache.m = m;
    t_cache.stride = stride;
    return true;
}

bool ensure_out(std::size_t floats) {
    if (t_cache.out_capacity >= floats && t_cache.d_out) return true;
    if (t_cache.d_out) cudaFree(t_cache.d_out);
    t_cache.d_out = nullptr;
    CORESG_CUDA_TRY(cudaMalloc((void**)&t_cache.d_out, floats * sizeof(float)));
    t_cache.out_capacity = floats;
    return true;
}

}  // namespace

void release_thread_cache() {
    t_cache.release();
    cudaGetLastError();
}

bool mrd_block_gpu(int device, const float* host_pts, std::size_t m, std::uint32_t dim,
                   std::uint32_t stride, int metric, const float* host_core, std::size_t r0,
                   std::size_t r1, float* host_out) {
    (void)dim;
    if (m == 0 || r1 <= r0) return true;
    if (metric != kMetricL2 && metric != kMetricCosine) {
        set_last_error("mrd_block_gpu: unsupported metric for the device path");
        return false;
    }
    CORESG_CUDA_TRY(cudaSetDevice(device));

    const std::size_t rows = r1 - r0;
    if (!ensure_resident(device, host_pts, m, stride, host_core)) return false;
    if (!ensure_out(rows * m)) return false;

    const dim3 block(kMrdTile, kMrdTile);
    const dim3 grid(static_cast<unsigned>((m + kMrdTile - 1) / kMrdTile),
                    static_cast<unsigned>((rows + kMrdTile - 1) / kMrdTile));
    mrd_block_kernel<<<grid, block>>>(t_cache.d_pts, static_cast<std::uint32_t>(m), stride,
                                      t_cache.d_core, static_cast<std::uint32_t>(r0),
                                      static_cast<std::uint32_t>(rows), metric, t_cache.d_out);
    CORESG_CUDA_TRY(cudaGetLastError());
    CORESG_CUDA_TRY(
        cudaMemcpy(host_out, t_cache.d_out, rows * m * sizeof(float), cudaMemcpyDeviceToHost));
    return true;
}

bool mrd_row_min_gpu(int device, const float* host_pts, std::size_t m, std::uint32_t dim,
                     std::uint32_t stride, int metric, const float* host_core,
                     const std::uint32_t* host_component, std::size_t r0, std::size_t r1,
                     float* host_min, std::uint32_t* host_argmin) {
    (void)dim;
    if (m == 0 || r1 <= r0) return true;
    CORESG_CUDA_TRY(cudaSetDevice(device));

    const std::size_t rows = r1 - r0;
    float* d_pts = nullptr;
    float* d_core = nullptr;
    std::uint32_t* d_comp = nullptr;
    float* d_min = nullptr;
    std::uint32_t* d_arg = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_pts);
        cudaFree(d_core);
        cudaFree(d_comp);
        cudaFree(d_min);
        cudaFree(d_arg);
    };

#define CORESG_TRY_CLEAN(expr)                                                    \
    do {                                                                          \
        cudaError_t _e = (expr);                                                  \
        if (_e != cudaSuccess) {                                                  \
            set_last_error(std::string(#expr) + " -> " + cudaGetErrorString(_e)); \
            cleanup();                                                            \
            return false;                                                         \
        }                                                                         \
    } while (0)

    CORESG_TRY_CLEAN(cudaMalloc((void**)&d_pts, m * stride * sizeof(float)));
    CORESG_TRY_CLEAN(cudaMalloc((void**)&d_core, m * sizeof(float)));
    CORESG_TRY_CLEAN(cudaMalloc((void**)&d_min, rows * sizeof(float)));
    CORESG_TRY_CLEAN(cudaMalloc((void**)&d_arg, rows * sizeof(std::uint32_t)));
    CORESG_TRY_CLEAN(
        cudaMemcpy(d_pts, host_pts, m * stride * sizeof(float), cudaMemcpyHostToDevice));
    CORESG_TRY_CLEAN(cudaMemcpy(d_core, host_core, m * sizeof(float), cudaMemcpyHostToDevice));
    if (host_component) {
        CORESG_TRY_CLEAN(cudaMalloc((void**)&d_comp, m * sizeof(std::uint32_t)));
        CORESG_TRY_CLEAN(cudaMemcpy(d_comp, host_component, m * sizeof(std::uint32_t),
                                    cudaMemcpyHostToDevice));
    }

    mrd_row_min_kernel<<<static_cast<int>(rows), 256>>>(
        d_pts, static_cast<std::uint32_t>(m), stride, d_core, d_comp,
        static_cast<std::uint32_t>(r0), static_cast<std::uint32_t>(rows), metric, d_min, d_arg);
    CORESG_TRY_CLEAN(cudaGetLastError());
    CORESG_TRY_CLEAN(cudaMemcpy(host_min, d_min, rows * sizeof(float), cudaMemcpyDeviceToHost));
    CORESG_TRY_CLEAN(
        cudaMemcpy(host_argmin, d_arg, rows * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
#undef CORESG_TRY_CLEAN

    cleanup();
    return true;
}

}  // namespace cuda
}  // namespace coresg
