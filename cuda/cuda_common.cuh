#pragma once
// Shared device-side helpers.
//
// Tiling constants. A local join compares the sampled new neighbours (rows)
// against the sampled new *and* old neighbours (columns). With smpN capped at
// 32 the worst case is 64 rows (32 sampled + 32 reverse) and 192 columns
// (64 new + 128 old), which is what the fixed-size shared buffers below are
// sized for.

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace coresg {
namespace cuda {

void set_last_error(const std::string& msg);

#define CORESG_CUDA_TRY(expr)                                                       \
    do {                                                                            \
        cudaError_t _err = (expr);                                                  \
        if (_err != cudaSuccess) {                                                  \
            ::coresg::cuda::set_last_error(std::string(#expr) + " -> " +            \
                                           cudaGetErrorString(_err));               \
            return false;                                                           \
        }                                                                           \
    } while (0)

constexpr int kMaxSample = 32;
constexpr int kMaxNew = 2 * kMaxSample;   // sampled new + reverse new
constexpr int kMaxOld = 4 * kMaxSample;   // 3*smpN sampled old + reverse old
constexpr int kMaxCand = kMaxNew + kMaxOld;

constexpr int kTileR = 16;
constexpr int kTileC = 16;
constexpr int kTileD = 32;
constexpr int kJoinThreads = kTileR * kTileC;  // 256

// ---------------------------------------------------------------------------
// Per-node spin lock. Safe because independent thread scheduling (sm_70+)
// guarantees a warp makes progress even when some lanes hold a lock others
// are waiting on. CMAKE_CUDA_ARCHITECTURES must therefore stay >= 70.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void lock_acquire(int* lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(32);
#endif
    }
    __threadfence();
}

__device__ __forceinline__ void lock_release(int* lock) {
    __threadfence();
    atomicExch(lock, 0);
}

__device__ __forceinline__ std::uint32_t hash_u32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ std::uint32_t rng_next(std::uint32_t* state) {
    *state = hash_u32(*state + 0x9E3779B9u);
    return *state;
}

/// Sorted-list insertion, identical in behaviour to KnnGraph::try_insert on
/// the host so that a graph can move between backends mid-pipeline.
/// Distances are in *rank* space (squared L2) throughout the descent.
__device__ inline int device_insert(std::uint32_t node, std::uint32_t cand, float d,
                                    unsigned char is_new, std::uint32_t* ids, float* dist,
                                    unsigned char* flags, std::uint32_t* count, int* locks,
                                    std::uint32_t k) {
    if (cand == node) return 0;
    const std::size_t base = static_cast<std::size_t>(node) * k;
    // Unfilled slots hold +inf, so this covers "not full yet" as well.
    if (!(d < dist[base + k - 1])) return 0;

    lock_acquire(&locks[node]);
    const std::uint32_t cnt = count[node];
    if (cnt == k && !(d < dist[base + k - 1])) {
        lock_release(&locks[node]);
        return 0;
    }
    for (std::uint32_t j = 0; j < cnt; ++j) {
        if (ids[base + j] == cand) {
            lock_release(&locks[node]);
            return 0;
        }
    }
    std::uint32_t start;
    if (cnt < k) {
        start = cnt;
        count[node] = cnt + 1;
    } else {
        start = k - 1;
    }
    std::uint32_t pos = start;
    while (pos > 0 && dist[base + pos - 1] > d) {
        dist[base + pos] = dist[base + pos - 1];
        ids[base + pos] = ids[base + pos - 1];
        flags[base + pos] = flags[base + pos - 1];
        --pos;
    }
    dist[base + pos] = d;
    ids[base + pos] = cand;
    flags[base + pos] = is_new;
    lock_release(&locks[node]);
    return 1;
}

}  // namespace cuda
}  // namespace coresg
