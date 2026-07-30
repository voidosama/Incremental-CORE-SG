#pragma once
// Boundary between the C++ code base and the CUDA translation units.
//
// Only POD types and standard headers appear here so that nvcc never has to
// parse the templated C++ side (and MSVC never has to see a .cu). Everything
// crossing this line is a raw host pointer; device allocation, streams and
// device binding live entirely inside the .cu files.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace coresg {
namespace cuda {

struct DeviceProps {
    int id = 0;
    char name[256] = {0};
    std::size_t total_mem = 0;
    std::size_t free_mem = 0;
    int sm_count = 0;
    int major = 0;
    int minor = 0;
};

/// Metric codes. Only metrics expressible as squared-L2 on (possibly
/// normalised) vectors are supported on device; the C++ side falls back to the
/// CPU backend for L1 and raw inner product.
enum MetricCode : int {
    kMetricL2 = 0,
    kMetricCosine = 1,
};

struct NndConfig {
    std::uint32_t n = 0;
    std::uint32_t dim = 0;
    std::uint32_t stride = 0;
    std::uint32_t k = 0;
    std::uint32_t sample = 16;
    std::uint32_t iters = 10;
    std::uint64_t seed = 42;
    int metric = kMetricL2;
    float delta = 0.001f;
    bool random_fill = true;  ///< top up empty slots with random candidates
};

int device_count();
bool get_device_props(int id, DeviceProps* out);

/// Message for the most recent failed call on the calling thread.
const char* last_error();

/// Dynamic NN-Descent, start to finish on `device`. The graph arrays are host
/// buffers (n*k ids/dists/flags, n counts); they are uploaded, refined and
/// copied back. Distances are returned in the final metric units.
bool nndescent_gpu(int device, const NndConfig& cfg, const float* host_vectors,
                   std::uint32_t* host_ids, float* host_dists, std::uint8_t* host_flags,
                   std::uint32_t* host_counts, std::uint64_t* out_updates);

/// Rows [r0, r1) of the m x m mutual reachability matrix; `host_out` receives
/// (r1 - r0) * m floats.
bool mrd_block_gpu(int device, const float* host_pts, std::size_t m, std::uint32_t dim,
                   std::uint32_t stride, int metric, const float* host_core, std::size_t r0,
                   std::size_t r1, float* host_out);

/// For each row i in [r0, r1), the minimum mrd to any column j != i and the
/// argmin. Lets the exact reference MST run in O(n^2) time and O(n) memory.
bool mrd_row_min_gpu(int device, const float* host_pts, std::size_t m, std::uint32_t dim,
                     std::uint32_t stride, int metric, const float* host_core,
                     const std::uint32_t* host_component, std::size_t r0, std::size_t r1,
                     float* host_min, std::uint32_t* host_argmin);

/// Best-effort peer access between the listed devices (used to keep multi-GPU
/// merges off the host when the topology allows it).
bool enable_peer_access(const std::vector<int>& devices);

/// Free and total device memory, from the driver. Note this is process
/// agnostic: it reflects everything sharing the device.
bool device_memory(int device, std::size_t* free_bytes, std::size_t* total_bytes);

/// Drop the calling thread's resident mrd operands. Call from the thread that
/// owns a backend slot; safe to call when nothing is cached.
void release_thread_cache();

}  // namespace cuda
}  // namespace coresg
