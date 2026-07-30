#pragma once
// Memory accounting.
//
// Claim B ("this scales across devices") cannot rest on timing alone: the
// property that actually makes the instance design scale is that per-device
// memory stays flat as devices are added, because an instance only ever holds
// |sample| + |batch| vectors rather than the whole data set. That has to be
// measured, not asserted.
//
// Two views are reported, and they answer different questions:
//
//   * measured   -- host resident set (current and peak) and, per CUDA device,
//                   used/total from the driver. Catches allocator overhead and
//                   fragmentation, but the device figure is process-agnostic:
//                   it includes anything else sharing the GPU.
//   * analytic   -- what an instance provably needs, from its own dimensions.
//                   Exact, deterministic, and independent of what else is
//                   running, which makes it the right number for a capacity
//                   argument ("this configuration cannot fit on an 8 GB card").

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace coresg {

struct HostMemory {
    std::size_t current_bytes = 0;
    std::size_t peak_bytes = 0;
};

/// Resident set of this process. Returns zeros where unsupported.
HostMemory host_memory();

struct DeviceMemory {
    int id = 0;
    std::size_t used_bytes = 0;
    std::size_t total_bytes = 0;
};

/// Per visible CUDA device. Empty without CUDA.
std::vector<DeviceMemory> device_memory();

/// Bytes one NN-Descent instance needs resident on a device:
/// vectors + graph (ids/dists/flags) + counts + locks + the reverse cache.
/// This is the figure that decides whether a configuration fits.
std::size_t instance_device_bytes(std::size_t n, std::uint32_t stride, std::uint32_t k,
                                  std::uint32_t sample);

/// Per-point cost, i.e. instance_device_bytes / n. Useful for "how many points
/// fit in 8 GB".
double device_bytes_per_point(std::uint32_t stride, std::uint32_t k, std::uint32_t sample);

std::string format_bytes(std::size_t bytes);

}  // namespace coresg
