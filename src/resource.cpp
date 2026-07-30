#include "coresg/resource.hpp"

#include <cstdio>
#include <sstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// psapi.h must follow windows.h.
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <cstring>
#include <fstream>
#endif

#if CORESG_WITH_CUDA
#include "coresg/cuda_api.hpp"
#endif

namespace coresg {

HostMemory host_memory() {
    HostMemory m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    std::memset(&pmc, 0, sizeof(pmc));
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        m.current_bytes = static_cast<std::size_t>(pmc.WorkingSetSize);
        m.peak_bytes = static_cast<std::size_t>(pmc.PeakWorkingSetSize);
    }
#else
    // VmRSS is the current resident set, VmHWM its high-water mark.
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        const char* p = nullptr;
        std::size_t* dst = nullptr;
        if (line.compare(0, 6, "VmRSS:") == 0) {
            p = line.c_str() + 6;
            dst = &m.current_bytes;
        } else if (line.compare(0, 6, "VmHWM:") == 0) {
            p = line.c_str() + 6;
            dst = &m.peak_bytes;
        }
        if (p && dst) {
            *dst = static_cast<std::size_t>(std::strtoull(p, nullptr, 10)) * 1024u;
        }
    }
#endif
    return m;
}

std::vector<DeviceMemory> device_memory() {
    std::vector<DeviceMemory> out;
#if CORESG_WITH_CUDA
    const int n = cuda::device_count();
    for (int i = 0; i < n; ++i) {
        std::size_t freeb = 0, totalb = 0;
        if (!cuda::device_memory(i, &freeb, &totalb)) continue;
        DeviceMemory d;
        d.id = i;
        d.total_bytes = totalb;
        d.used_bytes = (totalb >= freeb) ? (totalb - freeb) : 0;
        out.push_back(d);
    }
#endif
    return out;
}

std::size_t instance_device_bytes(std::size_t n, std::uint32_t stride, std::uint32_t k,
                                  std::uint32_t sample) {
    // Mirrors the allocations in nndescent_gpu.
    const std::size_t vectors = n * stride * sizeof(float);
    const std::size_t ids = n * k * sizeof(std::uint32_t);
    const std::size_t dists = n * k * sizeof(float);
    const std::size_t flags = n * k * sizeof(std::uint8_t);
    const std::size_t counts = n * sizeof(std::uint32_t);
    const std::size_t locks = n * sizeof(int);
    const std::size_t reverse = 2u * n * sample * sizeof(std::uint32_t);
    const std::size_t reverse_counts = 2u * n * sizeof(std::uint32_t);
    return vectors + ids + dists + flags + counts + locks + reverse + reverse_counts;
}

double device_bytes_per_point(std::uint32_t stride, std::uint32_t k, std::uint32_t sample) {
    return static_cast<double>(instance_device_bytes(1, stride, k, sample));
}

std::string format_bytes(std::size_t bytes) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(bytes);
    int u = 0;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        ++u;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", v, units[u]);
    return std::string(buf);
}

}  // namespace coresg
