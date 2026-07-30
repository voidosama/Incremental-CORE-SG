// Device enumeration, error reporting and peer access.

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "coresg/cuda_api.hpp"
#include "cuda_common.cuh"

namespace coresg {
namespace cuda {
namespace {

// Thread local: several devices are driven from several host threads at once,
// and each one needs its own error slot.
thread_local std::string g_last_error;

}  // namespace

void set_last_error(const std::string& msg) { g_last_error = msg; }

const char* last_error() { return g_last_error.c_str(); }

int device_count() {
    static int cached = -2;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (cached == -2) {
        int n = 0;
        const cudaError_t err = cudaGetDeviceCount(&n);
        if (err != cudaSuccess) {
            // No driver, no devices, or a driver/runtime mismatch: treat as
            // "CPU only" rather than failing the whole process.
            set_last_error(std::string("cudaGetDeviceCount -> ") + cudaGetErrorString(err));
            cudaGetLastError();
            n = 0;
        }
        cached = n;
    }
    return cached;
}

bool get_device_props(int id, DeviceProps* out) {
    cudaDeviceProp prop;
    CORESG_CUDA_TRY(cudaGetDeviceProperties(&prop, id));
    out->id = id;
    std::snprintf(out->name, sizeof(out->name), "%s", prop.name);
    out->total_mem = prop.totalGlobalMem;
    out->sm_count = prop.multiProcessorCount;
    out->major = prop.major;
    out->minor = prop.minor;

    int current = 0;
    if (cudaGetDevice(&current) == cudaSuccess) {
        std::size_t freem = 0, totalm = 0;
        if (cudaSetDevice(id) == cudaSuccess && cudaMemGetInfo(&freem, &totalm) == cudaSuccess) {
            out->free_mem = freem;
        }
        cudaSetDevice(current);
    }
    return true;
}

bool device_memory(int device, std::size_t* free_bytes, std::size_t* total_bytes) {
    int current = 0;
    const bool have_current = cudaGetDevice(&current) == cudaSuccess;
    CORESG_CUDA_TRY(cudaSetDevice(device));
    std::size_t f = 0, t = 0;
    const cudaError_t err = cudaMemGetInfo(&f, &t);
    if (have_current) cudaSetDevice(current);
    if (err != cudaSuccess) {
        set_last_error(std::string("cudaMemGetInfo -> ") + cudaGetErrorString(err));
        return false;
    }
    if (free_bytes) *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    return true;
}

bool enable_peer_access(const std::vector<int>& devices) {
    bool all = true;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (cudaSetDevice(devices[i]) != cudaSuccess) {
            all = false;
            continue;
        }
        for (std::size_t j = 0; j < devices.size(); ++j) {
            if (i == j) continue;
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, devices[i], devices[j]) != cudaSuccess || !can) {
                all = false;
                continue;
            }
            const cudaError_t err = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) all = false;
            cudaGetLastError();
        }
    }
    return all;
}

}  // namespace cuda
}  // namespace coresg
