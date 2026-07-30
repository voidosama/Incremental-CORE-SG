#include "coresg/backend.hpp"

#include <sstream>

#if CORESG_WITH_CUDA
#include "coresg/cuda_api.hpp"
#endif

namespace coresg {

std::string describe(const DeviceInfo& d) {
    std::ostringstream oss;
    if (d.kind == DeviceKind::Cpu) {
        oss << "cpu (" << d.multiprocessors << " threads)";
    } else {
        oss << "cuda:" << d.id << " " << d.name << " sm_" << d.cc_major << d.cc_minor << " "
            << (d.total_memory >> 20) << " MiB";
    }
    return oss.str();
}

bool cuda_available() {
#if CORESG_WITH_CUDA
    return cuda::device_count() > 0;
#else
    return false;
#endif
}

std::vector<DeviceInfo> enumerate_cuda_devices() {
    std::vector<DeviceInfo> out;
#if CORESG_WITH_CUDA
    const int n = cuda::device_count();
    for (int i = 0; i < n; ++i) {
        cuda::DeviceProps p;
        if (!cuda::get_device_props(i, &p)) continue;
        DeviceInfo d;
        d.kind = DeviceKind::Cuda;
        d.id = p.id;
        d.name = p.name;
        d.total_memory = p.total_mem;
        d.free_memory = p.free_mem;
        d.multiprocessors = p.sm_count;
        d.cc_major = p.major;
        d.cc_minor = p.minor;
        out.push_back(d);
    }
#endif
    return out;
}

std::unique_ptr<Backend> make_backend(const BackendOptions& opts, ThreadPool* cpu_pool) {
#if CORESG_WITH_CUDA
    if (opts.prefer_gpu && cuda::device_count() > 0) {
        std::vector<int> ids = opts.device_ids;
        if (ids.empty()) {
            for (int i = 0; i < cuda::device_count(); ++i) ids.push_back(i);
        }
        if (!ids.empty()) return make_cuda_backend(ids);
    }
#else
    (void)opts;
#endif
    return make_cpu_backend_slots(opts.cpu_threads, opts.cpu_slots, cpu_pool);
}

}  // namespace coresg
