#include <algorithm>
#include <stdexcept>
#include <string>

#include "coresg/backend.hpp"

#if CORESG_WITH_CUDA
#include "coresg/cuda_api.hpp"

namespace coresg {
namespace {

/// Metrics the device kernels cover. L1 and raw inner product fall back to the
/// CPU path rather than silently returning something else.
bool metric_to_code(MetricKind m, int* code) {
    switch (m) {
        case MetricKind::L2: *code = cuda::kMetricL2; return true;
        case MetricKind::Cosine: *code = cuda::kMetricCosine; return true;
        default: return false;
    }
}

class CudaBackend : public Backend {
public:
    explicit CudaBackend(const std::vector<int>& device_ids) : ids_(device_ids) {
        for (int id : ids_) {
            cuda::DeviceProps p;
            if (!cuda::get_device_props(id, &p)) {
                throw std::runtime_error(std::string("cuda device ") + std::to_string(id) +
                                         " unavailable: " + cuda::last_error());
            }
            if (p.major < 7) {
                // The per-node spin locks in the local join rely on
                // independent thread scheduling (Volta and newer).
                throw std::runtime_error(std::string("cuda device ") + std::to_string(id) +
                                         " is sm_" + std::to_string(p.major) +
                                         std::to_string(p.minor) + "; sm_70+ is required");
            }
            DeviceInfo d;
            d.kind = DeviceKind::Cuda;
            d.id = p.id;
            d.name = p.name;
            d.total_memory = p.total_mem;
            d.free_memory = p.free_mem;
            d.multiprocessors = p.sm_count;
            d.cc_major = p.major;
            d.cc_minor = p.minor;
            devices_.push_back(d);
        }
        if (ids_.size() > 1) cuda::enable_peer_access(ids_);
        // Host-side fallback for metrics and jobs the device path declines.
        fallback_ = make_cpu_backend_slots(0, static_cast<unsigned>(ids_.size()), nullptr);
    }

    std::string name() const override { return "cuda"; }
    const std::vector<DeviceInfo>& devices() const override { return devices_; }
    unsigned parallelism() const override { return static_cast<unsigned>(ids_.size()); }

    void run_knn_job(unsigned slot, KnnJob& job) override {
        int code = 0;
        if (!metric_to_code(job.metric, &code) || job.local_n < 512) {
            // Tiny instances are dominated by transfer latency; the host is
            // faster and the result is identical in kind.
            fallback_->run_knn_job(slot, job);
            return;
        }

        cuda::NndConfig cfg;
        cfg.n = static_cast<std::uint32_t>(job.local_n);
        cfg.dim = job.dim;
        cfg.stride = job.stride;
        cfg.k = job.graph.k();
        cfg.sample = job.params.sample;
        cfg.iters = job.params.iters;
        cfg.seed = job.params.seed + job.job_id * 0x9E3779B97F4A7C15ull;
        cfg.metric = code;
        cfg.delta = static_cast<float>(job.params.delta);
        cfg.random_fill = true;

        job.graph.grow(static_cast<idx_t>(job.local_n));
        std::uint64_t updates = 0;
        const int device = device_for(slot);
        const bool ok = cuda::nndescent_gpu(
            device, cfg, job.vectors.data(), job.graph.ids_storage().data(),
            job.graph.dist_storage().data(), job.graph.flag_storage().data(),
            job.graph.count_storage().data(), &updates);
        if (!ok) {
            const std::string err = cuda::last_error();
            // Running out of device memory is a *result*, not a failure: it is
            // exactly what the capacity experiment sets out to demonstrate.
            // Record it and fall back so the run produces a data point instead
            // of aborting an eight-hour sweep.
            if (err.find("out of memory") != std::string::npos ||
                err.find("cudaMalloc") != std::string::npos) {
                job.out_of_memory = true;
                job.error = err;
                fallback_->run_knn_job(slot, job);
                return;
            }
            throw std::runtime_error(std::string("cuda nndescent failed on device ") +
                                     std::to_string(device) + ": " + err);
        }
        job.ran_on = DeviceKind::Cuda;
        job.ran_on_device = device;
        job.stats.rounds = job.params.iters;
        job.stats.updates = static_cast<std::size_t>(updates);
    }

    void mrd_block(unsigned slot, const float* pts, std::size_t m, std::uint32_t dim,
                   std::uint32_t stride, MetricKind metric, const float* core, std::size_t r0,
                   std::size_t r1, float* out) override {
        int code = 0;
        if (!metric_to_code(metric, &code)) {
            fallback_->mrd_block(slot, pts, m, dim, stride, metric, core, r0, r1, out);
            return;
        }
        if (!cuda::mrd_block_gpu(device_for(slot), pts, m, dim, stride, code, core, r0, r1,
                                 out)) {
            throw std::runtime_error(std::string("cuda mrd_block failed: ") + cuda::last_error());
        }
    }

    void trim() override { cuda::release_thread_cache(); }

private:
    int device_for(unsigned slot) const { return ids_[slot < ids_.size() ? slot : 0]; }

    std::vector<int> ids_;
    std::vector<DeviceInfo> devices_;
    std::unique_ptr<Backend> fallback_;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend(const std::vector<int>& device_ids) {
    return std::unique_ptr<Backend>(new CudaBackend(device_ids));
}

}  // namespace coresg
#endif  // CORESG_WITH_CUDA
