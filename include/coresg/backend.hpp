#pragma once
// Execution backends.
//
// The unit of work is a *self-contained* k-NN sub-problem: a contiguous block
// of vectors plus a warm-started local graph, with no reference to any global
// state. That is the whole reason the incremental scheme in this project
// scales to several GPUs -- an instance never has to see the full data set,
// so instances can be queued onto whatever device is free, and a device only
// ever holds `|sample| + |new data|` vectors.
//
// A backend exposes `parallelism()` execution slots (one per GPU, or one per
// CPU worker group). `JobScheduler` owns exactly that many threads and pins
// slot == thread id, so a CUDA backend can bind a device to a thread once and
// keep its streams and allocations alive across jobs.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "coresg/distance.hpp"
#include "coresg/knn_graph.hpp"
#include "coresg/nndescent.hpp"
#include "coresg/thread_pool.hpp"
#include "coresg/types.hpp"

namespace coresg {

enum class DeviceKind : std::uint8_t { Cpu = 0, Cuda = 1 };

struct DeviceInfo {
    DeviceKind kind = DeviceKind::Cpu;
    int id = 0;
    std::string name;
    std::size_t total_memory = 0;
    std::size_t free_memory = 0;
    int multiprocessors = 0;
    int cc_major = 0;
    int cc_minor = 0;
};

std::string describe(const DeviceInfo& d);

/// One NN-Descent instance, ready to be shipped to a device.
struct KnnJob {
    std::vector<float> vectors;  ///< local_n * stride, row major, zero padded
    std::uint32_t dim = 0;
    std::uint32_t stride = 0;
    std::size_t local_n = 0;
    MetricKind metric = MetricKind::L2;
    NNDescentParams params;
    KnnGraph graph;  ///< warm-started on input, result on output
    NNDescentStats stats;
    int job_id = 0;

    /// Where the job actually ran, filled in by the backend. Reported because
    /// the CUDA backend silently hands small instances to the host, and a
    /// scaling row labelled "4 GPUs" that quietly ran on the CPU is worse than
    /// no measurement at all.
    DeviceKind ran_on = DeviceKind::Cpu;
    int ran_on_device = -1;
    /// Set when the device declined the job for lack of memory. The capacity
    /// experiment depends on this being a recorded outcome rather than a crash.
    bool out_of_memory = false;
    std::string error;

    DataView view() const { return DataView{vectors.data(), dim, stride, local_n}; }
};

/// Backends must be usable concurrently from `parallelism()` distinct threads,
/// as long as each thread uses its own slot.
class Backend {
public:
    virtual ~Backend() = default;

    virtual std::string name() const = 0;
    virtual const std::vector<DeviceInfo>& devices() const = 0;
    virtual unsigned parallelism() const = 0;

    /// Build/refine `job.graph` in place.
    virtual void run_knn_job(unsigned slot, KnnJob& job) = 0;

    /// Rows [r0, r1) of the m x m mutual-reachability matrix of `pts`.
    /// `out` receives (r1 - r0) * m floats. Used by the anti-hub MST and by
    /// the exact O(n^2) reference MST.
    virtual void mrd_block(unsigned slot, const float* pts, std::size_t m, std::uint32_t dim,
                           std::uint32_t stride, MetricKind metric, const float* core,
                           std::size_t r0, std::size_t r1, float* out) = 0;

    /// Optional hint that the backend should release cached device memory.
    virtual void trim() {}
};

struct BackendOptions {
    bool prefer_gpu = true;
    /// Empty means "every visible device". Otherwise CUDA device ordinals.
    std::vector<int> device_ids;
    /// CPU worker count for the CPU backend (0 = hardware concurrency).
    unsigned cpu_threads = 0;
    /// How many CPU jobs run concurrently. 1 (the default) gives every job the
    /// whole machine, which is what you want when there is a single instance;
    /// raise it to trade intra-job parallelism for job throughput.
    unsigned cpu_slots = 1;
};

/// Returns the CUDA backend when it is compiled in, at least one device is
/// visible, and `prefer_gpu` is set; otherwise the CPU backend.
std::unique_ptr<Backend> make_backend(const BackendOptions& opts, ThreadPool* cpu_pool = nullptr);

std::unique_ptr<Backend> make_cpu_backend(unsigned threads, ThreadPool* pool = nullptr);

/// CPU backend running `slots` jobs concurrently, `threads / slots` each.
std::unique_ptr<Backend> make_cpu_backend_slots(unsigned threads, unsigned slots,
                                                ThreadPool* pool = nullptr);

#if CORESG_WITH_CUDA
std::unique_ptr<Backend> make_cuda_backend(const std::vector<int>& device_ids);
#endif

bool cuda_available();
std::vector<DeviceInfo> enumerate_cuda_devices();

/// Runs jobs across a backend's slots. Slot == worker thread id, so a device
/// stays bound to one thread for the lifetime of the scheduler.
class JobScheduler {
public:
    explicit JobScheduler(Backend* backend)
        : backend_(backend), pool_(backend->parallelism() ? backend->parallelism() : 1u) {}

    unsigned slots() const { return pool_.size(); }

    /// Run every job, `slots()` at a time. Returns when all are done.
    void run_all(std::vector<KnnJob>* jobs) {
        for (std::size_t i = 0; i < jobs->size(); ++i) {
            KnnJob* job = &(*jobs)[i];
            pool_.submit([this, job](int tid) {
                backend_->run_knn_job(static_cast<unsigned>(tid), *job);
            });
        }
        pool_.wait_all();
    }

    /// Split [0, m) row-wise across slots and fill the full m x m matrix.
    void mrd_matrix(const float* pts, std::size_t m, std::uint32_t dim, std::uint32_t stride,
                    MetricKind metric, const float* core, float* out) {
        const unsigned ns = slots();
        const std::size_t chunk = (m + ns - 1) / ns;
        for (unsigned s = 0; s < ns; ++s) {
            const std::size_t r0 = std::size_t(s) * chunk;
            if (r0 >= m) break;
            const std::size_t r1 = (r0 + chunk < m) ? r0 + chunk : m;
            pool_.submit([=](int tid) {
                backend_->mrd_block(static_cast<unsigned>(tid), pts, m, dim, stride, metric, core,
                                    r0, r1, out + r0 * m);
            });
        }
        pool_.wait_all();
    }

private:
    Backend* backend_;
    ThreadPool pool_;
};

}  // namespace coresg
