#include <algorithm>
#include <sstream>
#include <thread>

#include "coresg/backend.hpp"

namespace coresg {
namespace {

class CpuBackend : public Backend {
public:
    CpuBackend(unsigned threads, unsigned slots, ThreadPool* shared) {
        if (threads == 0) {
            threads = std::thread::hardware_concurrency();
            if (threads == 0) threads = 1;
        }
        if (slots == 0) slots = 1;
        slots = std::min(slots, threads);

        if (shared && slots == 1) {
            // Reuse the caller's pool: avoids spawning a second set of threads
            // for what is by far the common case (one instance at a time,
            // using the whole machine).
            pools_.push_back(nullptr);
            shared_ = shared;
        } else {
            const unsigned per_slot = std::max(1u, threads / slots);
            for (unsigned s = 0; s < slots; ++s) {
                owned_.emplace_back(new ThreadPool(per_slot));
                pools_.push_back(owned_.back().get());
            }
        }

        DeviceInfo info;
        info.kind = DeviceKind::Cpu;
        info.id = 0;
        info.name = "cpu";
        info.multiprocessors = static_cast<int>(threads);
        devices_.push_back(info);
        threads_ = threads;
    }

    std::string name() const override { return "cpu"; }
    const std::vector<DeviceInfo>& devices() const override { return devices_; }
    unsigned parallelism() const override { return static_cast<unsigned>(pools_.size()); }

    void run_knn_job(unsigned slot, KnnJob& job) override {
        ThreadPool* p = pool_for(slot);
        job.ran_on = DeviceKind::Cpu;
        job.ran_on_device = -1;
        job.graph.ensure_locks();
        // Seed before descending. An instance arrives warm started from
        // whatever edges its sampled base points already had -- and the new
        // points have none at all -- so the empty slots must be filled with
        // random candidates or the local joins have nothing to descend from.
        // (The CUDA path does the same via NndConfig::random_fill.)
        init_random(job.view(), job.metric, &job.graph, job.params.seed, p);
        job.stats = nndescent(job.view(), job.metric, &job.graph, job.params, p);
    }

    void mrd_block(unsigned slot, const float* pts, std::size_t m, std::uint32_t dim,
                   std::uint32_t stride, MetricKind metric, const float* core, std::size_t r0,
                   std::size_t r1, float* out) override {
        ThreadPool* p = pool_for(slot);
        dispatch_metric(metric, [&](auto op) {
            using Op = decltype(op);
            auto body = [&](std::size_t lo, std::size_t hi, int) {
                for (std::size_t i = lo; i < hi; ++i) {
                    const float* a = pts + (r0 + i) * stride;
                    const float ci = core[r0 + i];
                    float* row = out + i * m;
                    for (std::size_t j = 0; j < m; ++j) {
                        if (r0 + i == j) {
                            row[j] = 0.0f;
                            continue;
                        }
                        const float d = Op::finalize(Op::rank(a, pts + j * stride, stride));
                        row[j] = std::max(d, std::max(ci, core[j]));
                    }
                }
            };
            const std::size_t rows = r1 - r0;
            if (p && p->size() > 1) {
                p->parallel_ranges(rows, body);
            } else {
                body(0, rows, 0);
            }
            return 0;
        });
    }

private:
    ThreadPool* pool_for(unsigned slot) {
        if (shared_) return shared_;
        return pools_[slot < pools_.size() ? slot : 0];
    }

    std::vector<std::unique_ptr<ThreadPool>> owned_;
    std::vector<ThreadPool*> pools_;
    ThreadPool* shared_ = nullptr;
    std::vector<DeviceInfo> devices_;
    unsigned threads_ = 1;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(unsigned threads, ThreadPool* pool) {
    return std::unique_ptr<Backend>(new CpuBackend(threads, 1, pool));
}

std::unique_ptr<Backend> make_cpu_backend_slots(unsigned threads, unsigned slots,
                                                ThreadPool* pool) {
    return std::unique_ptr<Backend>(new CpuBackend(threads, slots, pool));
}

}  // namespace coresg
