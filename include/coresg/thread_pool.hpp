#pragma once
// A small work-stealing-free thread pool.
//
// Two usage patterns are needed by the pipeline and both are supported here:
//
//   * parallel_for / parallel_ranges  -- data parallelism inside one kernel
//     (NN-Descent rounds, degree counting, edge materialisation).
//   * submit / wait_all               -- task parallelism over heterogeneous
//     jobs (one NN-Descent *instance* per queued task, which is how work is
//     spread over several GPUs).
//
// std::thread is used rather than OpenMP because MSVC only ships OpenMP 2.0,
// which cannot express the atomics and unsigned induction variables the
// NN-Descent kernels need.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace coresg {

class ThreadPool {
public:
    explicit ThreadPool(unsigned num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;
        }
        workers_.reserve(num_threads);
        for (unsigned t = 0; t < num_threads; ++t) {
            workers_.emplace_back([this, t] { worker_loop(static_cast<int>(t)); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    unsigned size() const { return static_cast<unsigned>(workers_.size()); }

    /// Enqueue a task. `fn` receives the worker id, which callers use to index
    /// per-thread scratch buffers (and, in the CUDA backend, to pick a device).
    void submit(std::function<void(int)> fn) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            tasks_.push(std::move(fn));
            ++pending_;
        }
        cv_.notify_one();
    }

    /// Block until every submitted task has finished. Safe to call from the
    /// thread that submitted the work only (the pool has no nested-wait
    /// support, which the pipeline does not need).
    void wait_all() {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
        if (error_) {
            std::exception_ptr e = error_;
            error_ = nullptr;
            std::rethrow_exception(e);
        }
    }

    /// Split [0, n) into at most `size()` contiguous ranges and run
    /// `fn(lo, hi, slot)`. Contiguity matters: the NN-Descent kernels touch
    /// nodes in id order to keep the vector reads cache friendly.
    ///
    /// `slot` is the *range* index, not the worker id, and is unique per call
    /// and always < size(). That distinction is load bearing: tasks are pulled
    /// off a shared queue, so one worker can execute several ranges, and a
    /// worker id would repeat. Indexing a per-range output array by a repeated
    /// id silently drops results. Because slots are unique and no two ranges
    /// run concurrently on one slot, `slot` is safe for both output arrays and
    /// per-task scratch buffers.
    template <typename F>
    void parallel_ranges(std::size_t n, F&& fn) {
        if (n == 0) return;
        const std::size_t nt = size();
        if (nt <= 1 || n == 1) {
            fn(std::size_t(0), n, 0);
            return;
        }
        const std::size_t chunk = (n + nt - 1) / nt;
        for (std::size_t t = 0; t < nt; ++t) {
            const std::size_t lo = t * chunk;
            if (lo >= n) break;
            const std::size_t hi = (lo + chunk < n) ? lo + chunk : n;
            const int slot = static_cast<int>(t);
            submit([&fn, lo, hi, slot](int) { fn(lo, hi, slot); });
        }
        wait_all();
    }

    /// Dynamically scheduled element-wise loop, `fn(i, tid)`. Preferred when
    /// the per-item cost is uneven (local joins on nodes with very different
    /// degrees).
    ///
    /// Here `tid` really is the worker id and repeats across chunks, so
    /// per-thread state indexed by it must only ever be *accumulated* into,
    /// never assigned. Concurrent chunks always carry distinct tids, so no
    /// locking is needed.
    template <typename F>
    void parallel_for(std::size_t n, F&& fn, std::size_t grain = 256) {
        if (n == 0) return;
        if (size() <= 1) {
            for (std::size_t i = 0; i < n; ++i) fn(i, 0);
            return;
        }
        std::atomic<std::size_t> cursor{0};
        const std::size_t nt = size();
        for (std::size_t t = 0; t < nt; ++t) {
            submit([&, grain, n](int tid) {
                for (;;) {
                    const std::size_t lo = cursor.fetch_add(grain, std::memory_order_relaxed);
                    if (lo >= n) return;
                    const std::size_t hi = (lo + grain < n) ? lo + grain : n;
                    for (std::size_t i = lo; i < hi; ++i) fn(i, tid);
                }
            });
        }
        wait_all();
    }

private:
    void worker_loop(int tid) {
        for (;;) {
            std::function<void(int)> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            try {
                task(tid);
            } catch (...) {
                std::lock_guard<std::mutex> lk(mu_);
                if (!error_) error_ = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--pending_ == 0) done_cv_.notify_all();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void(int)>> tasks_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::size_t pending_ = 0;
    bool stop_ = false;
    std::exception_ptr error_ = nullptr;
};

/// Owns a pool only when the caller did not supply one. Every public entry
/// point takes an optional `ThreadPool*` so a single pool can be reused across
/// the whole pipeline instead of respawning threads per phase.
class PoolRef {
public:
    PoolRef(ThreadPool* external, unsigned threads_if_owned) {
        if (external) {
            pool_ = external;
        } else {
            owned_.reset(new ThreadPool(threads_if_owned));
            pool_ = owned_.get();
        }
    }
    ThreadPool* operator->() const { return pool_; }
    ThreadPool& operator*() const { return *pool_; }
    ThreadPool* get() const { return pool_; }

private:
    std::unique_ptr<ThreadPool> owned_;
    ThreadPool* pool_ = nullptr;
};

}  // namespace coresg
