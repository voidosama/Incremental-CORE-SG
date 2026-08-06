#pragma once
// The k-NN graph under construction / maintenance.
//
// Layout: three flat arrays of n*k plus a per-node fill count. Each
// neighbourhood is kept sorted by ascending distance, which buys three things
// at once:
//
//   * core distances are an O(1) lookup (`core_dist`), and every mpts <= k_max
//     is served by the same array -- exactly the reuse CORE-SG relies on;
//   * the "is this candidate worth inserting" test is a single comparison
//     against the last slot, so the vast majority of local-join results are
//     rejected without taking a lock;
//   * merging the neighbourhoods produced by several NN-Descent instances is
//     just repeated insertion, so "keep the top-k" needs no extra structure.
//
// `try_insert` reports the evicted neighbour. That is what lets the hubness
// index maintain reverse degrees exactly and incrementally, instead of
// rescanning the graph after every batch.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "coresg/types.hpp"

namespace coresg {

class ThreadPool;

struct InsertResult {
    bool inserted = false;
    idx_t evicted = kInvalidIdx;  // neighbour pushed out of the list, if any
};

/// Notified on every directed edge change. Implemented by HubnessIndex.
class GraphObserver {
public:
    virtual ~GraphObserver() = default;
    /// `src` now has `dst` in its neighbour list.
    virtual void on_edge_added(idx_t src, idx_t dst) = 0;
    /// `dst` was dropped from `src`'s neighbour list.
    virtual void on_edge_removed(idx_t src, idx_t dst) = 0;
};

/// Read-only view of the reverse adjacency, when something is maintaining it.
/// Implemented by HubnessIndex, which already keeps reverse lists exact.
///
/// It exists so a graph *search* can walk the undirected graph. A directed
/// k-NN graph has sinks -- a node that appears in nobody's list cannot be left
/// by following forward edges -- and a greedy walk that reaches one is stuck.
/// Supplying the back edges costs nothing here because they are already there.
///
/// Callers must treat the returned reference as valid only while the graph is
/// quiescent: the lists are mutated by the observer hooks on every insertion.
class ReverseNeighbours {
public:
    virtual ~ReverseNeighbours() = default;
    /// False when no reverse table is being kept; `reverse_of` is then empty.
    virtual bool reverse_available() const = 0;
    virtual const std::vector<idx_t>& reverse_of(idx_t v) const = 0;
};

/// One spinlock per node. Kept outside the arrays so that growing the graph
/// (new points arriving) does not have to preserve lock state.
class SpinLocks {
public:
    SpinLocks() = default;
    explicit SpinLocks(std::size_t n) { resize(n); }

    void resize(std::size_t n) {
        std::unique_ptr<std::atomic<std::uint32_t>[]> next(new std::atomic<std::uint32_t>[n]);
        for (std::size_t i = 0; i < n; ++i) next[i].store(0, std::memory_order_relaxed);
        locks_ = std::move(next);
        n_ = n;
    }

    void lock(std::size_t i) {
        std::atomic<std::uint32_t>& l = locks_[i];
        std::uint32_t expected = 0;
        while (!l.compare_exchange_weak(expected, 1u, std::memory_order_acquire,
                                        std::memory_order_relaxed)) {
            expected = 0;
            std::this_thread::yield();
        }
    }

    void unlock(std::size_t i) { locks_[i].store(0, std::memory_order_release); }

    std::size_t size() const { return n_; }

private:
    std::unique_ptr<std::atomic<std::uint32_t>[]> locks_;
    std::size_t n_ = 0;
};

class KnnGraph {
public:
    KnnGraph() = default;
    KnnGraph(idx_t n, std::uint32_t k) { reset(n, k); }

    void reset(idx_t n, std::uint32_t k);

    /// Grow to `new_n` nodes. Existing neighbourhoods are untouched; the added
    /// nodes start empty. Not thread safe -- call it between batches.
    void grow(idx_t new_n);

    idx_t size() const { return n_; }
    std::uint32_t k() const { return k_; }

    const idx_t* ids(idx_t node) const { return ids_.data() + std::size_t(node) * k_; }
    const dist_t* dists(idx_t node) const { return dist_.data() + std::size_t(node) * k_; }
    const std::uint8_t* flags(idx_t node) const { return flags_.data() + std::size_t(node) * k_; }
    std::uint32_t degree(idx_t node) const { return count_[node]; }

    idx_t* mutable_ids(idx_t node) { return ids_.data() + std::size_t(node) * k_; }
    dist_t* mutable_dists(idx_t node) { return dist_.data() + std::size_t(node) * k_; }
    std::uint8_t* mutable_flags(idx_t node) { return flags_.data() + std::size_t(node) * k_; }

    /// Raw handles, for bulk transfers to/from a device.
    std::vector<idx_t>& ids_storage() { return ids_; }
    std::vector<dist_t>& dist_storage() { return dist_; }
    std::vector<std::uint8_t>& flag_storage() { return flags_; }
    std::vector<std::uint32_t>& count_storage() { return count_; }
    const std::vector<idx_t>& ids_storage() const { return ids_; }
    const std::vector<dist_t>& dist_storage() const { return dist_; }
    const std::vector<std::uint8_t>& flag_storage() const { return flags_; }
    const std::vector<std::uint32_t>& count_storage() const { return count_; }

    void set_observer(GraphObserver* obs) { observer_ = obs; }
    GraphObserver* observer() const { return observer_; }

    /// Single-threaded insertion. `cand == node` and duplicates are rejected.
    InsertResult try_insert(idx_t node, idx_t cand, dist_t d, bool is_new);

    /// Same, holding the node's spinlock. `ensure_locks()` must have been
    /// called (grow()/reset() do it for you).
    InsertResult try_insert_mt(idx_t node, idx_t cand, dist_t d, bool is_new);

    /// Cheap rejection test used before taking a lock. Unfilled slots hold
    /// +inf, so this single comparison also covers "the list is not full yet"
    /// and never has to read the concurrently mutated degree counter.
    bool would_reject(idx_t node, dist_t d) const {
        return !(d < dist_[std::size_t(node) * k_ + (k_ - 1)]);
    }

    void clear_new_flags();
    void set_all_new_flags();

    /// Core distance w.r.t. `mpts`, i.e. the distance to the mpts-th nearest
    /// neighbour (self excluded), following the CORE-SG paper's definition.
    /// `mpts` is 1-based. If the neighbourhood is shorter than `mpts` (only
    /// possible for freshly inserted points before refinement) the largest
    /// known distance is returned, and `+inf` for an empty neighbourhood.
    dist_t core_dist(idx_t node, std::uint32_t mpts) const {
        const std::uint32_t c = count_[node];
        if (c == 0) return kInfDist;
        std::uint32_t pos = (mpts == 0) ? 0 : mpts - 1;
        if (pos >= c) pos = c - 1;
        return dist_[std::size_t(node) * k_ + pos];
    }

    /// Vector of core distances for a fixed mpts, computed once and reused by
    /// every mutual-reachability evaluation.
    std::vector<dist_t> core_distances(std::uint32_t mpts, ThreadPool* pool = nullptr) const;

    /// Number of nodes whose neighbourhood is shorter than k.
    idx_t incomplete_nodes() const;

    /// Undirected edge list of the k-NN graph, deduplicated (mutual pairs
    /// contribute one edge, as the CORE-SG paper prescribes). Weights are the
    /// base distances; use `mutual_reachability_edges` for CORE-SG weights.
    std::vector<Edge> undirected_edges(ThreadPool* pool = nullptr) const;

    std::size_t undirected_edge_count(ThreadPool* pool = nullptr) const;

    /// Same edges, reweighted with mrd_mpts(p,q) = max(cd(p), cd(q), d(p,q)).
    std::vector<Edge> mutual_reachability_edges(const std::vector<dist_t>& core,
                                                ThreadPool* pool = nullptr) const;

    void ensure_locks();
    SpinLocks& locks() { return locks_; }

    /// Sanity check used by the tests: sorted, no self loops, no duplicates.
    bool validate(std::string* error = nullptr) const;

private:
    InsertResult insert_unlocked(idx_t node, idx_t cand, dist_t d, bool is_new);

    idx_t n_ = 0;
    std::uint32_t k_ = 0;
    std::vector<idx_t> ids_;
    std::vector<dist_t> dist_;
    std::vector<std::uint8_t> flags_;
    std::vector<std::uint32_t> count_;
    SpinLocks locks_;
    GraphObserver* observer_ = nullptr;
};

}  // namespace coresg
