#pragma once
// Dynamic NN-Descent (Wang, Zhao, Xiao, Yao & Zhang, IEEE TBD 2025).
//
// Differences from classic NN-Descent, all three of which are what make this
// the right base for an incremental, multi-device CORE-SG:
//
//   * sampling happens directly on the live graph G, so a neighbour inserted
//     by the current local join participates in the very next one -- there is
//     no static snapshot graph H to allocate or to keep in sync;
//   * the reverse graph R is *incomplete* and bounded to `sample` entries per
//     node and per flavour (old/new); a reverse neighbour that misses the
//     current round simply joins the next one;
//   * consequently there is no barrier inside a round. Threads (or devices)
//     touch disjoint nodes and only synchronise through the per-node spinlock
//     of the k-NN graph.
//
// Extension used by the incremental pipeline: `active` restricts which nodes
// drive a local join. Insertions still land anywhere in the graph, so a
// refinement pass over "the new points and their neighbours" costs
// O(|active|) rather than O(n) while still improving base-point
// neighbourhoods that the new points turn out to belong to.

#include <cstdint>
#include <vector>

#include "coresg/dataset.hpp"
#include "coresg/distance.hpp"
#include "coresg/knn_graph.hpp"
#include "coresg/types.hpp"

namespace coresg {

class ThreadPool;

enum class SamplingPolicy : std::uint8_t {
    /// Take the closest new/old neighbours (a prefix of the sorted list).
    /// Deterministic and cache-friendly; matches "only the close neighbors
    /// are sampled" from the paper.
    Closest = 0,
    /// Bernoulli sampling over the whole list, closer to the reference
    /// kgraph implementation. Kept for ablation studies.
    Random = 1
};

struct NNDescentParams {
    std::uint32_t k = 32;        ///< neighbourhood size (K)
    std::uint32_t sample = 16;   ///< smpN; |O| is capped at 3*smpN, |N| at smpN
    std::uint32_t iters = 10;    ///< iterN, i.e. sweeps over the active set
    double delta = 0.001;        ///< stop when updates < delta * k * |active|
    std::uint64_t seed = 42;
    SamplingPolicy policy = SamplingPolicy::Closest;
    bool verbose = false;

    /// Cap the local join's candidate scan to the first `join_prefix` entries
    /// of a neighbour list. 0 scans the whole list.
    ///
    /// A local join is quadratic in the number of candidates it draws, so
    /// halving the scan quarters the pairs it evaluates. Zhao et al. report
    /// graph quality holding for kept fractions between k/5 and k/2 when the
    /// rear of each list is withheld from the merge and restored afterwards.
    std::uint32_t join_prefix = 0;

    /// Per-node block label. When non-null, a pair whose two endpoints carry
    /// the same label is skipped.
    ///
    /// After a partitioned build every within-block pair has already been
    /// compared by that block's own descent, so re-evaluating it in the repair
    /// spends a distance computation on a known answer. Not owned; must
    /// outlive the call.
    const idx_t* block_of = nullptr;
};

struct NNDescentStats {
    std::uint32_t rounds = 0;
    std::size_t updates = 0;
    std::size_t distance_evals = 0;
    double seconds = 0.0;
};

/// Bounded, incomplete reverse-neighbour cache (graph R in the paper).
/// Deliberately race-tolerant: a lost or stale entry only changes which pairs
/// get compared in a given round, never the validity of the graph.
class ReverseCache {
public:
    ReverseCache() = default;
    ReverseCache(idx_t n, std::uint32_t sample) { reset(n, sample); }

    void reset(idx_t n, std::uint32_t sample);
    void grow(idx_t new_n);
    void clear();

    std::uint32_t sample() const { return sample_; }
    idx_t size() const { return n_; }

    void push_old(idx_t node, idx_t src);
    void push_new(idx_t node, idx_t src);

    /// Move the stored ids into `out` and empty the slot.
    void drain_old(idx_t node, std::vector<idx_t>* out);
    void drain_new(idx_t node, std::vector<idx_t>* out);

    std::size_t memory_bytes() const;

private:
    idx_t n_ = 0;
    std::uint32_t sample_ = 0;
    std::vector<idx_t> old_;
    std::vector<idx_t> new_;
    std::unique_ptr<std::atomic<std::uint32_t>[]> old_cnt_;
    std::unique_ptr<std::atomic<std::uint32_t>[]> new_cnt_;
};

/// Top every neighbourhood up to `k` with random candidates, flagged new.
/// Existing edges are kept, so this both cold-starts an empty graph and seeds
/// a warm-started one.
void init_random(const DataView& data, MetricKind metric, KnnGraph* g, std::uint64_t seed,
                 ThreadPool* pool = nullptr);

/// Fill only the empty slots of the given nodes with random candidates. Used
/// to top up warm-started instances (base points keep the edges inherited from
/// the global graph, new points start from noise).
void init_random_fill(const DataView& data, MetricKind metric, KnnGraph* g,
                      const std::vector<idx_t>& nodes, std::uint64_t seed,
                      ThreadPool* pool = nullptr);

/// Run Dynamic NN-Descent. `active == nullptr` means "every node".
NNDescentStats nndescent(const DataView& data, MetricKind metric, KnnGraph* g,
                         const NNDescentParams& params, ThreadPool* pool = nullptr,
                         const std::vector<idx_t>* active = nullptr,
                         ReverseCache* reverse = nullptr);

/// Exact k-NN by brute force. O(n^2 d); reference for recall measurements and
/// the fallback for instances small enough that a descent is not worth it.
void brute_force_knn(const DataView& data, MetricKind metric, KnnGraph* g,
                     ThreadPool* pool = nullptr);

}  // namespace coresg
