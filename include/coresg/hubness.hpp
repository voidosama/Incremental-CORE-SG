#pragma once
// Local hubness: reverse degrees, the normalised reverse score, and anti-hub
// selection.
//
// Motivation (this is the part of the project that replaces incremental MST
// maintenance). A CORE-SG is k-NNG union MST_kmax. The k-NN graph already
// carries the overwhelming majority of the edges; the MST only matters for the
// few edges whose weight is *not* determined by a core distance, i.e. edges
// that bridge regions the neighbourhoods cannot reach. Those bridges start at
// the border of a cluster, and in more than a couple of dimensions "border" is
// best read as "low local hubness".
//
//     reverse_degree(v)  = |{u : v in NN(u)}|
//     reverse_score(v)   = reverse_degree(v) / mean_{u in NN(v)} reverse_degree(u)
//
// Because the denominator is the local mean, the score is invariant to the
// absolute density of the region: the border point of a dense cluster scores
// as low as the border point of a sparse one, which is exactly what is needed
// when the graph mixes regions of very different densities.
//
//     score == 1 -> neutral,  > 1 -> local hub,  < 1 -> local anti-hub
//
// Degenerate cases (no reverse neighbours anywhere in the neighbourhood) are
// resolved as: rd > 0, sum == 0 -> k;  rd == 0, sum > 0 -> 0;  both 0 -> 1.
//
// Incrementality. The index attaches to the k-NN graph as an observer, so
// every inserted/evicted directed edge updates a reverse degree exactly and in
// O(1). A batch therefore only has to rescore the nodes whose own reverse
// degree moved plus the nodes that *have one of those as a neighbour* -- the
// reverse neighbours of the dirty set. With `store_reverse_lists` the reverse
// neighbourhoods are materialised and the propagation is exact; without them
// the cheaper forward approximation is used (propagate to a dirty node's
// neighbours instead of its reverse neighbours).

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "coresg/knn_graph.hpp"
#include "coresg/types.hpp"

namespace coresg {

class ThreadPool;

/// What "the typical reverse degree around v" means, i.e. the denominator of
/// the reverse score.
///
/// ArithmeticMean is the specified formula. It has one measurable side effect:
/// a mean is dominated by the few high-degree hubs in a neighbourhood, so it
/// sits well above the median degree and most nodes score below 1. The other
/// two are robust alternatives that shift the neutral point back towards the
/// middle of the population without changing the *ordering* much -- which is
/// what matters, since only the lowest d*ln(|E|) scores feed the bridges.
enum class ScoreNormalization : std::uint8_t {
    ArithmeticMean = 0,
    /// exp(mean(ln(rd + 1))) - 1. Defined when some neighbours have degree 0,
    /// and far less sensitive to a single very large hub.
    GeometricMean = 1,
    /// Lower median of the neighbours' reverse degrees. The most robust, but
    /// it collapses to the degenerate rule whenever at least half of the
    /// neighbourhood has reverse degree 0.
    Median = 2,
};

const char* to_string(ScoreNormalization n);
bool score_normalization_from_string(const std::string& s, ScoreNormalization* out);

struct HubnessParams {
    ScoreNormalization normalization = ScoreNormalization::ArithmeticMean;
    /// Number of anti-hubs kept for the missing-edge estimate:
    ///   count = round(multiplier * dim * ln(num_edges))
    double antihub_multiplier = 1.0;
    /// Materialise reverse neighbourhoods (exact score propagation, extra
    /// memory of roughly the k-NN graph itself). Auto-disabled above
    /// `reverse_list_node_limit` nodes.
    bool store_reverse_lists = true;
    std::size_t reverse_list_node_limit = 50u * 1000u * 1000u;
    /// Value of k used by the degenerate-case rule; 0 means "graph's k".
    std::uint32_t k_for_score = 0;
    /// Hard floor/ceiling on the selected set (0 = no ceiling).
    std::size_t min_selected = 2;
    std::size_t max_selected = 0;

    /// Ceiling on the work the bridge MST may do, in distance-coordinate
    /// operations (`m^2 * d / 2` for m selected anti-hubs in d dimensions).
    /// 0 disables it.
    ///
    /// This exists because the d*ln(|E|) budget is linear in the *ambient*
    /// dimensionality while the MST over the selected set is quadratic in the
    /// budget, so the bridge cost grows as d^3. Measured across the evaluation
    /// data sets: 0.1 G ops at d=84, but 128 G at d=960, and at d=1000 with
    /// n=11314 the budget exceeds n outright -- the "approximation" then
    /// selects every point and silently becomes the exact MST it was meant to
    /// replace. Capping on cost rather than on count leaves moderate
    /// dimensionalities completely untouched and only binds where the formula
    /// runs away.
    double max_pair_ops = 2e9;

    /// Guarantee this many anti-hubs from *every* k-NN component, on top of
    /// the global d*ln(|E|) budget. 0 keeps the pure budget rule.
    ///
    /// The budget alone is blind to how many clusters exist -- it depends only
    /// on d and |E| -- so once the number of density-separated regions
    /// approaches it, some regions receive no representative at all and no
    /// bridge can leave them. A per-component quota fixes that directly, and
    /// it also sidesteps the fact that a small uniform component has *no*
    /// node scoring below 1: in a region where everyone has the same reverse
    /// degree everyone looks locally neutral, however detached the region is.
    ///
    /// Defaults to 2 because it is measurably free -- where the budget already
    /// covers every component it adds exactly nothing -- and it takes the
    /// "bridges alone left the graph split" rate from 6/24 configurations to
    /// 0/24. Set to 0 for the pure d*ln(|E|) rule.
    std::size_t min_per_component = 2;
};

struct HubnessStats {
    idx_t nodes = 0;
    idx_t tagged_antihubs = 0;   ///< every node with score < 1
    std::size_t selected = 0;    ///< the d*ln(|E|) lowest-scoring ones
    std::size_t budget_selected = 0;   ///< how many came from the global budget
    std::size_t quota_selected = 0;    ///< added purely to satisfy the quota
    std::size_t budget_uncapped = 0;   ///< what d*ln(|E|) asked for
    bool budget_capped = false;        ///< max_pair_ops (or max_selected) bound it
    std::size_t covered_components = 0;
    std::size_t rescored = 0;    ///< nodes whose score was recomputed
    double min_score = 0.0;
    double max_score = 0.0;
    double mean_score = 0.0;
    double selection_cutoff = 0.0;
    double seconds = 0.0;
};

class HubnessIndex : public GraphObserver {
public:
    HubnessIndex() = default;
    explicit HubnessIndex(const HubnessParams& p) : params_(p) {}

    const HubnessParams& params() const { return params_; }
    void set_params(const HubnessParams& p) { params_ = p; }

    /// Full (re)build from the graph. Also sizes the internal arrays.
    void build(const KnnGraph& g, ThreadPool* pool = nullptr);

    /// Grow to match a graph that gained nodes. Keeps existing degrees.
    void grow(idx_t new_n);

    /// Register/unregister as the graph's observer. While attached, reverse
    /// degrees (and reverse lists) track every edge change exactly.
    void attach(KnnGraph* g);
    void detach(KnnGraph* g);

    // ---- GraphObserver -------------------------------------------------
    void on_edge_added(idx_t src, idx_t dst) override;
    void on_edge_removed(idx_t src, idx_t dst) override;

    /// Component labelling used by the per-component quota. Not owned; the
    /// pointer must stay valid across the following refresh/rescore call.
    /// Passing nullptr (the default) disables the quota.
    void set_components(const std::vector<idx_t>* labels, idx_t num_components) {
        comp_labels_ = labels;
        comp_count_ = num_components;
    }

    /// Rescore the dirty region and reselect the anti-hubs.
    /// `num_edges` is the undirected edge count of the current k-NN graph.
    HubnessStats refresh(const KnnGraph& g, std::uint32_t dim, std::size_t num_edges,
                         ThreadPool* pool = nullptr);

    /// Rescore *every* node (used for the initial build and by the tests to
    /// check that the incremental path agrees with the batch one).
    HubnessStats rescore_all(const KnnGraph& g, std::uint32_t dim, std::size_t num_edges,
                             ThreadPool* pool = nullptr);

    std::uint32_t reverse_degree(idx_t v) const {
        return rdeg_[v].load(std::memory_order_relaxed);
    }
    double reverse_score(idx_t v) const { return score_[v]; }
    bool is_antihub(idx_t v) const { return antihub_[v] != 0; }

    /// The d*ln(|E|) lowest-scoring nodes, ascending by score.
    const std::vector<idx_t>& selected() const { return selected_; }

    /// Materialised reverse neighbourhood (empty when disabled).
    const std::vector<idx_t>& reverse_neighbours(idx_t v) const { return rev_[v]; }
    bool has_reverse_lists() const { return rev_enabled_; }

    std::size_t dirty_count() const { return dirty_list_.size(); }
    std::size_t memory_bytes() const;

    /// Nodes whose own neighbour list changed since the last refresh. These
    /// are the only nodes that can have contributed a *new* edge, which is
    /// what lets the connectivity tracker union over a batch-sized set
    /// instead of the whole graph.
    const std::vector<idx_t>& changed_nodes() const { return nbr_changed_list_; }

    /// Directed edges evicted since the last refresh, as (src, dst). A pair
    /// here is only a *candidate*: the undirected edge survives if dst still
    /// lists src. Filtering has to happen once the graph is quiescent, so it
    /// is left to the consumer. Only recorded when tracking is enabled.
    const std::vector<std::pair<idx_t, idx_t>>& removal_candidates() const {
        return removals_;
    }
    void set_track_removals(bool on) { track_removals_ = on; }
    bool track_removals() const { return track_removals_; }
    void clear_removals() { removals_.clear(); }

    /// Score of a single node, computed from scratch. Exposed for tests.
    double compute_score(const KnnGraph& g, idx_t v) const;

    /// Fraction of nodes with score < 1. Reported because it is the number
    /// that exposes how skewed a given normalisation is.
    double antihub_fraction() const;

private:
    /// `scratch` holds the neighbours' reverse degrees; the median variant
    /// sorts it in place. Passing it in keeps a full rescore allocation-free.
    double compute_score(const KnnGraph& g, idx_t v, std::vector<std::int32_t>* scratch) const;

    void mark_dirty(idx_t v);
    void mark_neighbourhood_changed(idx_t v);
    std::vector<idx_t> collect_rescore_set(const KnnGraph& g);
    void apply_scores(const KnnGraph& g, const std::vector<idx_t>& nodes, ThreadPool* pool);
    HubnessStats select(std::uint32_t dim, std::size_t num_edges, std::size_t rescored,
                        ThreadPool* pool);

    HubnessParams params_;
    idx_t n_ = 0;
    std::uint32_t k_ = 0;
    bool rev_enabled_ = false;

    std::unique_ptr<std::atomic<std::int32_t>[]> rdeg_;
    std::vector<double> score_;
    std::vector<std::uint8_t> antihub_;

    // Dirty tracking. The flag arrays dedupe (one atomic exchange per node per
    // batch); the lists preserve the set so a refresh never scans all n.
    std::unique_ptr<std::atomic<std::uint8_t>[]> dirty_;
    std::unique_ptr<std::atomic<std::uint8_t>[]> nbr_changed_;
    std::vector<idx_t> dirty_list_;
    std::vector<idx_t> nbr_changed_list_;
    std::vector<std::pair<idx_t, idx_t>> removals_;
    bool track_removals_ = false;
    std::mutex list_mu_;

    std::vector<std::vector<idx_t>> rev_;
    SpinLocks rev_locks_;

    std::vector<std::uint32_t> stamp_;
    std::uint32_t epoch_ = 0;

    const std::vector<idx_t>* comp_labels_ = nullptr;
    idx_t comp_count_ = 0;

    std::vector<idx_t> selected_;
};

}  // namespace coresg
