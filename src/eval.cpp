#include "coresg/eval.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "coresg/nndescent.hpp"
#include "coresg/thread_pool.hpp"

namespace coresg {

KnnGraph brute_force_graph(const Dataset& data, std::uint32_t k, ThreadPool* pool) {
    KnnGraph g(static_cast<idx_t>(data.size()), k);
    brute_force_knn(data.view(), data.metric(), &g, pool);
    return g;
}

namespace {

RecallResult recall_impl(const KnnGraph& approx, const KnnGraph& exact, std::uint32_t k,
                         const std::vector<idx_t>* nodes) {
    RecallResult r;
    r.k = k;
    const std::size_t count = nodes ? nodes->size() : exact.size();
    if (count == 0 || k == 0) return r;

    std::size_t hits = 0;
    std::size_t total = 0;
    double sum_approx = 0.0, sum_exact = 0.0;
    std::size_t counted = 0;

    std::unordered_set<idx_t> truth;
    for (std::size_t t = 0; t < count; ++t) {
        const idx_t v = nodes ? (*nodes)[t] : static_cast<idx_t>(t);
        const std::uint32_t ce = std::min(k, exact.degree(v));
        const std::uint32_t ca = std::min(k, approx.degree(v));
        if (ce == 0) continue;

        truth.clear();
        const idx_t* E = exact.ids(v);
        for (std::uint32_t j = 0; j < ce; ++j) truth.insert(E[j]);

        const idx_t* A = approx.ids(v);
        for (std::uint32_t j = 0; j < ca; ++j) {
            if (truth.count(A[j])) ++hits;
        }
        total += ce;

        // Distance ratio at the k-th neighbour: catches the case where recall
        // is low but the graph is still metrically fine (ties, duplicates).
        if (ca > 0) {
            sum_approx += approx.dists(v)[ca - 1];
            sum_exact += exact.dists(v)[ce - 1];
            ++counted;
        }
    }
    r.recall = total ? double(hits) / double(total) : 0.0;
    r.distance_ratio = (counted && sum_exact > 0.0) ? sum_approx / sum_exact : 1.0;
    r.nodes = counted;
    return r;
}

}  // namespace

RecallResult knn_recall(const KnnGraph& approx, const KnnGraph& exact, std::uint32_t k) {
    return recall_impl(approx, exact, k, nullptr);
}

RecallResult knn_recall_subset(const KnnGraph& approx, const KnnGraph& exact, std::uint32_t k,
                               const std::vector<idx_t>& nodes) {
    return recall_impl(approx, exact, k, &nodes);
}

MstComparison compare_msts(const MstResult& approx, const MstResult& exact) {
    MstComparison c;
    c.approx_weight = approx.weight;
    c.exact_weight = exact.weight;
    c.weight_ratio = (exact.weight > 0.0) ? approx.weight / exact.weight : 1.0;
    c.approx_edges = approx.edges.size();
    c.exact_edges = exact.edges.size();
    c.approx_components = approx.num_components;
    c.exact_components = exact.num_components;

    std::unordered_map<std::uint64_t, float> ex;
    ex.reserve(exact.edges.size() * 2);
    for (const Edge& e : exact.edges) ex.emplace(edge_key(e.u, e.v), e.w);

    std::size_t shared = 0;
    for (const Edge& e : approx.edges) {
        auto it = ex.find(edge_key(e.u, e.v));
        if (it != ex.end()) ++shared;
    }
    c.shared_edges = shared;
    const std::size_t uni = c.approx_edges + c.exact_edges - shared;
    c.edge_jaccard = uni ? double(shared) / double(uni) : 1.0;

    // Compare the sorted weight profiles: the density levels at which merges
    // happen matter more to a density-based hierarchy than which particular
    // edge realised each merge.
    std::vector<float> wa, we;
    wa.reserve(approx.edges.size());
    we.reserve(exact.edges.size());
    for (const Edge& e : approx.edges) wa.push_back(e.w);
    for (const Edge& e : exact.edges) we.push_back(e.w);
    std::sort(wa.begin(), wa.end());
    std::sort(we.begin(), we.end());
    const std::size_t m = std::min(wa.size(), we.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < m; ++i) worst = std::max(worst, double(std::fabs(wa[i] - we[i])));
    c.max_edge_weight_gap = worst;
    return c;
}

double adjusted_rand_index(const std::vector<idx_t>& a, const std::vector<idx_t>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    const std::size_t n = a.size();

    std::unordered_map<std::uint64_t, double> joint;
    std::unordered_map<idx_t, double> ca, cb;
    for (std::size_t i = 0; i < n; ++i) {
        joint[(std::uint64_t(a[i]) << 32) | b[i]] += 1.0;
        ca[a[i]] += 1.0;
        cb[b[i]] += 1.0;
    }
    auto comb2 = [](double x) { return x * (x - 1.0) / 2.0; };

    double sum_joint = 0.0, sum_a = 0.0, sum_b = 0.0;
    for (const auto& kv : joint) sum_joint += comb2(kv.second);
    for (const auto& kv : ca) sum_a += comb2(kv.second);
    for (const auto& kv : cb) sum_b += comb2(kv.second);

    const double total = comb2(double(n));
    const double expected = (total > 0.0) ? (sum_a * sum_b) / total : 0.0;
    const double maxi = 0.5 * (sum_a + sum_b);
    const double denom = maxi - expected;
    if (std::fabs(denom) < 1e-12) return 1.0;
    return (sum_joint - expected) / denom;
}

std::vector<idx_t> dbscan_labels_from_mst(const MstResult& mst, idx_t n, double threshold,
                                          idx_t* num_clusters) {
    UnionFind uf(n);
    for (const Edge& e : mst.edges) {
        if (double(e.w) <= threshold) uf.unite(e.u, e.v);
    }
    std::vector<idx_t> labels = uf.labels();
    if (num_clusters) *num_clusters = uf.components();
    return labels;
}

std::string format_recall(const RecallResult& r) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(4);
    oss << "recall@" << r.k << " = " << r.recall << " (dist ratio " << r.distance_ratio << ", "
        << r.nodes << " nodes)";
    return oss.str();
}

std::string format_mst_comparison(const MstComparison& c) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(6);
    oss << "weight " << c.approx_weight << " vs " << c.exact_weight << " (ratio "
        << c.weight_ratio << "), edges " << c.approx_edges << "/" << c.exact_edges << ", shared "
        << c.shared_edges << " (jaccard " << c.edge_jaccard << "), components "
        << c.approx_components << "/" << c.exact_components << ", max weight gap "
        << c.max_edge_weight_gap;
    return oss.str();
}

}  // namespace coresg
