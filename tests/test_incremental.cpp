#include <cstdio>
#include <vector>

#include "coresg/coresg.hpp"
#include "coresg/eval.hpp"
#include "coresg/io.hpp"
#include "test_util.hpp"

using namespace coresg;

namespace {

CoreSGParams params(std::uint32_t k_max, std::uint32_t instances) {
    CoreSGParams p;
    p.k_max = k_max;
    p.metric = MetricKind::L2;
    p.missing_mode = MissingEdgeMode::AntiHub;
    p.threads = 8;
    p.backend.prefer_gpu = false;
    p.incr.instances = instances;
    p.incr.min_base_for_split = 1000;
    p.incr.instance_nnd.k = k_max;
    p.incr.instance_nnd.sample = 14;
    p.incr.instance_nnd.iters = 10;
    p.incr.refine_nnd.k = k_max;
    p.incr.refine_nnd.sample = 14;
    p.incr.refine_nnd.iters = 6;
    return p;
}

/// The core claim of the incremental scheme: inserting a batch through
/// sampled instances plus a restricted refinement gives the new points
/// neighbourhoods comparable to a full rebuild, at a fraction of the work.
void test_incremental_recall() {
    const std::size_t n_base = 12000;
    const std::size_t n_new = 3000;
    const std::uint32_t dim = 12;
    const std::uint32_t k = 20;

    std::vector<float> all = make_blobs(n_base + n_new, dim, 8, 6.0f, 2024);
    const float* base = all.data();
    const float* fresh = all.data() + n_base * dim;

    CoreSG sg(params(k, 4));
    sg.build(base, n_base, dim);
    CoreSGStats up = sg.insert(fresh, n_new);

    Dataset ref(dim, MetricKind::L2);
    ref.append(all.data(), n_base + n_new);
    ThreadPool pool(8);
    KnnGraph exact = brute_force_graph(ref, k, &pool);

    std::vector<idx_t> new_nodes;
    for (idx_t v = static_cast<idx_t>(n_base); v < n_base + n_new; ++v) new_nodes.push_back(v);

    RecallResult all_r = knn_recall(sg.knn(), exact, k);
    RecallResult new_r = knn_recall_subset(sg.knn(), exact, k, new_nodes);
    std::printf("  after insert: overall %s\n", format_recall(all_r).c_str());
    std::printf("  new points  : %s\n", format_recall(new_r).c_str());
    std::printf("  cost: %zu instances over %zu nodes, refinement touched %zu nodes "
                "(%.1f%% of the graph; %zu new + %zu ring-1%s)\n",
                up.knn_stats.instances, up.knn_stats.instance_nodes, up.knn_stats.refine_nodes,
                100.0 * double(up.knn_stats.refine_nodes) / double(n_base + n_new),
                up.knn_stats.refine_ring0, up.knn_stats.refine_ring1,
                up.knn_stats.refine_capped ? ", capped" : "");
    // The update must stay well below a full pass over the graph.
    CHECK_LE(double(up.knn_stats.refine_nodes), 0.30 * double(n_base + n_new));

    CHECK_GE(new_r.recall, 0.90);
    CHECK_GE(all_r.recall, 0.90);

    std::string err;
    CHECK_MSG(sg.knn().validate(&err), err);

    // A rebuild from scratch on the same data is the quality bar.
    CoreSG rebuilt(params(k, 4));
    rebuilt.build(all.data(), n_base + n_new, dim);
    RecallResult rb = knn_recall(rebuilt.knn(), exact, k);
    std::printf("  full rebuild: %s\n", format_recall(rb).c_str());
    CHECK_GE(all_r.recall, rb.recall - 0.05);
}

/// Repeated small batches must not let the graph rot: the reverse degrees,
/// the anti-hub selection and the spanning property all have to survive.
void test_repeated_batches() {
    const std::size_t n_base = 6000;
    const std::size_t batch = 1000;
    const int batches = 5;
    const std::uint32_t dim = 8;
    const std::uint32_t k = 16;

    std::vector<float> all = make_blobs(n_base + batch * batches, dim, 6, 6.0f, 777);
    CoreSG sg(params(k, 3));
    sg.build(all.data(), n_base, dim);

    for (int b = 0; b < batches; ++b) {
        const float* ptr = all.data() + (n_base + b * batch) * dim;
        CoreSGStats st = sg.insert(ptr, batch);
        CHECK_EQ(st.components_after_repair, 1u);
        CHECK_GE(st.antihubs_selected, 1u);
    }

    const std::size_t total = n_base + batch * batches;
    CHECK_EQ(sg.knn().size(), static_cast<idx_t>(total));
    std::string err;
    CHECK_MSG(sg.knn().validate(&err), err);

    // Reverse degrees tracked through 5 batches of edge churn must still match
    // a recount of the final graph.
    std::vector<std::uint32_t> rd(total, 0);
    for (idx_t v = 0; v < total; ++v) {
        for (std::uint32_t j = 0; j < sg.knn().degree(v); ++j) ++rd[sg.knn().ids(v)[j]];
    }
    std::size_t drift = 0;
    for (idx_t v = 0; v < total; ++v) {
        if (sg.hubness().reverse_degree(v) != rd[v]) ++drift;
    }
    CHECK_EQ(drift, 0u);

    Dataset ref(dim, MetricKind::L2);
    ref.append(all.data(), total);
    ThreadPool pool(8);
    KnnGraph exact = brute_force_graph(ref, k, &pool);
    RecallResult r = knn_recall(sg.knn(), exact, k);
    std::printf("  after %d batches (n=%zu): %s\n", batches, total, format_recall(r).c_str());
    CHECK_GE(r.recall, 0.88);

    CHECK_EQ(sg.mst(k).num_components, 1u);
}

/// The anti-hub budget is d * ln(|E|), so it must grow as points arrive.
void test_budget_grows_with_data() {
    const std::uint32_t dim = 6;
    std::vector<float> all = make_blobs(20000, dim, 5, 6.0f, 55);
    CoreSG sg(params(16, 3));
    CoreSGStats s0 = sg.build(all.data(), 5000, dim);
    CoreSGStats s1 = sg.insert(all.data() + 5000 * dim, 15000);
    std::printf("  selected anti-hubs: %zu at n=5000 -> %zu at n=20000\n", s0.antihubs_selected,
                s1.antihubs_selected);
    CHECK_GE(s1.antihubs_selected, s0.antihubs_selected);
}

void test_instance_counts_agree() {
    // Whatever the number of instances, quality must be comparable: the
    // instance count is a scheduling knob, not an accuracy knob.
    const std::size_t n_base = 8000;
    const std::size_t n_new = 2000;
    const std::uint32_t dim = 10;
    const std::uint32_t k = 16;
    std::vector<float> all = make_blobs(n_base + n_new, dim, 6, 6.0f, 606);

    Dataset ref(dim, MetricKind::L2);
    ref.append(all.data(), n_base + n_new);
    ThreadPool pool(8);
    KnnGraph exact = brute_force_graph(ref, k, &pool);

    std::vector<idx_t> new_nodes;
    for (idx_t v = static_cast<idx_t>(n_base); v < n_base + n_new; ++v) new_nodes.push_back(v);

    double first = -1.0;
    for (std::uint32_t instances : {1u, 2u, 6u}) {
        CoreSG sg(params(k, instances));
        sg.build(all.data(), n_base, dim);
        sg.insert(all.data() + n_base * dim, n_new);
        RecallResult r = knn_recall_subset(sg.knn(), exact, k, new_nodes);
        std::printf("  instances=%u -> new point %s\n", instances, format_recall(r).c_str());
        CHECK_GE(r.recall, 0.88);
        if (first < 0.0) {
            first = r.recall;
        } else {
            CHECK_LE(std::fabs(r.recall - first), 0.10);
        }
    }
}

}  // namespace

int main() {
    test_incremental_recall();
    test_repeated_batches();
    test_budget_grows_with_data();
    test_instance_counts_agree();
    return coresg_test::summary("incremental");
}
