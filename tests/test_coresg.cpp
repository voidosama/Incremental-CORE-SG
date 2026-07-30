#include "coresg/coresg.hpp"

#include <cstdio>
#include <vector>

#include "coresg/eval.hpp"
#include "coresg/io.hpp"
#include "test_util.hpp"

using namespace coresg;

namespace {

CoreSGParams base_params(std::uint32_t k_max, MissingEdgeMode mode, bool exact_knn) {
    CoreSGParams p;
    p.k_max = k_max;
    p.metric = MetricKind::L2;
    p.missing_mode = mode;
    p.threads = 4;
    p.backend.prefer_gpu = false;
    p.incr.instances = 2;
    p.incr.instance_nnd.k = k_max;
    p.incr.instance_nnd.sample = 12;
    p.incr.instance_nnd.iters = 10;
    p.incr.refine_nnd.k = k_max;
    p.incr.refine_nnd.sample = 12;
    p.incr.refine_nnd.iters = 6;
    if (exact_knn) p.incr.brute_force_limit = 100000;
    return p;
}

/// Theorem 1 of the CORE-SG paper: for every mpts <= k_max there is an
/// MST_mpts contained in kmax-NNG union MST_kmax. With an exact k-NN half and
/// the exact MST_kmax as the bridge set, the MST extracted from the CORE-SG
/// must therefore weigh exactly what the MST over the complete mutual
/// reachability graph weighs.
void test_theorem1_exact() {
    const std::size_t n = 700;
    const std::uint32_t k_max = 16;
    std::vector<float> raw = make_blobs(n, 5, 4, 6.0f, 1234);

    CoreSG sg(base_params(k_max, MissingEdgeMode::ExactMst, /*exact_knn=*/true));
    CoreSGStats st = sg.build(raw.data(), n, 5);
    CHECK_EQ(st.components_after_repair, 1u);

    const std::uint32_t values[] = {1, 2, 4, 8, 12, 16};
    for (std::uint32_t mpts : values) {
        MstResult from_sg = sg.mst(mpts);
        MstResult exact = sg.exact_mst(mpts);
        CHECK_EQ(from_sg.num_components, 1u);
        CHECK_EQ(exact.num_components, 1u);
        CHECK_EQ(from_sg.edges.size(), exact.edges.size());
        const double rel = (exact.weight > 0.0)
                               ? std::fabs(from_sg.weight - exact.weight) / exact.weight
                               : 0.0;
        CHECK_MSG(rel < 1e-5, "mpts=" + std::to_string(mpts) + " weight " +
                                  std::to_string(from_sg.weight) + " vs " +
                                  std::to_string(exact.weight));
    }
    std::printf("  theorem 1 holds for mpts in [1, %u]\n", k_max);
}

/// The proposed approximation: bridges estimated from the anti-hubs instead of
/// a real MST. It cannot be better than exact, and this pins down how much
/// worse it is on data with clearly separated, differently dense clusters.
void test_antihub_vs_exact() {
    const std::size_t n = 900;
    const std::uint32_t k_max = 16;
    std::vector<float> raw = make_blobs(n, 6, 5, 8.0f, 99);

    CoreSG approx(base_params(k_max, MissingEdgeMode::AntiHub, true));
    CoreSGStats sa = approx.build(raw.data(), n, 6);

    CoreSG exact(base_params(k_max, MissingEdgeMode::ExactMst, true));
    exact.build(raw.data(), n, 6);

    std::printf("  anti-hub bridges: %zu selected, %zu missing edges, %zu repair edges, "
                "%u components before repair\n",
                sa.antihubs_selected, sa.missing_edges, sa.repair_edges,
                static_cast<unsigned>(sa.components_before_repair));

    const std::uint32_t values[] = {2, 8, 16};
    for (std::uint32_t mpts : values) {
        MstResult a = approx.mst(mpts);
        MstResult e = exact.mst(mpts);
        MstComparison c = compare_msts(a, e);
        std::printf("  mpts=%2u  %s\n", mpts, format_mst_comparison(c).c_str());
        CHECK_EQ(a.num_components, 1u);
        // An approximate bridge set can only make the spanning tree heavier.
        CHECK_GE(c.weight_ratio, 0.999);
        CHECK_LE(c.weight_ratio, 1.5);
    }
}

void test_knn_only_mode_and_connectivity() {
    const std::size_t n = 600;
    std::vector<float> raw = make_blobs(n, 4, 6, 12.0f, 5);

    // Well separated clusters: the k-NN graph alone must fall apart, and the
    // repair step must put it back together.
    CoreSGParams p = base_params(12, MissingEdgeMode::None, true);
    p.ensure_connected = false;
    CoreSG bare(p);
    CoreSGStats sb = bare.build(raw.data(), n, 4);
    CHECK_GE(sb.components_before_repair, 2u);
    CHECK_EQ(sb.missing_edges, 0u);

    p.ensure_connected = true;
    CoreSG repaired(p);
    CoreSGStats sr = repaired.build(raw.data(), n, 4);
    CHECK_EQ(sr.components_after_repair, 1u);
    CHECK_GE(sr.repair_edges, 1u);
    CHECK_EQ(repaired.mst(8).num_components, 1u);
}

void test_edges_contain_knn_and_bridges() {
    const std::size_t n = 500;
    std::vector<float> raw = make_blobs(n, 4, 3, 6.0f, 17);
    CoreSG sg(base_params(12, MissingEdgeMode::AntiHub, true));
    CoreSGStats st = sg.build(raw.data(), n, 4);

    const std::vector<Edge> e = sg.edges(12);
    CHECK_EQ(e.size(), st.knn_edges + st.missing_edges);

    // Every weight must be a valid mutual reachability distance.
    const std::vector<dist_t> core = sg.core_distances(12);
    bool ok = true;
    for (const Edge& x : e) {
        const dist_t floor_w = std::max(core[x.u], core[x.v]);
        if (x.w + 1e-4f < floor_w) ok = false;
    }
    CHECK_MSG(ok, "an edge weight fell below max(core(u), core(v))");
}

void test_mpts_monotonicity() {
    // Core distances grow with mpts, so no mutual reachability weight can
    // shrink -- and neither can the MST's total weight.
    const std::size_t n = 600;
    std::vector<float> raw = make_blobs(n, 5, 4, 6.0f, 8);
    CoreSG sg(base_params(16, MissingEdgeMode::AntiHub, true));
    sg.build(raw.data(), n, 5);

    double prev = -1.0;
    for (std::uint32_t mpts = 1; mpts <= 16; ++mpts) {
        const double w = sg.mst(mpts).weight;
        CHECK_GE(w, prev - 1e-4);
        prev = w;
    }
}

void test_save_load_roundtrip() {
    const std::size_t n = 400;
    std::vector<float> raw = make_blobs(n, 4, 3, 6.0f, 31);
    CoreSGParams p = base_params(12, MissingEdgeMode::AntiHub, true);

    double before = 0.0;
    {
        CoreSG sg(p);
        sg.build(raw.data(), n, 4);
        before = sg.mst(8).weight;
        sg.save("test_roundtrip.coresg");
    }
    CoreSG loaded(p);
    loaded.load("test_roundtrip.coresg");
    CHECK_EQ(loaded.knn().size(), static_cast<idx_t>(n));
    CHECK_NEAR(loaded.mst(8).weight, before, 1e-4);
    std::remove("test_roundtrip.coresg");
}

#if CORESG_WITH_CUDA
void test_gpu_matches_cpu() {
    if (!cuda_available()) {
        std::printf("  (no CUDA device visible -- GPU parity test skipped)\n");
        return;
    }
    const std::size_t n = 20000;
    const std::uint32_t k_max = 24;
    std::vector<float> raw = make_blobs(n, 16, 8, 7.0f, 4242);

    CoreSGParams cpu = base_params(k_max, MissingEdgeMode::AntiHub, false);
    cpu.incr.instances = 2;
    CoreSG a(cpu);
    CoreSGStats sa = a.build(raw.data(), n, 16);

    CoreSGParams gpu = cpu;
    gpu.backend.prefer_gpu = true;
    CoreSG b(gpu);
    CoreSGStats sb = b.build(raw.data(), n, 16);

    Dataset ref(16, MetricKind::L2);
    ref.append(raw.data(), n);
    ThreadPool pool(8);
    KnnGraph exact = brute_force_graph(ref, k_max, &pool);

    RecallResult ra = knn_recall(a.knn(), exact, k_max);
    RecallResult rb = knn_recall(b.knn(), exact, k_max);
    std::printf("  cpu  %s in %.3fs\n", format_recall(ra).c_str(), sa.knn_seconds);
    std::printf("  cuda %s in %.3fs\n", format_recall(rb).c_str(), sb.knn_seconds);
    CHECK_GE(ra.recall, 0.90);
    CHECK_GE(rb.recall, 0.90);
    // The device path must not be materially worse than the host path.
    CHECK_GE(rb.recall, ra.recall - 0.05);
    CHECK_EQ(b.mst(k_max).num_components, 1u);
}
#endif

}  // namespace

int main() {
    test_theorem1_exact();
    test_antihub_vs_exact();
    test_knn_only_mode_and_connectivity();
    test_edges_contain_knn_and_bridges();
    test_mpts_monotonicity();
    test_save_load_roundtrip();
#if CORESG_WITH_CUDA
    test_gpu_matches_cpu();
#endif
    return coresg_test::summary("coresg");
}
