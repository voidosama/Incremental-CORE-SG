// Benchmark / evaluation driver.
//
// Produces the numbers the thesis needs:
//
//   knn        -- recall and build time, CPU vs CUDA, at several thread counts
//   incr       -- incremental insertion vs a full rebuild: recall of the new
//                 points and the fraction of the graph that had to be touched
//   bridges    -- anti-hub bridges vs the exact MST_kmax, compared on the MSTs
//                 that come out of the CORE-SG for a range of mpts values
//   mpts       -- cost of extracting every MST from one CORE-SG versus
//                 recomputing each one from the complete graph
//
//   coresg_bench all --n 50000 --dim 16

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include "coresg/coresg.hpp"
#include "coresg/eval.hpp"
#include "coresg/io.hpp"

using namespace coresg;

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct Opt {
    std::size_t n = 50000;
    std::uint32_t dim = 16;
    std::uint32_t k = 32;
    std::size_t clusters = 12;
    std::size_t exact_n = 4000;  // ceiling for the O(n^2) references
    std::uint64_t seed = 7;
    bool gpu = true;
};

CoreSGParams make_params(const Opt& o, bool gpu, std::uint32_t instances, unsigned threads) {
    CoreSGParams p;
    p.k_max = o.k;
    p.metric = MetricKind::L2;
    p.threads = threads;
    p.seed = o.seed;
    p.backend.prefer_gpu = gpu;
    p.incr.instances = instances;
    p.incr.min_base_for_split = 2000;
    p.incr.instance_nnd.k = o.k;
    p.incr.instance_nnd.sample = 16;
    p.incr.instance_nnd.iters = 10;
    p.incr.refine_nnd.k = o.k;
    p.incr.refine_nnd.sample = 16;
    p.incr.refine_nnd.iters = 5;
    return p;
}

void bench_knn(const Opt& o) {
    std::printf("\n=== k-NN construction (n=%zu, dim=%u, k=%u) ===\n", o.n, o.dim, o.k);
    std::vector<float> raw = make_blobs(o.n, o.dim, o.clusters, 8.0f, o.seed);

    Dataset ref(o.dim, MetricKind::L2);
    ref.append(raw.data(), o.n);
    ThreadPool pool(0);

    KnnGraph exact;
    const bool have_exact = o.n <= 200000;
    if (have_exact) {
        const double t0 = now_s();
        exact = brute_force_graph(ref, o.k, &pool);
        std::printf("  brute force reference: %.3fs\n", now_s() - t0);
    } else {
        std::printf("  (n too large for a brute force reference; recall omitted)\n");
    }

    std::printf("  %-28s %10s %10s %8s\n", "configuration", "build s", "recall", "instances");
    for (unsigned threads : {1u, 4u, 0u}) {
        CoreSGParams p = make_params(o, false, 1, threads);
        CoreSG sg(p);
        CoreSGStats st = sg.build(raw.data(), o.n, o.dim);
        char label[64];
        std::snprintf(label, sizeof(label), "cpu, %s threads",
                      threads ? std::to_string(threads).c_str() : "all");
        const double rec =
            have_exact ? knn_recall(sg.knn(), exact, o.k).recall : -1.0;
        std::printf("  %-28s %10.3f %10.4f %8zu\n", label, st.knn_seconds, rec,
                    st.knn_stats.instances);
    }

    if (o.gpu && cuda_available()) {
        for (std::uint32_t instances : {1u, 4u}) {
            CoreSGParams p = make_params(o, true, instances, 0);
            CoreSG sg(p);
            CoreSGStats st = sg.build(raw.data(), o.n, o.dim);
            char label[64];
            std::snprintf(label, sizeof(label), "cuda, %u instance(s)", instances);
            const double rec =
                have_exact ? knn_recall(sg.knn(), exact, o.k).recall : -1.0;
            std::printf("  %-28s %10.3f %10.4f %8zu\n", label, st.knn_seconds, rec,
                        st.knn_stats.instances);
        }
    } else {
        std::printf("  (no CUDA device -- GPU rows omitted)\n");
    }
}

void bench_incremental(const Opt& o) {
    std::printf("\n=== incremental insertion vs full rebuild ===\n");
    const std::size_t base = o.n / 2;
    const std::size_t batch = (o.n - base) / 4;
    std::vector<float> raw = make_blobs(o.n, o.dim, o.clusters, 8.0f, o.seed + 1);

    Dataset ref(o.dim, MetricKind::L2);
    ref.append(raw.data(), o.n);
    ThreadPool pool(0);
    KnnGraph exact = brute_force_graph(ref, o.k, &pool);

    CoreSGParams p = make_params(o, o.gpu && cuda_available(), 4, 0);
    CoreSG sg(p);
    CoreSGStats bs = sg.build(raw.data(), base, o.dim);
    std::printf("  initial build n=%zu in %.3fs\n", base, bs.total_seconds);

    std::printf("  %-8s %10s %10s %10s %12s %10s\n", "batch", "insert s", "knn s", "hub s",
                "touched %", "new recall");
    std::size_t cursor = base;
    double incremental_total = bs.total_seconds;
    CoreSGStats last;
    for (int b = 0; b < 4 && cursor < o.n; ++b) {
        const std::size_t count = std::min(batch, o.n - cursor);
        CoreSGStats st = sg.insert(raw.data() + cursor * o.dim, count);
        last = st;
        incremental_total += st.total_seconds;

        // Ground truth has to be the exact k-NN graph over the points that
        // exist *at this moment*. Scoring an early batch against a reference
        // built over the whole file would count neighbours that have not been
        // inserted yet as misses.
        const std::size_t present = cursor + count;
        Dataset sofar(o.dim, MetricKind::L2);
        sofar.append(raw.data(), present);
        KnnGraph exact_now = brute_force_graph(sofar, o.k, &pool);

        std::vector<idx_t> fresh;
        for (idx_t v = static_cast<idx_t>(cursor); v < present; ++v) fresh.push_back(v);
        const RecallResult r = knn_recall_subset(sg.knn(), exact_now, o.k, fresh);
        std::printf("  %-8d %10.3f %10.3f %10.3f %11.1f%% %10.4f\n", b + 1, st.total_seconds,
                    st.knn_seconds, st.hubness_seconds,
                    100.0 * double(st.knn_stats.refine_nodes) / double(st.n), r.recall);
        cursor += count;
    }

    const RecallResult all_r = knn_recall(sg.knn(), exact, o.k);
    std::printf("  incremental total: %.3fs, final %s\n", incremental_total,
                format_recall(all_r).c_str());

    // Parallel fraction. Only the instance phase spreads across devices; the
    // merge and refinement are host side. This is what caps multi-GPU
    // speed-up, so predict the ceiling here rather than discovering it on the
    // cluster: S(p) = 1 / ((1 - f) + f/p) with f = instance fraction.
    {
        const IncrementalKnnStats& k = last.knn_stats;
        const double phase = k.instance_seconds + k.merge_seconds + k.refine_seconds;
        if (phase > 0.0) {
            const double f = k.instance_seconds / phase;
            std::printf("\n  last update phase split: instances %.3fs (%.0f%%) | merge %.3fs "
                        "(%.0f%%) | refine %.3fs (%.0f%%)\n",
                        k.instance_seconds, 100.0 * k.instance_seconds / phase,
                        k.merge_seconds, 100.0 * k.merge_seconds / phase, k.refine_seconds,
                        100.0 * k.refine_seconds / phase);
            std::printf("  parallel fraction f = %.3f -> Amdahl ceiling", f);
            for (int p : {2, 4, 8}) {
                std::printf("  %dx: %.2fx", p, 1.0 / ((1.0 - f) + f / double(p)));
            }
            std::printf("\n  (only the instance phase spreads across devices)\n");
        }
    }

    CoreSG rebuilt(p);
    CoreSGStats rs = rebuilt.build(raw.data(), cursor, o.dim);
    const RecallResult rb = knn_recall(rebuilt.knn(), exact, o.k);
    std::printf("  rebuild from scratch: %.3fs, %s\n", rs.total_seconds,
                format_recall(rb).c_str());
    std::printf("  speed-up of incremental path over rebuild-per-batch: %.2fx\n",
                (rs.total_seconds * 4.0) / (incremental_total - bs.total_seconds + 1e-9));
}

void bench_bridges(const Opt& o) {
    const std::size_t n = std::min(o.n, o.exact_n);
    std::printf("\n=== anti-hub bridges vs exact MST_kmax (n=%zu) ===\n", n);
    std::vector<float> raw = make_blobs(n, o.dim, o.clusters, 8.0f, o.seed + 2);

    CoreSGParams pa = make_params(o, false, 2, 0);
    pa.missing_mode = MissingEdgeMode::AntiHub;
    pa.incr.brute_force_limit = n;  // isolate the bridge approximation
    CoreSG approx(pa);
    CoreSGStats sa = approx.build(raw.data(), n, o.dim);

    CoreSGParams pe = pa;
    pe.missing_mode = MissingEdgeMode::ExactMst;
    CoreSG exact(pe);
    CoreSGStats se = exact.build(raw.data(), n, o.dim);

    std::printf("  bridge construction: anti-hub %.3fs (%zu selected, %zu edges) vs "
                "exact %.3fs (%zu edges)\n",
                sa.missing_seconds, sa.antihubs_selected, sa.missing_edges, se.missing_seconds,
                se.missing_edges);
    std::printf("  speed-up: %.1fx\n", se.missing_seconds / (sa.missing_seconds + 1e-9));

    std::printf("  %-6s %12s %12s %8s %10s %8s\n", "mpts", "approx w", "exact w", "ratio",
                "jaccard", "comps");
    for (std::uint32_t m : {1u, 2u, 4u, 8u, 16u, o.k}) {
        if (m > o.k) continue;
        MstComparison c = compare_msts(approx.mst(m), exact.mst(m));
        std::printf("  %-6u %12.4f %12.4f %8.4f %10.4f %8u\n", m, c.approx_weight,
                    c.exact_weight, c.weight_ratio, c.edge_jaccard,
                    static_cast<unsigned>(c.approx_components));
    }

    // Downstream effect: DBSCAN* labels at a common density threshold.
    MstResult ma = approx.mst(8);
    MstResult me = exact.mst(8);
    double thresh = 0.0;
    for (const Edge& e : me.edges) thresh += e.w;
    thresh /= double(me.edges.empty() ? 1 : me.edges.size());
    idx_t ka = 0, ke = 0;
    const std::vector<idx_t> la =
        dbscan_labels_from_mst(ma, static_cast<idx_t>(n), thresh, &ka);
    const std::vector<idx_t> le =
        dbscan_labels_from_mst(me, static_cast<idx_t>(n), thresh, &ke);
    std::printf("  DBSCAN* at eps=%.4f (mpts=8): ARI %.4f, %u vs %u clusters\n", thresh,
                adjusted_rand_index(la, le), static_cast<unsigned>(ka),
                static_cast<unsigned>(ke));
}

void bench_mpts_sweep(const Opt& o) {
    const std::size_t n = std::min(o.n, o.exact_n);
    std::printf("\n=== cost of a full mpts sweep (n=%zu) ===\n", n);
    std::vector<float> raw = make_blobs(n, o.dim, o.clusters, 8.0f, o.seed + 3);

    CoreSGParams p = make_params(o, o.gpu && cuda_available(), 2, 0);
    CoreSG sg(p);
    CoreSGStats st = sg.build(raw.data(), n, o.dim);

    std::vector<std::uint32_t> values;
    for (std::uint32_t m = 1; m <= o.k; ++m) values.push_back(m);

    const double t0 = now_s();
    std::vector<MstResult> from_sg = sg.all_msts(values);
    const double t_sg = now_s() - t0;

    const double t1 = now_s();
    double t_exact = 0.0;
    const std::size_t probe = std::min<std::size_t>(values.size(), 4);
    for (std::size_t i = 0; i < probe; ++i) sg.exact_mst(values[i]);
    t_exact = (now_s() - t1) * double(values.size()) / double(probe);

    std::printf("  %zu MSTs from one CORE-SG : %.3fs (build %.3fs amortised over the sweep)\n",
                values.size(), t_sg, st.total_seconds);
    std::printf("  same sweep from the complete graph (extrapolated from %zu): %.3fs\n", probe,
                t_exact);
    std::printf("  speed-up: %.1fx\n", t_exact / (t_sg + 1e-9));
}

/// How the choice of denominator in the reverse score affects (a) how
/// informative the anti-hub tag is, (b) which nodes get selected, and (c) the
/// quality of the bridges those nodes produce. The k-NN half is built exactly
/// so that only the hubness layer varies.
void bench_normalizations(const Opt& o) {
    const std::size_t n = std::min(o.n, o.exact_n);
    std::printf("\n=== reverse-score normalisations (n=%zu, d=%u, k=%u) ===\n", n, o.dim, o.k);
    std::vector<float> raw = make_blobs(n, o.dim, o.clusters, 8.0f, o.seed + 4);

    CoreSGParams pe = make_params(o, false, 1, 0);
    pe.missing_mode = MissingEdgeMode::ExactMst;
    pe.incr.brute_force_limit = n;
    CoreSG exact(pe);
    exact.build(raw.data(), n, o.dim);

    std::printf("  %-12s %10s %10s %10s %12s %12s %10s\n", "denominator", "tagged %", "selected",
                "overlap %", "comps(raw)", "w-ratio@8", "ARI@8");

    std::vector<idx_t> reference_selection;
    for (ScoreNormalization norm : {ScoreNormalization::ArithmeticMean,
                                    ScoreNormalization::GeometricMean,
                                    ScoreNormalization::Median}) {
        CoreSGParams p = make_params(o, false, 1, 0);
        p.missing_mode = MissingEdgeMode::AntiHub;
        p.incr.brute_force_limit = n;
        p.hub.normalization = norm;
        p.ensure_connected = false;  // expose the raw component count
        CoreSG raw_sg(p);
        CoreSGStats sr = raw_sg.build(raw.data(), n, o.dim);

        p.ensure_connected = true;
        CoreSG sg(p);
        sg.build(raw.data(), n, o.dim);

        std::vector<idx_t> sel = sg.hubness().selected();
        std::sort(sel.begin(), sel.end());
        double overlap = 100.0;
        if (reference_selection.empty()) {
            reference_selection = sel;
        } else {
            std::vector<idx_t> common;
            std::set_intersection(sel.begin(), sel.end(), reference_selection.begin(),
                                  reference_selection.end(), std::back_inserter(common));
            overlap = 100.0 * double(common.size()) /
                      double(std::max<std::size_t>(1, reference_selection.size()));
        }

        MstResult ma = sg.mst(8);
        MstResult me = exact.mst(8);
        MstComparison c = compare_msts(ma, me);

        double thresh = 0.0;
        for (const Edge& e : me.edges) thresh += e.w;
        thresh /= double(me.edges.empty() ? 1 : me.edges.size());
        idx_t ka = 0, ke = 0;
        const std::vector<idx_t> la =
            dbscan_labels_from_mst(ma, static_cast<idx_t>(n), thresh, &ka);
        const std::vector<idx_t> le =
            dbscan_labels_from_mst(me, static_cast<idx_t>(n), thresh, &ke);

        std::vector<double> scores(n);
        for (idx_t v = 0; v < n; ++v) scores[v] = sg.hubness().reverse_score(v);
        std::sort(scores.begin(), scores.end());
        auto q = [&](double f) { return scores[std::min(n - 1, std::size_t(f * double(n)))]; };

        std::printf("  %-12s %9.1f%% %10zu %9.1f%% %12u %12.5f %10.4f   p10=%.2f p50=%.2f "
                    "p90=%.2f\n",
                    to_string(norm), 100.0 * sg.hubness().antihub_fraction(),
                    sg.hubness().selected().size(), overlap,
                    static_cast<unsigned>(sr.components_before_repair), c.weight_ratio,
                    adjusted_rand_index(la, le), q(0.10), q(0.50), q(0.90));
    }
    std::printf("  (overlap is against the arithmetic-mean selection; comps(raw) is the\n"
                "   component count from k-NN + bridges alone, repair disabled)\n");

    // Why the tag rate barely moves: a neighbourhood is not a uniform sample
    // of the graph. A node with reverse degree r appears in r neighbourhoods,
    // so the reverse degrees seen *inside* a neighbourhood are size-biased
    // upwards -- the friendship paradox. Every denominator built from
    // neighbours inherits that bias, mean or not.
    {
        CoreSGParams p = make_params(o, false, 1, 0);
        p.incr.brute_force_limit = n;
        CoreSG sg(p);
        sg.build(raw.data(), n, o.dim);
        const KnnGraph& g = sg.knn();

        std::vector<std::uint32_t> pop(n);
        for (idx_t v = 0; v < n; ++v) pop[v] = sg.hubness().reverse_degree(v);
        std::vector<std::uint32_t> sorted_pop = pop;
        std::sort(sorted_pop.begin(), sorted_pop.end());

        double sum_pop = 0.0;
        for (std::uint32_t d : pop) sum_pop += d;

        // The same degrees, but weighted the way a neighbourhood sees them.
        double sum_biased = 0.0, count_biased = 0.0;
        std::vector<std::uint32_t> biased;
        biased.reserve(std::size_t(n) * g.k());
        for (idx_t v = 0; v < n; ++v) {
            for (std::uint32_t j = 0; j < g.degree(v); ++j) {
                const std::uint32_t d = pop[g.ids(v)[j]];
                sum_biased += d;
                count_biased += 1.0;
                biased.push_back(d);
            }
        }
        std::sort(biased.begin(), biased.end());

        // E[rd] / (E[rd^2]/E[rd]) = E[rd]^2 / E[rd^2]: the score a node of
        // typical degree actually gets, given that its neighbours are drawn
        // size-biased. This is where "neutral" really sits.
        double sum_sq = 0.0;
        for (std::uint32_t d : pop) sum_sq += double(d) * double(d);
        const double e_rd = sum_pop / double(n);
        const double e_rd2 = sum_sq / double(n);
        const double neutral = (e_rd2 > 0.0) ? (e_rd * e_rd) / e_rd2 : 1.0;

        std::printf("\n  reverse degree seen from outside vs from inside a neighbourhood:\n");
        std::printf("    whole population     : mean %.2f  median %u\n", e_rd, sorted_pop[n / 2]);
        std::printf("    inside neighbourhoods: mean %.2f  median %u\n",
                    sum_biased / count_biased, biased[biased.size() / 2]);
        std::printf("    -> a node of reverse degree r sits in r neighbourhoods, so any\n"
                    "       denominator built from neighbours is size-biased upwards.\n"
                    "       That is what puts most nodes below 1, not the choice of mean.\n");
        std::printf("    predicted neutral point E[rd]^2/E[rd^2] = %.3f  (compare the p50\n"
                    "    column above: the robust denominators land on it, the mean does not)\n",
                    neutral);
    }
}

/// Does the anti-hub bridge set on its own ever fail to connect the graph?
/// Sweeps the shapes that should stress it: many clusters, wide separation,
/// rising dimensionality.
void bench_repair_rate(const Opt& o) {
    std::printf("\n=== how often do anti-hub bridges alone leave the graph split? ===\n");
    std::printf("  budget rule = d*ln(|E|); quota rule adds >=%zu anti-hubs per k-NN "
                "component\n",
                std::size_t(2));
    std::printf("  %-8s %5s %5s %9s %11s %11s %11s\n", "clusters", "dim", "sep", "sel(bud)",
                "comps(bud)", "sel(quota)", "comps(quota)");

    const std::size_t n = std::min<std::size_t>(o.n, 20000);
    std::size_t trials = 0, split_budget = 0, split_quota = 0;

    for (std::size_t clusters : {4u, 12u, 40u, 120u}) {
        for (std::uint32_t dim : {2u, 8u, 32u}) {
            for (float sep : {4.0f, 12.0f}) {
                std::vector<float> raw =
                    make_blobs(n, dim, clusters, sep, o.seed + trials * 17 + 1);
                Opt local = o;
                local.dim = dim;

                auto run = [&](std::size_t quota) {
                    CoreSGParams p = make_params(local, o.gpu && cuda_available(), 2, 0);
                    p.missing_mode = MissingEdgeMode::AntiHub;
                    p.ensure_connected = false;  // expose the raw component count
                    p.hub.min_per_component = quota;
                    CoreSG sg(p);
                    return sg.build(raw.data(), n, dim);
                };
                const CoreSGStats bud = run(0);
                const CoreSGStats quo = run(2);
                ++trials;
                if (bud.components_before_repair > 1) ++split_budget;
                if (quo.components_before_repair > 1) ++split_quota;

                std::printf("  %-8zu %5u %5.0f %9zu %11u %11zu %11u\n", clusters, dim, sep,
                            bud.antihubs_selected,
                            static_cast<unsigned>(bud.components_before_repair),
                            quo.antihubs_selected,
                            static_cast<unsigned>(quo.components_before_repair));
            }
        }
    }
    std::printf("  bridges alone left the graph split in %zu / %zu configs with the budget "
                "rule, %zu / %zu with the quota\n",
                split_budget, trials, split_quota, trials);
}

/// Is maintaining components across batches actually cheaper than recomputing
/// them? Compares three ways of answering "what are the components":
///   legacy      -- materialise the undirected edge list, then union-find
///   rebuild     -- union-find straight over the adjacency, parallel
///   incremental -- certificate + partial re-derivation
/// across batch sizes, since the churn per batch is what decides it.
void bench_connectivity(const Opt& o) {
    std::printf("\n=== connectivity: recompute vs maintain ===\n");
    const std::size_t base = o.n;
    ThreadPool pool(0);

    std::printf("  %-9s %10s %11s %12s %13s %11s %10s\n", "batch", "destroyed", "dirty cmp",
                "legacy ms", "rebuild ms", "incr ms", "verdict");

    for (double frac : {0.01, 0.05, 0.20}) {
        const std::size_t batch = std::max<std::size_t>(1, std::size_t(frac * double(base)));
        std::vector<float> raw = make_blobs(base + batch, o.dim, o.clusters, 8.0f, o.seed + 5);

        CoreSGParams p = make_params(o, o.gpu && cuda_available(), 4, 0);
        p.incremental_connectivity = true;
        CoreSG sg(p);
        sg.build(raw.data(), base, o.dim);
        CoreSGStats st = sg.insert(raw.data() + base * o.dim, batch);

        const KnnGraph& g = sg.knn();

        // Legacy: build the edge list, then run union-find over it.
        const double t0 = now_s();
        {
            idx_t nc = 0;
            std::vector<Edge> edges = g.undirected_edges(&pool);
            connected_components(edges, g.size(), &nc);
        }
        const double legacy_ms = (now_s() - t0) * 1000.0;

        // Rebuild: same answer, no edge list.
        ConnectivityTracker fresh;
        fresh.reset(g.size());
        const ConnectivityStats rb = fresh.rebuild(g, &pool);

        const double incr_ms = st.conn_stats.seconds * 1000.0;
        const char* verdict = (incr_ms < rb.seconds * 1000.0) ? "incremental" : "rebuild";
        std::printf("  %-9zu %10zu %11zu %12.2f %13.2f %11.2f %10s\n", batch,
                    st.conn_stats.destroyed_edges, st.conn_stats.dirty_components, legacy_ms,
                    rb.seconds * 1000.0, incr_ms, verdict);
    }
    std::printf("  (legacy is what repair_connectivity used to do; rebuild is the same\n"
                "   answer without materialising n*k/2 edges)\n");
}

/// Ambient vs intrinsic dimensionality, and what each implies for the anti-hub
/// budget d*ln(|E|) and the cost of the bridge MST it triggers.
void bench_intrinsic_dim(const Opt& o, const std::string& path, const std::string& label,
                         std::uint32_t raw_dim) {
    std::vector<float> raw;
    std::uint32_t dim = 0;
    std::size_t n = 0;
    std::string name = label;

    if (path.empty()) {
        n = std::min<std::size_t>(o.n, 100000);
        dim = o.dim;
        raw = make_blobs(n, dim, o.clusters, 8.0f, o.seed + 6);
        name = label.empty() ? "synthetic" : label;
    } else {
        RawData rd = read_any(path, raw_dim, std::min<std::size_t>(o.n, 100000));
        raw = std::move(rd.values);
        dim = rd.dim;
        n = rd.n;
    }
    if (n == 0) return;

    CoreSGParams p = make_params(o, o.gpu && cuda_available(), 2, 0);
    p.metric = MetricKind::L2;
    CoreSG sg(p);
    sg.build(raw.data(), n, dim);

    const IntrinsicDimStats id = sg.intrinsic_dimension();
    const std::size_t edges = sg.knn().undirected_edge_count(sg.pool());
    const double lg = std::log(double(edges));

    auto budget = [&](double d) {
        double b = d * lg;
        return std::min<double>(b, double(n));
    };
    auto ops = [&](double m, double d) { return m * m * 0.5 * d / 1e9; };

    const double id_used = id.reliable() ? std::min(id.value(), double(dim)) : double(dim);
    const double b_amb = budget(double(dim));
    const double b_int = budget(id_used);

    std::printf("  %-14s n=%-7zu ambient d=%-5u  %s\n", name.c_str(), n, dim,
                format_intrinsic_dim(id).c_str());
    std::printf("  %-14s budget: ambient %8.0f (%5.1f%% of n, %7.1f G ops) | intrinsic "
                "%8.0f (%5.1f%% of n, %7.2f G ops)  -> %.0fx cheaper\n\n",
                "", b_amb, 100.0 * b_amb / double(n), ops(b_amb, double(dim)), b_int,
                100.0 * b_int / double(n), ops(b_int, double(dim)),
                ops(b_amb, double(dim)) / (ops(b_int, double(dim)) + 1e-12));
}

void bench_intrinsic_dim_all(const Opt& o, const std::vector<std::string>& datasets) {
    std::printf("\n=== ambient vs intrinsic dimensionality ===\n");
    std::printf("  (intrinsic = Levina-Bickel / Hill MLE from the stored k-NN radii)\n\n");
    if (datasets.empty()) {
        bench_intrinsic_dim(o, "", "synthetic", 0);
        return;
    }
    for (const std::string& spec : datasets) {
        // "label=path" or just "path"
        const std::size_t eq = spec.find('=');
        const std::string label = (eq == std::string::npos) ? "" : spec.substr(0, eq);
        const std::string path = (eq == std::string::npos) ? spec : spec.substr(eq + 1);
        try {
            bench_intrinsic_dim(o, path, label.empty() ? path : label, 0);
        } catch (const std::exception& e) {
            std::printf("  %-14s SKIPPED: %s\n", label.c_str(), e.what());
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Opt o;
    std::vector<std::string> datasets;
    std::string what = argc > 1 ? argv[1] : "all";
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--dataset" && i + 1 < argc) { datasets.push_back(argv[++i]); continue; }
        if (i >= argc - 1) break;
        if (k == "--n") o.n = static_cast<std::size_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (k == "--dim") o.dim = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (k == "--k") o.k = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (k == "--clusters") o.clusters = static_cast<std::size_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (k == "--exact-n") o.exact_n = static_cast<std::size_t>(std::strtoul(argv[++i], nullptr, 10));
        else if (k == "--seed") o.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (k == "--gpu") o.gpu = std::strtol(argv[++i], nullptr, 10) != 0;
    }

    std::printf("coresg benchmark: n=%zu dim=%u k=%u clusters=%zu cuda=%s\n", o.n, o.dim, o.k,
                o.clusters, cuda_available() ? "yes" : "no");
    try {
        if (what == "all" || what == "knn") bench_knn(o);
        if (what == "all" || what == "incr") bench_incremental(o);
        if (what == "all" || what == "bridges") bench_bridges(o);
        if (what == "all" || what == "mpts") bench_mpts_sweep(o);
        if (what == "all" || what == "norm") bench_normalizations(o);
        if (what == "all" || what == "repair") bench_repair_rate(o);
        if (what == "all" || what == "conn") bench_connectivity(o);
        if (what == "all" || what == "id") bench_intrinsic_dim_all(o, datasets);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
