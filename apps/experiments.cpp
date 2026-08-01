// Thesis experiment harness.
//
// Every run emits long-format CSV -- one row per measurement -- so results
// from different machines and different days concatenate and load straight
// into pandas/R without reshaping:
//
//   experiment,dataset,n,dim,k,config,value,seed,batch,metric,measurement
//
// Design rules this file follows, because they are what makes the numbers
// defensible:
//
//   * every experiment runs over several seeds; nothing reports a single run;
//   * the ground truth for a *quality* claim is the exact CORE-SG (exact k-NN
//     half + exact MST_kmax), never human class labels. The claim under test
//     is "the approximation matches the thing it approximates", and comparing
//     to class labels would instead measure whether HDBSCAN* likes the data
//     set, which is a different question;
//   * MST agreement is reported as total-weight ratio and as the sorted
//     weight profile, never as edge overlap. All MSTs of a graph share a
//     weight multiset and induce identical single-linkage hierarchies, so two
//     optimal trees can overlap very little; overlap measures tie-breaking,
//     not quality.
//
// Usage:
//   coresg_exp --experiment incremental --data X.fvecs --k 32 --seeds 5 \
//              --out results.csv
//   coresg_exp --experiment all --synthetic 200000 --dim 32 --out results.csv

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "coresg/coresg.hpp"
#include "coresg/eval.hpp"
#include "coresg/io.hpp"
#include "coresg/resource.hpp"

using namespace coresg;

namespace {

using Clock = std::chrono::steady_clock;
double since(const Clock::time_point& t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

// ---------------------------------------------------------------------------
// CSV sink
// ---------------------------------------------------------------------------

class Csv {
public:
    Csv(const std::string& path, bool echo) : echo_(echo) {
        const bool exists = [&] {
            std::ifstream probe(path);
            return probe.good() && probe.peek() != std::ifstream::traits_type::eof();
        }();
        out_.open(path, std::ios::app);
        if (!out_) throw std::runtime_error("cannot open " + path + " for writing");
        if (!exists) {
            out_ << "experiment,dataset,n,dim,k,config,value,seed,batch,metric,measurement\n";
        }
    }

    void row(const std::string& experiment, const std::string& dataset, std::size_t n,
             std::uint32_t dim, std::uint32_t k, const std::string& config,
             const std::string& value, std::uint64_t seed, long batch, const std::string& metric,
             double measurement) {
        out_ << experiment << ',' << dataset << ',' << n << ',' << dim << ',' << k << ','
             << config << ',' << value << ',' << seed << ',' << batch << ',' << metric << ','
             << measurement << '\n';
        out_.flush();  // results survive a crash three hours into a sweep
        if (echo_) {
            std::printf("    %-22s %-16s %-10s seed=%llu batch=%ld  %-22s %.6f\n",
                        experiment.c_str(), config.c_str(), value.c_str(),
                        static_cast<unsigned long long>(seed), batch, metric.c_str(),
                        measurement);
        }
    }

private:
    std::ofstream out_;
    bool echo_;
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Options {
    std::string experiment = "all";
    std::string data_path;
    std::string dataset_name;
    std::uint32_t raw_dim = 0;      // for headerless .bin
    std::size_t limit = 0;          // cap rows read
    std::size_t synthetic = 0;      // generate n points instead of loading
    std::uint32_t dim = 32;
    std::size_t clusters = 12;
    std::uint32_t k = 32;
    int seeds = 5;
    int batches = 8;
    double base_fraction = 0.5;
    std::size_t exact_limit = 5000;  // ceiling for O(n^2) references
    std::string metric = "l2";
    bool gpu = true;
    unsigned threads = 0;
    std::string out = "results.csv";
    bool echo = true;
    /// Comma-separated instance counts for the scaling experiment. Empty means
    /// {devices, 2*devices, 4*devices}. The capacity run overrides it to reach
    /// instances=1, which is the configuration that cannot fit on one card and
    /// is therefore the whole point of that experiment.
    std::string scaling_instances;
    /// When a device runs out of memory the backend normally finishes the job
    /// on the host. That is right for a production run and wrong for the
    /// capacity probe: falling back means a 10M-point NN-Descent on the CPU,
    /// turning a two-second demonstration into hours. Disable it there.
    bool oom_fallback = true;
};

std::vector<std::uint32_t> parse_uint_list(const std::string& s) {
    std::vector<std::uint32_t> out;
    std::size_t start = 0;
    while (start <= s.size()) {
        const std::size_t comma = s.find(',', start);
        const std::string tok = s.substr(start, comma - start);
        if (!tok.empty()) out.push_back(static_cast<std::uint32_t>(std::stoul(tok)));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

struct Data {
    std::vector<float> values;
    std::uint32_t dim = 0;
    std::size_t n = 0;
    std::string name;
    MetricKind metric = MetricKind::L2;
};

Data load(const Options& o) {
    Data d;
    if (!metric_from_string(o.metric, &d.metric)) {
        throw std::runtime_error("unknown metric " + o.metric);
    }
    if (!o.data_path.empty()) {
        RawData raw = read_any(o.data_path, o.raw_dim, o.limit);
        d.values = std::move(raw.values);
        d.dim = raw.dim;
        d.n = raw.n;
        d.name = o.dataset_name.empty() ? o.data_path : o.dataset_name;
    } else {
        const std::size_t n = o.synthetic ? o.synthetic : 100000;
        d.values = make_blobs(n, o.dim, o.clusters, 8.0f, 20260729);
        d.dim = o.dim;
        d.n = n;
        d.name = o.dataset_name.empty()
                     ? ("blobs" + std::to_string(n) + "d" + std::to_string(o.dim))
                     : o.dataset_name;
    }
    std::printf("dataset %s: %zu x %u (%s)\n", d.name.c_str(), d.n, d.dim,
                metric_name(d.metric));
    return d;
}

CoreSGParams base_params(const Options& o, const Data& d, std::uint64_t seed) {
    CoreSGParams p;
    p.k_max = o.k;
    p.metric = d.metric;
    p.missing_mode = MissingEdgeMode::AntiHub;
    p.threads = o.threads;
    p.seed = seed;
    p.backend.prefer_gpu = o.gpu;
    p.backend.oom_fallback = o.oom_fallback;
    p.incr.instances = 4;
    p.incr.min_base_for_split = 2000;
    p.incr.seed = seed;
    p.incr.instance_nnd.k = o.k;
    p.incr.instance_nnd.sample = 16;
    p.incr.instance_nnd.iters = 10;
    p.incr.instance_nnd.seed = seed;
    p.incr.refine_nnd.k = o.k;
    p.incr.refine_nnd.sample = 16;
    p.incr.refine_nnd.iters = 5;
    p.incr.refine_nnd.seed = seed * 7919u + 13u;
    return p;
}

/// A view of the first `n` rows, for building references over "the data that
/// exists at this point in the stream".
Dataset prefix_dataset(const Data& d, std::size_t n) {
    Dataset ds(d.dim, d.metric);
    ds.reserve(n);
    ds.append(d.values.data(), n);
    return ds;
}

/// Arrival order. Shuffled is the benign case; Grouped delivers the data
/// cluster by cluster, which is what a real stream often looks like and is the
/// adversarial case for any incremental method.
enum class Arrival { Shuffled, Grouped };

std::vector<idx_t> arrival_order(const Data& d, Arrival mode, std::uint64_t seed) {
    std::vector<idx_t> order(d.n);
    std::iota(order.begin(), order.end(), idx_t(0));
    std::mt19937_64 rng(seed);
    if (mode == Arrival::Shuffled) {
        std::shuffle(order.begin(), order.end(), rng);
        return order;
    }
    // Group by a coarse spatial key so that points arrive region by region,
    // without needing ground-truth labels: sort on the first principal-ish
    // axis (the widest coordinate), which clumps clusters together.
    std::uint32_t widest = 0;
    double best_spread = -1.0;
    for (std::uint32_t c = 0; c < d.dim; ++c) {
        double lo = 1e30, hi = -1e30;
        for (std::size_t i = 0; i < d.n; i += std::max<std::size_t>(1, d.n / 4096)) {
            const double v = d.values[i * d.dim + c];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        if (hi - lo > best_spread) {
            best_spread = hi - lo;
            widest = c;
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](idx_t a, idx_t b) {
        return d.values[std::size_t(a) * d.dim + widest] <
               d.values[std::size_t(b) * d.dim + widest];
    });
    return order;
}

std::vector<float> gather_rows(const Data& d, const std::vector<idx_t>& order, std::size_t lo,
                               std::size_t hi) {
    std::vector<float> out((hi - lo) * d.dim);
    for (std::size_t i = lo; i < hi; ++i) {
        std::memcpy(out.data() + (i - lo) * d.dim, d.values.data() + std::size_t(order[i]) * d.dim,
                    d.dim * sizeof(float));
    }
    return out;
}

// ---------------------------------------------------------------------------
// E1 -- incremental insertion vs full rebuild
// ---------------------------------------------------------------------------

void exp_incremental(const Options& o, const Data& d, Csv* csv, Arrival arrival) {
    const char* name = (arrival == Arrival::Shuffled) ? "incremental" : "arrival_order";
    const char* order_label = (arrival == Arrival::Shuffled) ? "shuffled" : "grouped";
    std::printf("\n[%s] %s arrival, %d seeds, %d batches\n", name, order_label, o.seeds,
                o.batches);

    const std::size_t base = static_cast<std::size_t>(o.base_fraction * double(d.n));
    const std::size_t remaining = d.n - base;
    const std::size_t per_batch = remaining / std::max(1, o.batches);
    if (per_batch == 0) {
        std::printf("  (too few points for %d batches; skipped)\n", o.batches);
        return;
    }
    const bool can_reference = d.n <= 200000;

    for (int s = 0; s < o.seeds; ++s) {
        const std::uint64_t seed = 1000 + s;
        const std::vector<idx_t> order = arrival_order(d, arrival, seed);

        CoreSGParams p = base_params(o, d, seed);
        CoreSG sg(p);

        std::vector<float> head = gather_rows(d, order, 0, base);
        CoreSGStats bs = sg.build(head.data(), base, d.dim);
        csv->row(name, d.name, base, d.dim, o.k, "arrival", order_label, seed, 0,
                 "build_seconds", bs.total_seconds);

        double cumulative = bs.total_seconds;
        std::size_t cursor = base;
        for (int b = 0; b < o.batches && cursor < d.n; ++b) {
            const std::size_t count = std::min(per_batch, d.n - cursor);
            std::vector<float> chunk = gather_rows(d, order, cursor, cursor + count);
            const CoreSGStats st = sg.insert(chunk.data(), count);
            cumulative += st.total_seconds;
            cursor += count;

            const long batch = b + 1;
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "insert_seconds", st.total_seconds);
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "cumulative_seconds", cumulative);
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "knn_seconds", st.knn_seconds);
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "instance_seconds", st.knn_stats.instance_seconds);
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "merge_seconds", st.knn_stats.merge_seconds);
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "refine_seconds", st.knn_stats.refine_seconds);
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "hubness_seconds", st.hubness_seconds);
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "refine_nodes_fraction",
                     double(st.knn_stats.refine_nodes) / double(st.n));
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "components_before_repair", double(st.components_before_repair));
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "antihubs_selected", double(st.antihubs_selected));
            csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                     "throughput_points_per_sec", double(count) / st.total_seconds);

            // Quality against the exact k-NN graph over the points that exist
            // *now* -- scoring an early batch against a reference built over
            // the whole file would count not-yet-inserted neighbours as misses.
            if (can_reference && (b == o.batches - 1 || b % 2 == 1)) {
                Dataset present = prefix_dataset(d, 0);
                present = Dataset(d.dim, d.metric);
                std::vector<float> sofar = gather_rows(d, order, 0, cursor);
                present.append(sofar.data(), cursor);
                KnnGraph exact = brute_force_graph(present, o.k, sg.pool());

                std::vector<idx_t> fresh;
                for (idx_t v = static_cast<idx_t>(cursor - count);
                     v < static_cast<idx_t>(cursor); ++v) {
                    fresh.push_back(v);
                }
                csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                         "recall_all", knn_recall(sg.knn(), exact, o.k).recall);
                csv->row(name, d.name, st.n, d.dim, o.k, "arrival", order_label, seed, batch,
                         "recall_new", knn_recall_subset(sg.knn(), exact, o.k, fresh).recall);
            }
        }

        // The baseline: rebuild the same final data set from scratch.
        std::vector<float> all = gather_rows(d, order, 0, cursor);
        CoreSG rebuilt(base_params(o, d, seed));
        const CoreSGStats rs = rebuilt.build(all.data(), cursor, d.dim);
        csv->row(name, d.name, cursor, d.dim, o.k, "arrival", order_label, seed, -1,
                 "rebuild_seconds", rs.total_seconds);
        csv->row(name, d.name, cursor, d.dim, o.k, "arrival", order_label, seed, -1,
                 "incremental_total_seconds", cumulative);

        if (can_reference) {
            Dataset present(d.dim, d.metric);
            present.append(all.data(), cursor);
            KnnGraph exact = brute_force_graph(present, o.k, sg.pool());
            csv->row(name, d.name, cursor, d.dim, o.k, "arrival", order_label, seed, -1,
                     "recall_incremental", knn_recall(sg.knn(), exact, o.k).recall);
            csv->row(name, d.name, cursor, d.dim, o.k, "arrival", order_label, seed, -1,
                     "recall_rebuild", knn_recall(rebuilt.knn(), exact, o.k).recall);

            // Do the two graphs induce the same clustering? Compare the MSTs
            // they yield, and the DBSCAN* labels cut from them.
            const std::uint32_t mpts = std::min<std::uint32_t>(o.k, 8);
            const MstResult ma = sg.mst(mpts);
            const MstResult mb = rebuilt.mst(mpts);
            const MstComparison c = compare_msts(ma, mb);
            csv->row(name, d.name, cursor, d.dim, o.k, "arrival", order_label, seed, -1,
                     "mst_weight_ratio_vs_rebuild", c.weight_ratio);
            double thresh = 0.0;
            for (const Edge& e : mb.edges) thresh += e.w;
            thresh /= double(mb.edges.empty() ? 1 : mb.edges.size());
            const std::vector<idx_t> la =
                dbscan_labels_from_mst(ma, static_cast<idx_t>(cursor), thresh, nullptr);
            const std::vector<idx_t> lb =
                dbscan_labels_from_mst(mb, static_cast<idx_t>(cursor), thresh, nullptr);
            csv->row(name, d.name, cursor, d.dim, o.k, "arrival", order_label, seed, -1,
                     "ari_vs_rebuild", adjusted_rand_index(la, lb));
        }
    }
}

// ---------------------------------------------------------------------------
// E2 -- anti-hub bridges vs the exact MST_kmax
// ---------------------------------------------------------------------------

void exp_bridges(const Options& o, const Data& d, Csv* csv) {
    const std::size_t n = std::min(d.n, o.exact_limit);
    std::printf("\n[bridges] n=%zu (capped for the O(n^2) reference), %d seeds\n", n, o.seeds);

    for (int s = 0; s < o.seeds; ++s) {
        const std::uint64_t seed = 2000 + s;

        // Both sides get an exact k-NN half so the only thing that varies is
        // the bridge estimate.
        CoreSGParams pa = base_params(o, d, seed);
        pa.incr.brute_force_limit = n;
        pa.missing_mode = MissingEdgeMode::AntiHub;
        CoreSG approx(pa);
        const CoreSGStats sa = approx.build(d.values.data(), n, d.dim);

        CoreSGParams pe = pa;
        pe.missing_mode = MissingEdgeMode::ExactMst;
        CoreSG exact(pe);
        const CoreSGStats se = exact.build(d.values.data(), n, d.dim);

        csv->row("bridges", d.name, n, d.dim, o.k, "mode", "antihub", seed, -1,
                 "bridge_seconds", sa.missing_seconds);
        csv->row("bridges", d.name, n, d.dim, o.k, "mode", "exact", seed, -1, "bridge_seconds",
                 se.missing_seconds);
        csv->row("bridges", d.name, n, d.dim, o.k, "mode", "antihub", seed, -1, "bridge_edges",
                 double(sa.missing_edges));
        csv->row("bridges", d.name, n, d.dim, o.k, "mode", "exact", seed, -1, "bridge_edges",
                 double(se.missing_edges));
        csv->row("bridges", d.name, n, d.dim, o.k, "mode", "antihub", seed, -1,
                 "antihubs_selected", double(sa.antihubs_selected));
        csv->row("bridges", d.name, n, d.dim, o.k, "mode", "antihub", seed, -1,
                 "components_before_repair", double(sa.components_before_repair));

        for (std::uint32_t mpts : {1u, 2u, 4u, 8u, 16u, o.k}) {
            if (mpts > o.k) continue;
            const MstResult ma = approx.mst(mpts);
            const MstResult me = exact.mst(mpts);
            const MstComparison c = compare_msts(ma, me);
            const std::string v = std::to_string(mpts);

            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1, "weight_ratio",
                     c.weight_ratio);
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "max_weight_profile_gap", c.max_edge_weight_gap);
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1, "components",
                     double(c.approx_components));
            // Recorded for completeness, but see the header: overlap measures
            // tie-breaking between equally optimal trees, not quality.
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "edge_jaccard_diagnostic_only", c.edge_jaccard);

            double thresh = 0.0;
            for (const Edge& e : me.edges) thresh += e.w;
            thresh /= double(me.edges.empty() ? 1 : me.edges.size());
            idx_t ka = 0, ke = 0;
            const std::vector<idx_t> la =
                dbscan_labels_from_mst(ma, static_cast<idx_t>(n), thresh, &ka);
            const std::vector<idx_t> le =
                dbscan_labels_from_mst(me, static_cast<idx_t>(n), thresh, &ke);
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1, "ari_vs_exact",
                     adjusted_rand_index(la, le));
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1, "clusters_approx",
                     double(ka));
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1, "clusters_exact",
                     double(ke));
        }

        // Ablation: does the bridge set matter at all?
        CoreSGParams pn = pa;
        pn.missing_mode = MissingEdgeMode::None;
        pn.ensure_connected = false;
        CoreSG bare(pn);
        const CoreSGStats sb = bare.build(d.values.data(), n, d.dim);
        csv->row("bridges", d.name, n, d.dim, o.k, "mode", "knn_only", seed, -1,
                 "components_before_repair", double(sb.components_before_repair));

        // How many anti-hubs are actually needed? This lives here, not in the
        // sensitivity sweep, because it only means anything when the k-NN half
        // is held exact on both sides -- otherwise NN-Descent's own variation
        // is larger than the effect being measured.
        const std::uint32_t probe_mpts = std::min<std::uint32_t>(o.k, 8);
        const MstResult reference = exact.mst(probe_mpts);
        for (std::size_t cap : {std::size_t(4), std::size_t(16), std::size_t(64),
                                std::size_t(256), std::size_t(1024)}) {
            CoreSGParams pc = pa;
            pc.hub.max_selected = cap;
            pc.hub.max_pair_ops = 0.0;  // the explicit cap is the variable here
            pc.ensure_connected = false;  // measure the raw bridge estimate
            CoreSG sub(pc);
            const CoreSGStats ss = sub.build(d.values.data(), n, d.dim);
            const MstComparison c = compare_msts(sub.mst(probe_mpts), reference);
            const std::string v = std::to_string(cap);
            csv->row("bridges", d.name, n, d.dim, o.k, "max_antihubs", v, seed, -1,
                     "weight_ratio", c.weight_ratio);
            csv->row("bridges", d.name, n, d.dim, o.k, "max_antihubs", v, seed, -1,
                     "components_before_repair", double(ss.components_before_repair));
            csv->row("bridges", d.name, n, d.dim, o.k, "max_antihubs", v, seed, -1,
                     "bridge_seconds", ss.missing_seconds);
            csv->row("bridges", d.name, n, d.dim, o.k, "max_antihubs", v, seed, -1,
                     "antihubs_selected", double(ss.antihubs_selected));
        }
    }
}

// ---------------------------------------------------------------------------
// E2b -- ambient vs intrinsic dimensionality in the anti-hub budget
//
// The budget is d*ln(|E|). Whether `d` should be the column count or the
// dimensionality of the manifold the data occupies is a design question, and
// this isolates it: both arms get an exact k-NN half, so the only thing that
// differs is the number of anti-hubs the budget asks for.
// ---------------------------------------------------------------------------

void exp_budget_dimension(const Options& o, const Data& d, Csv* csv) {
    const std::size_t n = std::min(d.n, o.exact_limit);
    std::printf("\n[budget_dim] n=%zu (exact k-NN half on both arms), %d seeds\n", n, o.seeds);

    for (int s = 0; s < o.seeds; ++s) {
        const std::uint64_t seed = 7000 + s;

        CoreSGParams pe = base_params(o, d, seed);
        pe.incr.brute_force_limit = n;
        pe.missing_mode = MissingEdgeMode::ExactMst;
        CoreSG reference(pe);
        const CoreSGStats sr = reference.build(d.values.data(), n, d.dim);
        csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", "exact_mst", seed, -1,
                 "bridge_seconds", sr.missing_seconds);

        const IntrinsicDimStats id = reference.intrinsic_dimension();
        csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", "exact_mst", seed, -1,
                 "intrinsic_dim", id.value());
        csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", "exact_mst", seed, -1,
                 "intrinsic_usable_fraction", id.usable_fraction());
        csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", "exact_mst", seed, -1,
                 "ambient_dim", double(d.dim));

        // The four candidate rules, all on the same exact k-NN half and the
        // same exact-MST reference, so the only thing that varies is how many
        // anti-hubs the budget asks for. Naming them here rather than sweeping
        // parameters independently keeps the output directly readable as
        // "which rule should I adopt".
        struct BudgetOption {
            const char* name;
            BudgetDimension dim;
            double multiplier;
            double max_pair_ops;  // 0 disables the cost cap
        };
        const BudgetOption options[] = {
            // A: the specified rule, uncapped. Degenerates in high dimensions.
            {"ambient", BudgetDimension::Ambient, 1.0, 0.0},
            // D: the specified rule plus a cost backstop. Treats the symptom.
            {"ambient_capped", BudgetDimension::Ambient, 1.0, 2e9},
            // B: intrinsic dimensionality. Fixes the cause, may under-provision.
            {"intrinsic", BudgetDimension::Intrinsic, 1.0, 0.0},
            // C: intrinsic scaling with headroom, since cost is nearly flat up
            // to a few hundred anti-hubs.
            {"intrinsic_x2", BudgetDimension::Intrinsic, 2.0, 0.0},
            {"intrinsic_x4", BudgetDimension::Intrinsic, 4.0, 0.0},
        };

        for (const BudgetOption& opt : options) {
            CoreSGParams p = base_params(o, d, seed);
            p.incr.brute_force_limit = n;
            p.missing_mode = MissingEdgeMode::AntiHub;
            p.budget_dimension = opt.dim;
            p.hub.antihub_multiplier = opt.multiplier;
            p.hub.max_pair_ops = opt.max_pair_ops;
            CoreSG sg(p);
            const CoreSGStats st = sg.build(d.values.data(), n, d.dim);
            const std::string v = opt.name;

            csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", v, seed, -1,
                     "effective_dim", st.effective_dim);
            csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", v, seed, -1,
                     "antihubs_selected", double(st.antihubs_selected));
            csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", v, seed, -1,
                     "antihub_fraction_of_n", double(st.antihubs_selected) / double(n));
            csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", v, seed, -1,
                     "bridge_seconds", st.missing_seconds);
            csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", v, seed, -1,
                     "components_before_repair", double(st.components_before_repair));
            csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", v, seed, -1,
                     "total_seconds", st.total_seconds);
            // What the rule asked for before any cap, and whether a cap bound
            // it -- the pair that distinguishes "the formula is well scaled"
            // from "the backstop rescued it".
            csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", v, seed, -1,
                     "budget_uncapped", double(st.hub_stats.budget_uncapped));
            csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", v, seed, -1,
                     "budget_capped", st.hub_stats.budget_capped ? 1.0 : 0.0);
            csv->row("budget_dim", d.name, n, d.dim, o.k, "mode", v, seed, -1,
                     "quota_selected", double(st.hub_stats.quota_selected));

            for (std::uint32_t mpts : {2u, 8u, o.k}) {
                if (mpts > o.k) continue;
                const MstComparison c =
                    compare_msts(sg.mst(mpts), reference.mst(mpts));
                const std::string mv = v + "@mpts" + std::to_string(mpts);
                csv->row("budget_dim", d.name, n, d.dim, o.k, "mode_mpts", mv, seed, -1,
                         "weight_ratio", c.weight_ratio);
                csv->row("budget_dim", d.name, n, d.dim, o.k, "mode_mpts", mv, seed, -1,
                         "max_weight_profile_gap", c.max_edge_weight_gap);

                double thresh = 0.0;
                const MstResult ref_mst = reference.mst(mpts);
                for (const Edge& e : ref_mst.edges) thresh += e.w;
                thresh /= double(ref_mst.edges.empty() ? 1 : ref_mst.edges.size());
                const std::vector<idx_t> la = dbscan_labels_from_mst(
                    sg.mst(mpts), static_cast<idx_t>(n), thresh, nullptr);
                const std::vector<idx_t> le = dbscan_labels_from_mst(
                    ref_mst, static_cast<idx_t>(n), thresh, nullptr);
                csv->row("budget_dim", d.name, n, d.dim, o.k, "mode_mpts", mv, seed, -1,
                         "ari_vs_exact", adjusted_rand_index(la, le));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// E3 -- the mpts sweep, which is what CORE-SG exists for
// ---------------------------------------------------------------------------

void exp_mpts(const Options& o, const Data& d, Csv* csv) {
    const std::size_t n = std::min(d.n, o.exact_limit * 4);
    std::printf("\n[mpts] n=%zu, sweeping mpts in [1, %u]\n", n, o.k);

    for (int s = 0; s < o.seeds; ++s) {
        const std::uint64_t seed = 3000 + s;
        CoreSG sg(base_params(o, d, seed));
        const CoreSGStats st = sg.build(d.values.data(), n, d.dim);
        csv->row("mpts_sweep", d.name, n, d.dim, o.k, "phase", "build", seed, -1, "seconds",
                 st.total_seconds);

        std::vector<std::uint32_t> values;
        for (std::uint32_t m = 1; m <= o.k; ++m) values.push_back(m);

        const auto t0 = Clock::now();
        for (std::uint32_t m : values) sg.mst(m);
        const double sweep = since(t0);
        csv->row("mpts_sweep", d.name, n, d.dim, o.k, "phase", "coresg_sweep", seed, -1,
                 "seconds", sweep);

        // Extrapolate the naive alternative from a few exact runs; a full
        // sweep over the complete graph is not affordable at this n.
        const std::size_t probe = std::min<std::size_t>(values.size(), 3);
        const auto t1 = Clock::now();
        for (std::size_t i = 0; i < probe; ++i) sg.exact_mst(values[i]);
        const double naive =
            since(t1) * double(values.size()) / double(probe ? probe : 1);
        csv->row("mpts_sweep", d.name, n, d.dim, o.k, "phase", "complete_graph_sweep", seed, -1,
                 "seconds_extrapolated", naive);
        csv->row("mpts_sweep", d.name, n, d.dim, o.k, "phase", "speedup", seed, -1, "ratio",
                 naive / (sweep + 1e-12));
    }
}

// ---------------------------------------------------------------------------
// E4 -- scaling. Threads on CPU, instances and devices on GPU.
// ---------------------------------------------------------------------------

void exp_scaling(const Options& o, const Data& d, Csv* csv) {
    std::printf("\n[scaling] devices visible: %zu\n", enumerate_cuda_devices().size());
    const std::size_t base = static_cast<std::size_t>(o.base_fraction * double(d.n));
    const std::size_t batch = (d.n - base) / 4;
    if (batch == 0) {
        std::printf("  (data set too small; skipped)\n");
        return;
    }

    // CPU thread scaling.
    for (unsigned threads : {1u, 2u, 4u, 8u, 0u}) {
        for (int s = 0; s < std::min(o.seeds, 3); ++s) {
            const std::uint64_t seed = 4000 + s;
            Options local = o;
            local.gpu = false;
            local.threads = threads;
            CoreSGParams p = base_params(local, d, seed);
            p.incr.instances = 4;
            CoreSG sg(p);
            const CoreSGStats bs = sg.build(d.values.data(), base, d.dim);
            const CoreSGStats us = sg.insert(d.values.data() + base * d.dim, batch);
            const std::string v = threads ? std::to_string(threads) : "all";
            csv->row("scaling", d.name, d.n, d.dim, o.k, "cpu_threads", v, seed, -1,
                     "build_seconds", bs.total_seconds);
            csv->row("scaling", d.name, d.n, d.dim, o.k, "cpu_threads", v, seed, -1,
                     "insert_seconds", us.total_seconds);
            csv->row("scaling", d.name, d.n, d.dim, o.k, "cpu_threads", v, seed, -1,
                     "instance_seconds", us.knn_stats.instance_seconds);
        }
    }

    const std::vector<DeviceInfo> devices = enumerate_cuda_devices();
    if (devices.empty()) {
        std::printf("  (no CUDA devices -- GPU rows omitted; rerun on the cluster)\n");
        return;
    }

    // Device-count scaling. Instances are pinned to a multiple of the device
    // count so the queue balances.
    // Instance counts are FIXED across device counts, not scaled with them.
    // Tying instances to the device count changes the amount of work as devices
    // are added, so the resulting numbers are not a strong-scaling curve and can
    // read as a slowdown. With a fixed count, `1 GPU x 8` vs `4 GPUs x 8` is a
    // real speed-up measurement. Counts are multiples of the largest device
    // count so the queue divides evenly.
    const std::uint32_t nd = static_cast<std::uint32_t>(devices.size());
    const std::vector<std::uint32_t> instance_counts =
        o.scaling_instances.empty() ? std::vector<std::uint32_t>{nd, nd * 2u, nd * 4u}
                                    : parse_uint_list(o.scaling_instances);
    for (std::size_t ngpu = 1; ngpu <= devices.size(); ++ngpu) {
        for (std::uint32_t instances : instance_counts) {
            for (int s = 0; s < std::min(o.seeds, 3); ++s) {
                const std::uint64_t seed = 5000 + s;
                CoreSGParams p = base_params(o, d, seed);
                p.backend.prefer_gpu = true;
                p.backend.device_ids.clear();
                for (std::size_t g = 0; g < ngpu; ++g) {
                    p.backend.device_ids.push_back(devices[g].id);
                }
                p.incr.instances = instances;
                CoreSG sg(p);
                // Warm-up: CUDA context creation costs 100-300 ms per device
                // and would otherwise dominate a sub-second update.
                sg.build(d.values.data(), base, d.dim);
                const CoreSGStats us = sg.insert(d.values.data() + base * d.dim, batch);

                const std::string v = std::to_string(ngpu);
                const std::string inst = std::to_string(instances);
                csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                         v + "x" + inst, seed, -1, "insert_seconds", us.total_seconds);
                csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                         v + "x" + inst, seed, -1, "instance_seconds",
                         us.knn_stats.instance_seconds);
                csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                         v + "x" + inst, seed, -1, "merge_seconds",
                         us.knn_stats.merge_seconds);
                csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                         v + "x" + inst, seed, -1, "refine_seconds",
                         us.knn_stats.refine_seconds);
                const double phase = us.knn_stats.instance_seconds +
                                     us.knn_stats.merge_seconds + us.knn_stats.refine_seconds;
                if (phase > 0.0) {
                    csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                             v + "x" + inst, seed, -1, "parallel_fraction",
                             us.knn_stats.instance_seconds / phase);
                }

                // Memory is the load-bearing part of the scaling claim: the
                // per-device footprint has to stay flat as devices are added,
                // because an instance only ever holds |sample| + |batch|
                // vectors rather than the whole data set.
                // The capacity claim lives or dies on these two: how many
                // instances were asked for, and whether the device refused.
                csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                         v + "x" + inst, seed, -1, "instances", double(instances));
                csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                         v + "x" + inst, seed, -1, "jobs_out_of_memory",
                         double(us.knn_stats.jobs_out_of_memory));
                csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                         v + "x" + inst, seed, -1, "largest_instance_nodes",
                         double(us.knn_stats.largest_instance));
                csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                         v + "x" + inst, seed, -1, "largest_instance_bytes",
                         double(us.knn_stats.largest_instance_bytes));
                csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                         v + "x" + inst, seed, -1, "peak_host_bytes",
                         double(us.peak_host_bytes));
                for (const DeviceMemory& dm : device_memory()) {
                    csv->row("scaling", d.name, d.n, d.dim, o.k, "gpus_x_instances",
                             v + "x" + inst, seed, dm.id, "device_used_bytes",
                             double(dm.used_bytes));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// E6 -- long-horizon drift
//
// Separate from `incremental` because it asks a different question and needs a
// different shape. `incremental` measures the cost and quality of a handful of
// large batches; this one runs many small batches and samples quality at
// *every* one, because the failure mode it is looking for -- slow degradation
// over a long stream -- is invisible when you only check the endpoints. The
// data set is capped so the per-batch exact reference stays affordable.
// ---------------------------------------------------------------------------

void exp_drift(const Options& o, const Data& d, Csv* csv) {
    const std::size_t n = std::min(d.n, std::size_t(30000));
    const int batches = std::max(o.batches, 20);
    const std::size_t base = n / 4;
    const std::size_t per_batch = (n - base) / batches;
    std::printf("\n[drift] n=%zu, base=%zu, %d batches of %zu, quality every batch\n", n, base,
                batches, per_batch);
    if (per_batch == 0) {
        std::printf("  (too few points; skipped)\n");
        return;
    }

    for (int s = 0; s < std::min(o.seeds, 3); ++s) {
        const std::uint64_t seed = 8000 + s;
        CoreSG sg(base_params(o, d, seed));
        sg.build(d.values.data(), base, d.dim);

        std::size_t cursor = base;
        double first_recall = -1.0;
        for (int b = 0; b < batches && cursor < n; ++b) {
            const std::size_t count = std::min(per_batch, n - cursor);
            const CoreSGStats st = sg.insert(d.values.data() + cursor * d.dim, count);
            cursor += count;

            Dataset present(d.dim, d.metric);
            present.append(d.values.data(), cursor);
            KnnGraph exact = brute_force_graph(present, o.k, sg.pool());
            const RecallResult all_r = knn_recall(sg.knn(), exact, o.k);

            std::vector<idx_t> fresh;
            for (idx_t v = static_cast<idx_t>(cursor - count);
                 v < static_cast<idx_t>(cursor); ++v) {
                fresh.push_back(v);
            }
            const RecallResult new_r = knn_recall_subset(sg.knn(), exact, o.k, fresh);

            const long batch = b + 1;
            csv->row("drift", d.name, st.n, d.dim, o.k, "batch_index",
                     std::to_string(batch), seed, batch, "recall_all", all_r.recall);
            csv->row("drift", d.name, st.n, d.dim, o.k, "batch_index",
                     std::to_string(batch), seed, batch, "recall_new", new_r.recall);
            csv->row("drift", d.name, st.n, d.dim, o.k, "batch_index",
                     std::to_string(batch), seed, batch, "insert_seconds", st.total_seconds);
            csv->row("drift", d.name, st.n, d.dim, o.k, "batch_index",
                     std::to_string(batch), seed, batch, "components_before_repair",
                     double(st.components_before_repair));
            csv->row("drift", d.name, st.n, d.dim, o.k, "batch_index",
                     std::to_string(batch), seed, batch, "antihubs_selected",
                     double(st.antihubs_selected));
            csv->row("drift", d.name, st.n, d.dim, o.k, "batch_index",
                     std::to_string(batch), seed, batch, "peak_host_bytes",
                     double(st.peak_host_bytes));
            // The drift metric proper: how far quality has moved from where it
            // started. Flat means the graph is not rotting.
            if (first_recall < 0.0) first_recall = all_r.recall;
            csv->row("drift", d.name, st.n, d.dim, o.k, "batch_index",
                     std::to_string(batch), seed, batch, "recall_delta_vs_first",
                     all_r.recall - first_recall);
        }
    }
}

// ---------------------------------------------------------------------------
// E5 -- sensitivity to the parameters a reader will ask about
// ---------------------------------------------------------------------------

void exp_sensitivity(const Options& o, const Data& d, Csv* csv) {
    const std::size_t n = std::min(d.n, std::size_t(50000));
    std::printf("\n[sensitivity] n=%zu\n", n);
    const bool can_reference = n <= 200000;

    Dataset ref(d.dim, d.metric);
    ref.append(d.values.data(), n);

    auto measure = [&](const std::string& config, const std::string& value,
                       const CoreSGParams& p, std::uint64_t seed) {
        CoreSG sg(p);
        const std::size_t base = n / 2;
        const CoreSGStats bs = sg.build(d.values.data(), base, d.dim);
        const CoreSGStats us = sg.insert(d.values.data() + base * d.dim, n - base);
        csv->row("sensitivity", d.name, n, d.dim, p.k_max, config, value, seed, -1,
                 "build_seconds", bs.total_seconds);
        csv->row("sensitivity", d.name, n, d.dim, p.k_max, config, value, seed, -1,
                 "insert_seconds", us.total_seconds);
        csv->row("sensitivity", d.name, n, d.dim, p.k_max, config, value, seed, -1,
                 "antihubs_selected", double(us.antihubs_selected));
        csv->row("sensitivity", d.name, n, d.dim, p.k_max, config, value, seed, -1,
                 "components_before_repair", double(us.components_before_repair));
        csv->row("sensitivity", d.name, n, d.dim, p.k_max, config, value, seed, -1,
                 "bridge_seconds", us.missing_seconds);
        // No MST-quality ratio here on purpose. Two separately built CoreSGs
        // get two different (randomised, race-tolerant) k-NN halves, and that
        // variation is larger than the bridge effect being probed -- it shows
        // up as a ratio below 1.0, which is impossible for a real
        // approximation. Bridge quality is measured in exp_bridges instead,
        // where both sides share an exact k-NN half.
        if (can_reference) {
            KnnGraph exact = brute_force_graph(ref, p.k_max, sg.pool());
            csv->row("sensitivity", d.name, n, d.dim, p.k_max, config, value, seed, -1,
                     "recall", knn_recall(sg.knn(), exact, p.k_max).recall);
        }
    };

    for (int s = 0; s < std::min(o.seeds, 3); ++s) {
        const std::uint64_t seed = 6000 + s;
        for (std::uint32_t k : {8u, 16u, 32u, 64u}) {
            Options local = o;
            local.k = k;
            CoreSGParams p = base_params(local, d, seed);
            measure("k_max", std::to_string(k), p, seed);
        }
        for (std::uint32_t inst : {1u, 2u, 4u, 8u, 16u}) {
            CoreSGParams p = base_params(o, d, seed);
            p.incr.instances = inst;
            measure("instances", std::to_string(inst), p, seed);
        }
        for (double mult : {0.5, 1.0, 2.0, 4.0}) {
            CoreSGParams p = base_params(o, d, seed);
            p.hub.antihub_multiplier = mult;
            measure("antihub_multiplier", std::to_string(mult), p, seed);
        }
        for (std::size_t quota : {std::size_t(0), std::size_t(1), std::size_t(2),
                                  std::size_t(4)}) {
            CoreSGParams p = base_params(o, d, seed);
            p.hub.min_per_component = quota;
            measure("min_per_component", std::to_string(quota), p, seed);
        }
        for (std::uint32_t smp : {8u, 16u, 24u, 32u}) {
            CoreSGParams p = base_params(o, d, seed);
            p.incr.instance_nnd.sample = smp;
            p.incr.refine_nnd.sample = smp;
            measure("sample_size", std::to_string(smp), p, seed);
        }
    }
}

// ---------------------------------------------------------------------------

void usage() {
    std::printf(
        "coresg_exp -- thesis experiment harness (long-format CSV)\n\n"
        "  --experiment  incremental|arrival|bridges|budget|mpts|scaling|sensitivity|all\n"
        "  --data FILE [--dim D] [--limit N] [--name LABEL]   or  --synthetic N\n"
        "  --dim D --clusters C     (synthetic only)\n"
        "  --k K --metric l2|cosine|l1\n"
        "  --seeds S --batches B --base-fraction F\n"
        "  --exact-limit N          ceiling for the O(n^2) references\n"
        "  --gpu 0|1 --threads T\n"
        "  --out results.csv --quiet\n");
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--experiment") o.experiment = next();
        else if (a == "--data") o.data_path = next();
        else if (a == "--name") o.dataset_name = next();
        else if (a == "--dim") o.dim = o.raw_dim = std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--limit") o.limit = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--synthetic") o.synthetic = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--clusters") o.clusters = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--k") o.k = std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--metric") o.metric = next();
        else if (a == "--seeds") o.seeds = std::atoi(next().c_str());
        else if (a == "--batches") o.batches = std::atoi(next().c_str());
        else if (a == "--base-fraction") o.base_fraction = std::atof(next().c_str());
        else if (a == "--exact-limit") o.exact_limit = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--gpu") o.gpu = std::atoi(next().c_str()) != 0;
        else if (a == "--scaling-instances") o.scaling_instances = next();
        else if (a == "--no-oom-fallback") o.oom_fallback = false;
        else if (a == "--threads") o.threads = std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--out") o.out = next();
        else if (a == "--quiet") o.echo = false;
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
    }

    try {
        const Data d = load(o);
        Csv csv(o.out, o.echo);
        const std::string e = o.experiment;
        const bool all = (e == "all");

        if (all || e == "incremental") exp_incremental(o, d, &csv, Arrival::Shuffled);
        if (all || e == "arrival") exp_incremental(o, d, &csv, Arrival::Grouped);
        if (all || e == "bridges") exp_bridges(o, d, &csv);
        if (all || e == "budget") exp_budget_dimension(o, d, &csv);
        if (all || e == "mpts") exp_mpts(o, d, &csv);
        if (all || e == "scaling") exp_scaling(o, d, &csv);
        if (all || e == "drift") exp_drift(o, d, &csv);
        if (all || e == "sensitivity") exp_sensitivity(o, d, &csv);

        std::printf("\nwrote %s\n", o.out.c_str());
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
