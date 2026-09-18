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
#include <iomanip>
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
        // A stream's default is 6 significant digits, which rounds a weight
        // ratio of 1.0000004 to "1" and makes a near-tie read as an identity.
        // Ten digits distinguishes them and costs a few bytes per row.
        out_ << std::setprecision(10);
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
    /// Skip the CPU thread sweep in `scaling`. It is unaffordable on a very
    /// large data set -- 4.3 hours on 10M points before the GPU section even
    /// starts -- and irrelevant when the run exists to probe device capacity.
    bool skip_cpu_scaling = false;
    /// Comma-separated k values for the `connectivity` sweep. Empty means the
    /// built-in {4,8,16,24,32,48,64}.
    std::string k_sweep;
    /// Comma-separated batch sizes, as fractions of n, for `batchsize`.
    /// Empty means the built-in {0.001, 0.005, 0.01, 0.05, 0.10, 0.25}.
    std::string batch_fractions;
    /// `stream` only: arrival order supplied by the driver script, as a raw
    /// array of n int32 indices. Both methods have to see the same points at
    /// the same batch or the per-batch ARIs do not join.
    std::string order_file;
    /// `stream` only: directory the per-batch MSTs are written to.
    std::string mst_dir;
    /// `stream` only: mpts the exported MST is built for. Must equal the
    /// `min_samples` of the reference clustering the Python side computes.
    std::uint32_t export_mpts = 10;
    /// First seed value. `stream` needs it to match the driver script's own
    /// numbering so rows from the two sides join on (dataset, seed, batch).
    std::uint64_t seed_base = 0;
    /// Which insertion strategy the run measures. `incremental` sweeps both
    /// regardless, so that the O(batch) claim is a measured comparison rather
    /// than a change of default nobody re-ran the numbers for.
    InsertStrategy strategy = InsertStrategy::Anchored;
    /// Anchored overrides, 0 = leave the default. These exist because the
    /// `sensitivity` sweep inserts the whole second half as ONE batch, where
    /// recall saturates at 0.998 and every knob looks inert -- while the loss
    /// they control is cumulative and only appears over many batches. Exposing
    /// them here lets any multi-batch experiment be run at a given setting.
    std::uint32_t search_ef = 0;
    double context_per_new = 0.0;
    double refine_per_new = 0.0;
    /// CUDA device ordinals to confine the run to. Empty means every visible
    /// device, which is the right default for a sweep and the wrong one for a
    /// head-to-head against a single-device baseline: cuML's HDBSCAN* occupies
    /// one GPU, so a comparison that silently uses four is not a per-device
    /// figure. `scaling` sets its own device list and ignores this.
    std::string device_ids;
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

std::vector<double> parse_double_list(const std::string& s) {
    std::vector<double> out;
    std::size_t start = 0;
    while (start <= s.size()) {
        const std::size_t comma = s.find(',', start);
        const std::string tok = s.substr(start, comma - start);
        if (!tok.empty()) out.push_back(std::stod(tok));
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
    for (std::uint32_t id : parse_uint_list(o.device_ids)) {
        p.backend.device_ids.push_back(static_cast<int>(id));
    }
    p.backend.oom_fallback = o.oom_fallback;
    p.incr.instances = 4;
    p.incr.min_base_for_split = 2000;
    p.incr.strategy = o.strategy;
    if (o.search_ef) p.incr.search_ef = o.search_ef;
    if (o.context_per_new > 0.0) p.incr.context_per_new = o.context_per_new;
    if (o.refine_per_new > 0.0) p.incr.refine_per_new = o.refine_per_new;
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

    // Phase breakdown of a *monolithic* build, emitted for both the initial
    // build and the final rebuild. Without it the only thing recorded about a
    // build is its total, which cannot answer the question the construction
    // claim rests on: how much of a build is device parallel. An update's
    // breakdown has been emitted per batch since the start; a build's has not,
    // so the two arms could not be compared on the one axis that matters.
    auto emit_build_phases = [&](const char* prefix, std::size_t n_at, long batch_id,
                                 std::uint64_t seed, const CoreSGStats& st) {
        const std::string pre(prefix);
        auto put = [&](const char* suffix, double value) {
            csv->row(name, d.name, n_at, d.dim, o.k, "arrival", order_label, seed, batch_id,
                     pre + suffix, value);
        };
        put("_knn_seconds", st.knn_seconds);
        put("_instance_seconds", st.knn_stats.instance_seconds);
        put("_stage_seconds", st.knn_stats.stage_seconds);
        put("_merge_seconds", st.knn_stats.merge_seconds);
        put("_refine_seconds", st.knn_stats.refine_seconds);
        put("_hubness_seconds", st.hubness_seconds);
        put("_missing_seconds", st.missing_seconds);
        put("_repair_seconds", st.repair_seconds);
        put("_instances", double(st.knn_stats.instances));
        put("_jobs_on_device", double(st.knn_stats.jobs_on_device));
        put("_jobs_on_host", double(st.knn_stats.jobs_on_host));
        put("_jobs_out_of_memory", double(st.knn_stats.jobs_out_of_memory));
        put("_largest_instance_nodes", double(st.knn_stats.largest_instance));
        put("_largest_instance_bytes", double(st.largest_instance_bytes));
        put("_peak_host_bytes", double(st.peak_host_bytes));
        // The parallel fraction of this build, on the definition the scaling
        // experiment already uses: the share of wall clock spent in the one
        // phase a device queue divides.
        if (st.total_seconds > 0.0) {
            put("_parallel_fraction", st.knn_stats.instance_seconds / st.total_seconds);
        }
    };

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
        emit_build_phases("build", base, 0, seed, bs);

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
        emit_build_phases("rebuild", cursor, -1, seed, rs);

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

            // The equal-work control, and the reason it is needed: the
            // incremental path performs strictly *more* total descent than one
            // rebuild. Each batch re-visits sampled base points inside its
            // instances, so after `batches` updates the base has been refined
            // `batches` times while the rebuild descends once. A plain rebuild
            // is therefore a cheaper baseline, not an equal one, and reporting
            // "incremental wins on recall" against it invites the obvious
            // reply that it simply did more work.
            //
            // Rather than hand-tune one matched configuration, sweep the
            // rebuild's descent effort and emit the whole (cost, recall)
            // frontier. The incremental point can then be read against the
            // rebuild curve *at equal cost*, which is the comparison a reader
            // actually wants -- and if the frontier is flat because the
            // descent has already converged, that is itself the answer, so
            // `refine_rounds` is recorded to tell the two cases apart.
            //
            // Only the first two seeds run it: this is four extra rebuilds per
            // seed and the frontier is a trend, not a per-seed quantity.
            if (s < 2) {
                struct RebuildEffort {
                    const char* name;
                    std::uint32_t iters_mult;   // more rounds, subject to `delta` early stop
                    std::uint32_t sample_mult;  // strictly more candidate pairs per round
                };
                const RebuildEffort efforts[] = {
                    {"iters_x2", 2, 1},
                    {"iters_x4", 4, 1},
                    {"sample_x2", 1, 2},
                    {"iters_x2_sample_x2", 2, 2},
                };
                for (const RebuildEffort& eff : efforts) {
                    CoreSGParams ph = base_params(o, d, seed);
                    ph.incr.instance_nnd.iters *= eff.iters_mult;
                    ph.incr.refine_nnd.iters *= eff.iters_mult;
                    ph.incr.instance_nnd.sample *= eff.sample_mult;
                    ph.incr.refine_nnd.sample *= eff.sample_mult;
                    CoreSG heavy(ph);
                    const CoreSGStats hs = heavy.build(all.data(), cursor, d.dim);
                    csv->row(name, d.name, cursor, d.dim, o.k, "rebuild_effort", eff.name, seed,
                             -1, "rebuild_seconds", hs.total_seconds);
                    csv->row(name, d.name, cursor, d.dim, o.k, "rebuild_effort", eff.name, seed,
                             -1, "recall_rebuild", knn_recall(heavy.knn(), exact, o.k).recall);
                    csv->row(name, d.name, cursor, d.dim, o.k, "rebuild_effort", eff.name, seed,
                             -1, "refine_rounds", double(hs.knn_stats.refine_stats.rounds));
                    csv->row(name, d.name, cursor, d.dim, o.k, "rebuild_effort", eff.name, seed,
                             -1, "distance_evals", double(hs.knn_stats.refine_stats.distance_evals));
                }
                // The two points the frontier is read against, repeated under
                // the same config name so a single group-by produces the plot.
                csv->row(name, d.name, cursor, d.dim, o.k, "rebuild_effort", "baseline", seed, -1,
                         "rebuild_seconds", rs.total_seconds);
                csv->row(name, d.name, cursor, d.dim, o.k, "rebuild_effort", "baseline", seed, -1,
                         "recall_rebuild", knn_recall(rebuilt.knn(), exact, o.k).recall);
                csv->row(name, d.name, cursor, d.dim, o.k, "rebuild_effort", "incremental", seed,
                         -1, "rebuild_seconds", cumulative);
                csv->row(name, d.name, cursor, d.dim, o.k, "rebuild_effort", "incremental", seed,
                         -1, "recall_rebuild", knn_recall(sg.knn(), exact, o.k).recall);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// E1b -- insertion strategy: does an update cost O(batch) or O(n)?
// ---------------------------------------------------------------------------

/// The two strategies against one rebuild baseline, with the batch size held
/// fixed so that per-batch cost can be read as a function of n alone.
///
/// This exists because the original claim -- "cheaper than rebuilding" -- was
/// only ever true per batch and only on a GPU. Under Resample the union of the
/// instances is the whole base, so an update is Theta(n_base) and the margin
/// shrinks as the stream lengthens; on a CPU backend it inverts outright and a
/// rebuild wins. The number that settles it is `instance_nodes` against n: a
/// flat line is the property, a sloped one is the bug.
void exp_strategy(const Options& o, const Data& d, Csv* csv) {
    std::printf("\n[strategy] %d seeds, %d batches\n", o.seeds, o.batches);
    const std::size_t base = static_cast<std::size_t>(o.base_fraction * double(d.n));
    const std::size_t per_batch = (d.n - base) / std::max(1, o.batches);
    if (per_batch == 0) {
        std::printf("  (too few points for %d batches; skipped)\n", o.batches);
        return;
    }
    const bool can_reference = d.n <= 200000;

    for (int s = 0; s < o.seeds; ++s) {
        const std::uint64_t seed = 1000 + s;
        const std::vector<idx_t> order = arrival_order(d, Arrival::Shuffled, seed);
        const std::size_t total = base + std::size_t(o.batches) * per_batch;
        const std::vector<float> all = gather_rows(d, order, 0, total);

        // The baseline is a rebuild *at every batch's own n*, not one rebuild
        // at the final size. Dividing an update at n=150k by a rebuild at
        // n=200k flatters the early batches and makes the ratio look like it
        // decays even when it does not -- the number a reader wants is "if I
        // had rebuilt instead of updating, right now, what would it have
        // cost". Batches share the arms, so this runs once per seed.
        CoreSGParams rp = base_params(o, d, seed);
        std::vector<double> rebuild_at(o.batches, 0.0);
        for (int b = 0; b < o.batches; ++b) {
            const std::size_t at = base + std::size_t(b + 1) * per_batch;
            CoreSG r(rp);
            const CoreSGStats s2 = r.build(all.data(), at, d.dim);
            rebuild_at[b] = s2.total_seconds;
            csv->row("strategy", d.name, at, d.dim, o.k, "strategy", "rebuild", seed, b + 1,
                     "rebuild_seconds", s2.total_seconds);
        }

        CoreSG rebuilt(rp);
        const CoreSGStats rs = rebuilt.build(all.data(), total, d.dim);
        csv->row("strategy", d.name, total, d.dim, o.k, "strategy", "rebuild", seed, -1,
                 "rebuild_seconds", rs.total_seconds);
        // What a rebuild needs resident on one device, so the memory claim sits
        // next to the time claim instead of in a separate experiment.
        csv->row("strategy", d.name, total, d.dim, o.k, "strategy", "rebuild", seed, -1,
                 "largest_instance_bytes", double(rs.largest_instance_bytes));

        KnnGraph exact;
        if (can_reference) {
            Dataset present(d.dim, d.metric);
            present.append(all.data(), total);
            exact = brute_force_graph(present, o.k, rebuilt.pool());
            csv->row("strategy", d.name, total, d.dim, o.k, "strategy", "rebuild", seed, -1,
                     "recall", knn_recall(rebuilt.knn(), exact, o.k).recall);
        }

        for (InsertStrategy strat : {InsertStrategy::Resample, InsertStrategy::Anchored}) {
            const char* label = to_string(strat);
            CoreSGParams p = base_params(o, d, seed);
            p.incr.strategy = strat;
            CoreSG sg(p);
            sg.build(all.data(), base, d.dim);

            double updates = 0.0;
            std::size_t cursor = base;
            for (int b = 0; b < o.batches && cursor < total; ++b) {
                const std::size_t count = std::min(per_batch, total - cursor);
                const CoreSGStats st = sg.insert(all.data() + cursor * d.dim, count);
                cursor += count;
                updates += st.total_seconds;

                const long batch = b + 1;
                csv->row("strategy", d.name, st.n, d.dim, o.k, "strategy", label, seed, batch,
                         "insert_seconds", st.total_seconds);
                // The load-bearing column: nodes the instances descended over.
                // Flat in n is the whole claim.
                csv->row("strategy", d.name, st.n, d.dim, o.k, "strategy", label, seed, batch,
                         "instance_nodes", double(st.knn_stats.instance_nodes));
                csv->row("strategy", d.name, st.n, d.dim, o.k, "strategy", label, seed, batch,
                         "search_seconds", st.knn_stats.search_seconds);
                csv->row("strategy", d.name, st.n, d.dim, o.k, "strategy", label, seed, batch,
                         "context_nodes", double(st.knn_stats.context_nodes));
                csv->row("strategy", d.name, st.n, d.dim, o.k, "strategy", label, seed, batch,
                         "refine_nodes", double(st.knn_stats.refine_nodes));
                // The capacity claim: what one device had to hold for this
                // update. Under Anchored it is a function of the batch, so it
                // should be flat while n grows.
                csv->row("strategy", d.name, st.n, d.dim, o.k, "strategy", label, seed, batch,
                         "largest_instance_bytes", double(st.largest_instance_bytes));
                // Speed-up against rebuilding at *this* n. This is the headline
                // curve: flat update over linear rebuild means it climbs.
                csv->row("strategy", d.name, st.n, d.dim, o.k, "strategy", label, seed, batch,
                         "rebuild_speedup", rebuild_at[b] / st.total_seconds);
            }
            csv->row("strategy", d.name, cursor, d.dim, o.k, "strategy", label, seed, -1,
                     "update_total_seconds", updates);
            if (can_reference) {
                csv->row("strategy", d.name, cursor, d.dim, o.k, "strategy", label, seed, -1,
                         "recall", knn_recall(sg.knn(), exact, o.k).recall);
            }
            std::printf("  seed %llu %-9s: %d updates %.3fs (%.3fs each) vs rebuild %.3fs\n",
                        static_cast<unsigned long long>(seed), label, o.batches, updates,
                        updates / double(o.batches), rs.total_seconds);
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

        // The ablation arm: the k-NN half alone, with no bridges and no repair.
        //
        // Section~\ref{sec:connectivity} finds that real k-NN graphs are already
        // connected at the operating kmax, and it is tempting to conclude that
        // the spanning tree adds nothing on such data. Connectivity does not
        // establish that. An MST over the complete mutual-reachability graph
        // may contain a short edge directly between two regions that the k-NN
        // graph only joins by a long detour around them, which is the situation
        // the CORE-SG paper illustrates with a horseshoe. The two graphs are
        // then both connected and induce different hierarchies. Extracting a
        // tree from this arm and scoring it against the exact one is what tells
        // the two cases apart.
        CoreSGParams pn = pa;
        pn.missing_mode = MissingEdgeMode::None;
        pn.ensure_connected = false;
        CoreSG bare(pn);
        const CoreSGStats sb = bare.build(d.values.data(), n, d.dim);
        csv->row("bridges", d.name, n, d.dim, o.k, "mode", "knn_only", seed, -1,
                 "components_before_repair", double(sb.components_before_repair));

        for (std::uint32_t mpts : {1u, 2u, 4u, 8u, 16u, o.k}) {
            if (mpts > o.k) continue;
            const MstResult ma = approx.mst(mpts);
            const MstResult me = exact.mst(mpts);
            const MstResult mb = bare.mst(mpts);
            const MstComparison c = compare_msts(ma, me);
            const MstComparison cb = compare_msts(mb, me);
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

            // The same three quantities for the k-NN half alone. The component
            // count is emitted next to them because it decides how the weight
            // ratio may be read: where this arm is a forest its total weight is
            // a sum over fewer edges and a ratio below 1 means only that, while
            // where it is a spanning tree the ratio is a like-for-like
            // comparison against the exact one.
            idx_t kb = 0;
            const std::vector<idx_t> lb =
                dbscan_labels_from_mst(mb, static_cast<idx_t>(n), thresh, &kb);
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "knnonly_ari_vs_exact", adjusted_rand_index(lb, le));
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "knnonly_weight_ratio", cb.weight_ratio);
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "knnonly_components", double(cb.approx_components));
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "knnonly_clusters", double(kb));
            // Scored against the anti-hub arm as well, which is the comparison
            // that says whether the bridges changed the answer this system
            // actually produces, as opposed to the ideal one.
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "knnonly_ari_vs_antihub", adjusted_rand_index(lb, la));
            // Emitted because the flat ARI above turned out not to discriminate:
            // the threshold sits far below the merge heights the bridges affect,
            // and both arms share every k-NN edge, so they agree on the short
            // edges by construction. These two compare the trees directly.
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "knnonly_edge_jaccard", cb.edge_jaccard);
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "knnonly_max_weight_gap", cb.max_edge_weight_gap);
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "knnonly_shared_edges", double(cb.shared_edges));
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "exact_mst_edges", double(cb.exact_edges));
            // Absolute weights, because a maximum profile gap of 0.27 says
            // nothing without the scale of the merge heights it is a gap in.
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "exact_mst_weight", cb.exact_weight);
            csv->row("bridges", d.name, n, d.dim, o.k, "mpts", v, seed, -1,
                     "knnonly_mst_weight", cb.approx_weight);
        }

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
// E2c -- how connected is the k-NN half, at full scale and as a function of k
//
// This exists because of a measurement hole. `components_before_repair` is
// filled in by CoreSG::repair_connectivity *after* the bridges have been folded
// into the component labelling, so on every full-scale run in the suite it
// reports 1 by construction -- it says the bridges connected the graph, not how
// far apart the graph was to begin with. The only arm that measures the k-NN
// half alone lives in `bridges`, which caps n at the O(n^2) reference. So the
// component count of the k-NN half at full scale has never been recorded, and
// any claim of the form "this data set fragments into N pieces and the bridges
// reduce it to 1" currently rests on N coming from somewhere else.
//
// The second arm is the one that may bite. Every fragmenting component count
// this project quotes was measured with an *exact* k-NN half
// (brute_force_limit = n). At full scale the half is NN-Descent, and its
// residual errors are precisely long, cross-cluster edges -- exactly the edges
// that would connect a graph the exact k-NN leaves in pieces. If the two arms
// disagree, the approximation of the k-NN half is partially masking the
// fragmentation the bridges exist to repair, and that has to be stated rather
// than discovered by a reader. Both arms are measured here at matching n.
//
// The k sweep turns the project's most transferable observation -- that real
// k-NN graphs are usually already connected -- from a single-k assertion into
// a curve: the k at which each data set connects. That also prices the
// "connectivity gate" idea, since `repair_seconds` is the cost of the check
// and `bridge_seconds` on the gate arm is what the check would let you skip.
// ---------------------------------------------------------------------------

void exp_connectivity(const Options& o, const Data& d, Csv* csv) {
    const std::vector<std::uint32_t> ks =
        o.k_sweep.empty() ? std::vector<std::uint32_t>{4u, 8u, 16u, 24u, 32u, 48u, 64u}
                          : parse_uint_list(o.k_sweep);
    // The exact arm is O(n^2 d), so it runs at the same cap the other
    // exact-reference experiments use -- and, critically, the approximate arm
    // is measured at that same n as well, or the comparison confounds the k-NN
    // half with the data set size.
    const std::size_t n_exact = std::min(d.n, o.exact_limit);
    std::printf("\n[connectivity] full n=%zu, matched n=%zu, k in {", d.n, n_exact);
    for (std::uint32_t k : ks) std::printf("%u ", k);
    std::printf("}\n");

    for (int s = 0; s < std::min(o.seeds, 3); ++s) {
        const std::uint64_t seed = 9100 + s;
        for (std::uint32_t k : ks) {
            Options local = o;
            local.k = k;
            const std::string ks_str = std::to_string(k);

            // `None` + `ensure_connected = false` is what makes
            // components_before_repair mean "components of the k-NN graph":
            // nothing is folded in and nothing is repaired.
            auto knn_only_params = [&](std::uint64_t sd) {
                CoreSGParams p = base_params(local, d, sd);
                p.missing_mode = MissingEdgeMode::None;
                p.ensure_connected = false;
                return p;
            };

            // Arm 1 -- full scale, approximate half. This is the number that
            // describes what the bridges actually face in production.
            {
                CoreSG sg(knn_only_params(seed));
                const CoreSGStats st = sg.build(d.values.data(), d.n, d.dim);
                csv->row("connectivity", d.name, d.n, d.dim, k, "half_k", "approx_full@k" + ks_str,
                         seed, -1, "components_knn_only", double(st.components_before_repair));
                csv->row("connectivity", d.name, d.n, d.dim, k, "half_k", "approx_full@k" + ks_str,
                         seed, -1, "knn_seconds", st.knn_seconds);
                // What a connectivity check costs on its own. `conn_stats` is
                // the union-find pass that derives the component labelling --
                // the actual price of the gate. `repair_seconds` is only the
                // fold-in of the bridges on top of an already-valid labelling,
                // which is microseconds and would flatter the gate badly if
                // quoted in its place.
                csv->row("connectivity", d.name, d.n, d.dim, k, "half_k", "approx_full@k" + ks_str,
                         seed, -1, "components_seconds", st.conn_stats.seconds);
                csv->row("connectivity", d.name, d.n, d.dim, k, "half_k", "approx_full@k" + ks_str,
                         seed, -1, "fold_in_seconds", st.repair_seconds);
            }

            // Arm 2 -- matched n, approximate half.
            // Arm 3 -- matched n, exact half. The pair isolates the effect of
            // approximating the k-NN half on connectivity, with n held equal.
            if (n_exact >= 2) {
                CoreSG approx(knn_only_params(seed));
                const CoreSGStats sa = approx.build(d.values.data(), n_exact, d.dim);
                csv->row("connectivity", d.name, n_exact, d.dim, k, "half_k",
                         "approx_matched@k" + ks_str, seed, -1, "components_knn_only",
                         double(sa.components_before_repair));

                CoreSGParams pe = knn_only_params(seed);
                pe.incr.brute_force_limit = n_exact;
                CoreSG exact(pe);
                const CoreSGStats se = exact.build(d.values.data(), n_exact, d.dim);
                csv->row("connectivity", d.name, n_exact, d.dim, k, "half_k",
                         "exact_matched@k" + ks_str, seed, -1, "components_knn_only",
                         double(se.components_before_repair));
                csv->row("connectivity", d.name, n_exact, d.dim, k, "half_k",
                         "exact_matched@k" + ks_str, seed, -1, "knn_recall_of_approx_half",
                         knn_recall(approx.knn(), exact.knn(), k).recall);
            }
        }

        // The gate arm, at the data set's own k: what the bridges cost, against
        // what the check that would skip them costs. Run once per seed, not per
        // k, because it is a cost comparison rather than a sweep.
        CoreSGParams pg = base_params(o, d, seed);
        pg.missing_mode = MissingEdgeMode::AntiHub;
        CoreSG gated(pg);
        const CoreSGStats sg_st = gated.build(d.values.data(), d.n, d.dim);
        csv->row("connectivity", d.name, d.n, d.dim, o.k, "gate", "antihub", seed, -1,
                 "bridge_seconds", sg_st.missing_seconds);
        csv->row("connectivity", d.name, d.n, d.dim, o.k, "gate", "antihub", seed, -1,
                 "hubness_seconds", sg_st.hubness_seconds);
        csv->row("connectivity", d.name, d.n, d.dim, o.k, "gate", "antihub", seed, -1,
                 "components_seconds", sg_st.conn_stats.seconds);
        csv->row("connectivity", d.name, d.n, d.dim, o.k, "gate", "antihub", seed, -1,
                 "total_seconds", sg_st.total_seconds);
        csv->row("connectivity", d.name, d.n, d.dim, o.k, "gate", "antihub", seed, -1,
                 "bridge_fraction_of_build",
                 (sg_st.hubness_seconds + sg_st.missing_seconds) /
                     (sg_st.total_seconds + 1e-12));
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
    // CPU thread scaling runs LAST on purpose: the device measurements are the
    // point of this experiment, and a run that gets cut short should still have
    // produced them.
    if (o.skip_cpu_scaling) {
        std::printf("  (CPU thread sweep skipped)\n");
        return;
    }
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
// E8 -- amortised cost as a function of batch size
//
// The suite currently contains two batch sizes -- 6 250 points in
// `incremental` and 900 in `drift` -- which is not a curve. It matters because
// an update has a fixed cost that does not shrink with the batch: the hubness
// refresh, the bridge MST over the anti-hub set, and the connectivity pass all
// run once per insert regardless of how many points arrived. Below some batch
// size that fixed cost stops being amortised and the structure is cheaper to
// rebuild than to update, and where that crossover sits is the practical
// question any reader with a stream will ask.
//
// The phase totals are emitted separately from the wall-clock total precisely
// so the fixed part can be separated from the per-point part.
// ---------------------------------------------------------------------------

void exp_batch_size(const Options& o, const Data& d, Csv* csv) {
    const std::size_t n = std::min(d.n, std::size_t(50000));
    const std::vector<double> fractions =
        o.batch_fractions.empty()
            ? std::vector<double>{0.001, 0.005, 0.01, 0.05, 0.10, 0.25}
            : parse_double_list(o.batch_fractions);
    const std::size_t base = n / 2;
    if (base == 0 || n <= base) {
        std::printf("\n[batchsize] (too few points; skipped)\n");
        return;
    }
    std::printf("\n[batchsize] n=%zu, base=%zu, %zu batch sizes\n", n, base, fractions.size());

    // One exact reference for the whole experiment: every arm ends on the same
    // n points in the same order, so recomputing it per arm would only buy
    // O(n^2 d) of identical work.
    ThreadPool pool(o.threads);
    Dataset present(d.dim, d.metric);
    present.append(d.values.data(), n);
    const KnnGraph exact = brute_force_graph(present, o.k, &pool);

    for (int s = 0; s < std::min(o.seeds, 3); ++s) {
        const std::uint64_t seed = 9300 + s;
        for (double frac : fractions) {
            const std::size_t per_batch =
                std::max<std::size_t>(1, static_cast<std::size_t>(frac * double(n)));
            CoreSG sg(base_params(o, d, seed));
            const CoreSGStats bs = sg.build(d.values.data(), base, d.dim);

            double insert_total = 0.0, hubness_total = 0.0, bridge_total = 0.0;
            double repair_total = 0.0, knn_total = 0.0;
            std::size_t cursor = base, batches = 0;
            while (cursor < n) {
                const std::size_t count = std::min(per_batch, n - cursor);
                const CoreSGStats st = sg.insert(d.values.data() + cursor * d.dim, count);
                insert_total += st.total_seconds;
                hubness_total += st.hubness_seconds;
                bridge_total += st.missing_seconds;
                repair_total += st.repair_seconds;
                knn_total += st.knn_seconds;
                cursor += count;
                ++batches;
            }

            // Label by the requested fraction, not by the realised batch size,
            // so arms line up across data sets of different n.
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.4f", frac);
            const std::string v = buf;
            const std::size_t inserted = n - base;

            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "batch_points", double(per_batch));
            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "batches", double(batches));
            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "build_seconds", bs.total_seconds);
            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "insert_total_seconds", insert_total);
            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "seconds_per_inserted_point", insert_total / double(inserted));
            // The three phases that do not shrink with the batch. Their sum
            // against `insert_total_seconds` is the amortisation story.
            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "knn_total_seconds", knn_total);
            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "hubness_total_seconds", hubness_total);
            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "bridge_total_seconds", bridge_total);
            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "repair_total_seconds", repair_total);
            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "fixed_phase_fraction",
                     (hubness_total + bridge_total + repair_total) / (insert_total + 1e-12));
            csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", v, seed, -1,
                     "recall_final", knn_recall(sg.knn(), exact, o.k).recall);
        }

        // The line the curve is read against: one rebuild of the same n.
        CoreSG rebuilt(base_params(o, d, seed));
        const CoreSGStats rs = rebuilt.build(d.values.data(), n, d.dim);
        csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", "rebuild", seed, -1,
                 "insert_total_seconds", rs.total_seconds);
        csv->row("batchsize", d.name, n, d.dim, o.k, "batch_fraction", "rebuild", seed, -1,
                 "recall_final", knn_recall(rebuilt.knn(), exact, o.k).recall);
    }
}

// ---------------------------------------------------------------------------
// E9 -- do the CPU and CUDA backends agree?
//
// Backend parity is covered by a unit test and by `preflight`, but no
// experiment writes it to the CSV, so the honest answer to "do the backends
// produce the same graph?" has been "a unit test says so". This puts a
// measurement behind it.
//
// Note what is and is not being claimed. The two arms are two separately built
// CoreSG objects, which elsewhere in this suite is a trap -- NN-Descent is
// randomised and race-tolerant, so two builds differ even at one seed, and
// attributing that difference to anything other than the thing under test is
// how wrong tables get made. Here the backend *is* the thing under test and
// everything else is held fixed, so the comparison is the right one. The
// weight ratio is reported as a symmetric agreement measure between two
// approximations, not as approx/exact: unlike in `bridges`, a value below 1.0
// is legitimate here and means only that the CPU arm happened to find a
// slightly heavier tree.
// ---------------------------------------------------------------------------

void exp_parity(const Options& o, const Data& d, Csv* csv) {
    const std::size_t n = std::min(d.n, std::size_t(50000));
    std::printf("\n[parity] n=%zu, CPU vs CUDA\n", n);
    if (enumerate_cuda_devices().empty()) {
        std::printf("  (no CUDA devices -- nothing to compare; rerun where a GPU exists)\n");
        return;
    }

    ThreadPool pool(o.threads);
    Dataset present(d.dim, d.metric);
    present.append(d.values.data(), n);
    const KnnGraph exact = brute_force_graph(present, o.k, &pool);

    for (int s = 0; s < std::min(o.seeds, 3); ++s) {
        const std::uint64_t seed = 9400 + s;

        CoreSGParams pg = base_params(o, d, seed);
        pg.backend.prefer_gpu = true;
        CoreSG gpu(pg);
        const CoreSGStats gs = gpu.build(d.values.data(), n, d.dim);

        CoreSGParams pc = base_params(o, d, seed);
        pc.backend.prefer_gpu = false;
        CoreSG cpu(pc);
        const CoreSGStats cs = cpu.build(d.values.data(), n, d.dim);

        // Proof the GPU arm really used a device. Without it a silent host
        // demotion would make the two arms agree perfectly and mean nothing.
        csv->row("parity", d.name, n, d.dim, o.k, "backend", "cuda", seed, -1, "jobs_on_device",
                 double(gs.knn_stats.jobs_on_device));
        csv->row("parity", d.name, n, d.dim, o.k, "backend", "cpu", seed, -1, "jobs_on_device",
                 double(cs.knn_stats.jobs_on_device));

        auto emit_arm = [&](const char* label, const CoreSG& sg, const CoreSGStats& st) {
            csv->row("parity", d.name, n, d.dim, o.k, "backend", label, seed, -1, "build_seconds",
                     st.total_seconds);
            csv->row("parity", d.name, n, d.dim, o.k, "backend", label, seed, -1, "recall",
                     knn_recall(sg.knn(), exact, o.k).recall);
            csv->row("parity", d.name, n, d.dim, o.k, "backend", label, seed, -1,
                     "antihubs_selected", double(st.antihubs_selected));
            csv->row("parity", d.name, n, d.dim, o.k, "backend", label, seed, -1,
                     "components_before_repair", double(st.components_before_repair));
        };
        emit_arm("cuda", gpu, gs);
        emit_arm("cpu", cpu, cs);

        // Agreement of what the two graphs actually produce downstream.
        const std::uint32_t mpts = std::min<std::uint32_t>(o.k, 8);
        const MstResult mg = gpu.mst(mpts);
        const MstResult mc = cpu.mst(mpts);
        const MstComparison c = compare_msts(mg, mc);
        csv->row("parity", d.name, n, d.dim, o.k, "backend", "gpu_over_cpu", seed, -1,
                 "mst_weight_ratio_symmetric", c.weight_ratio);
        csv->row("parity", d.name, n, d.dim, o.k, "backend", "gpu_over_cpu", seed, -1,
                 "max_weight_profile_gap", c.max_edge_weight_gap);

        double thresh = 0.0;
        for (const Edge& e : mc.edges) thresh += e.w;
        thresh /= double(mc.edges.empty() ? 1 : mc.edges.size());
        const std::vector<idx_t> lg =
            dbscan_labels_from_mst(mg, static_cast<idx_t>(n), thresh, nullptr);
        const std::vector<idx_t> lc =
            dbscan_labels_from_mst(mc, static_cast<idx_t>(n), thresh, nullptr);
        csv->row("parity", d.name, n, d.dim, o.k, "backend", "gpu_over_cpu", seed, -1,
                 "ari_gpu_vs_cpu", adjusted_rand_index(lg, lc));
    }
}

// ---------------------------------------------------------------------------
// E10 -- streaming export, so the FISHDBC comparison can actually be joined
//
// FISHDBC is the only external baseline, and as the suite stood the two sides
// could not be put in one table. `scripts/baseline_fishdbc.py` scores FISHDBC
// against sklearn's HDBSCAN* (EOM extraction at a given min_cluster_size);
// every CORE-SG experiment scores DBSCAN* (a flat cut of the MST) against the
// exact CORE-SG. Two different clusterings measured against two different
// references do not compare, whatever the numbers say.
//
// The fix is to make this side emit the one artefact that is method-neutral --
// the MST -- and let the driver script turn it into HDBSCAN* labels with the
// *same* extraction code FISHDBC itself calls (label -> condense_tree ->
// get_clusters). Then the only thing that differs between the two rows is the
// MST each method produced, which is exactly the thing under test.
//
// The arrival order is supplied by the driver rather than generated here: both
// methods must see the same points at the same batch, and reproducing numpy's
// PCG64 permutation in C++ is not worth the risk of getting it subtly wrong.
// ---------------------------------------------------------------------------

void write_mst_file(const std::string& path, std::size_t n, const MstResult& mst) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path + " for writing");
    const std::uint64_t header[2] = {static_cast<std::uint64_t>(n),
                                     static_cast<std::uint64_t>(mst.edges.size())};
    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    for (const Edge& e : mst.edges) {
        const std::uint32_t uv[2] = {e.u, e.v};
        const float w = e.w;
        f.write(reinterpret_cast<const char*>(uv), sizeof(uv));
        f.write(reinterpret_cast<const char*>(&w), sizeof(w));
    }
    if (!f) throw std::runtime_error("failed writing " + path);
}

std::vector<idx_t> read_order_file(const std::string& path, std::size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open order file " + path);
    std::vector<std::int32_t> raw(n);
    f.read(reinterpret_cast<char*>(raw.data()), std::streamsize(n * sizeof(std::int32_t)));
    if (static_cast<std::size_t>(f.gcount()) != n * sizeof(std::int32_t)) {
        throw std::runtime_error("order file " + path + " is shorter than the data set");
    }
    std::vector<idx_t> order(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (raw[i] < 0 || static_cast<std::size_t>(raw[i]) >= n) {
            throw std::runtime_error("order file " + path + " has an out-of-range index");
        }
        order[i] = static_cast<idx_t>(raw[i]);
    }
    return order;
}

void exp_stream_mst(const Options& o, const Data& d, Csv* csv) {
    if (o.order_file.empty() || o.mst_dir.empty()) {
        std::printf("\n[stream] needs --order-file and --mst-dir "
                    "(run it via scripts/compare_incremental.py); skipped\n");
        return;
    }
    const std::vector<idx_t> order = read_order_file(o.order_file, d.n);
    const std::size_t base = static_cast<std::size_t>(o.base_fraction * double(d.n));
    const std::size_t per_batch = (d.n - base) / std::max(1, o.batches);
    if (per_batch == 0) {
        std::printf("\n[stream] (too few points for %d batches; skipped)\n", o.batches);
        return;
    }
    std::printf("\n[stream] n=%zu, base=%zu, %d batches, exporting MST at mpts=%u\n", d.n, base,
                o.batches, o.export_mpts);

    // One run only: the driver invokes this once per seed so that the order
    // file, the seed label and the reference clustering all line up.
    const std::uint64_t seed = o.seed_base;
    CoreSGParams p = base_params(o, d, seed);
    CoreSG sg(p);

    std::vector<float> head = gather_rows(d, order, 0, base);
    const CoreSGStats bs = sg.build(head.data(), base, d.dim);
    csv->row("stream_coresg", d.name, base, d.dim, o.k, "method", "coresg", seed, 0,
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
        const auto t0 = Clock::now();
        const MstResult mst = sg.mst(o.export_mpts);
        const double mst_seconds = since(t0);
        write_mst_file(o.mst_dir + "/" + d.name + "_seed" + std::to_string(seed) + "_batch" +
                           std::to_string(batch) + ".mst",
                       cursor, mst);

        csv->row("stream_coresg", d.name, cursor, d.dim, o.k, "method", "coresg", seed, batch,
                 "insert_seconds", st.total_seconds);
        csv->row("stream_coresg", d.name, cursor, d.dim, o.k, "method", "coresg", seed, batch,
                 "cumulative_seconds", cumulative);
        csv->row("stream_coresg", d.name, cursor, d.dim, o.k, "method", "coresg", seed, batch,
                 "throughput_points_per_sec", double(count) / st.total_seconds);
        // Extraction is charged separately: FISHDBC pays the same cost in its
        // `cluster()` call, which the baseline records as
        // `cluster_extract_seconds`.
        csv->row("stream_coresg", d.name, cursor, d.dim, o.k, "method", "coresg", seed, batch,
                 "cluster_extract_seconds", mst_seconds);
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
        // Emitted so the refinement sweep is readable: `refine_max_fraction`
        // was found to be saturated at its ceiling on every data set and every
        // batch, which means the measurement was reporting the cap rather than
        // the algorithm's own appetite. These two say whether that is still
        // true at a given setting.
        csv->row("sensitivity", d.name, n, d.dim, p.k_max, config, value, seed, -1,
                 "refine_nodes_fraction", double(us.knn_stats.refine_nodes) / double(us.n));
        csv->row("sensitivity", d.name, n, d.dim, p.k_max, config, value, seed, -1,
                 "refine_capped", us.knn_stats.refine_capped ? 1.0 : 0.0);
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

        // Reference arm, and the sweep below is uninterpretable without it.
        // `antihub_multiplier` and `min_per_component` govern bridge coverage,
        // so the metric that can move for them is `components_before_repair`.
        // But a flat 1 across every setting has two completely different
        // explanations -- "the quota is doing its job" and "the k-NN graph was
        // already connected, so there was never anything to cover" -- and only
        // the component count of the k-NN half alone tells them apart. An
        // earlier reading of this sweep put the flatness down to the choice of
        // metric; on these data sets it was the data.
        {
            CoreSGParams pk = base_params(o, d, seed);
            pk.missing_mode = MissingEdgeMode::None;
            pk.ensure_connected = false;
            CoreSG bare(pk);
            const std::size_t half = n / 2;
            bare.build(d.values.data(), half, d.dim);
            const CoreSGStats bu = bare.insert(d.values.data() + half * d.dim, n - half);
            csv->row("sensitivity", d.name, n, d.dim, o.k, "reference", "knn_only", seed, -1,
                     "components_knn_only", double(bu.components_before_repair));
        }

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

        // The anchored strategy's own knobs. An anchored update is cheaper than
        // the resampling one and recalls less, so the first question a reader
        // asks is whether the difference is a fixed loss or a dial. These four
        // sweeps answer it, and `search_ef` is the one that matters: it is the
        // seeding search's beam width, the search is the dominant cost of an
        // update, and it trades cost against recall monotonically.
        for (std::uint32_t ef : {16u, 32u, 64u, 128u, 256u}) {
            CoreSGParams p = base_params(o, d, seed);
            p.incr.strategy = InsertStrategy::Anchored;
            p.incr.search_ef = ef;
            measure("search_ef", std::to_string(ef), p, seed);
        }
        // How much base context an instance is given per new point. Sets the
        // instance size, hence the device footprint, hence the capacity claim.
        for (double ctx : {0.5, 1.0, 2.0, 4.0, 8.0}) {
            CoreSGParams p = base_params(o, d, seed);
            p.incr.strategy = InsertStrategy::Anchored;
            p.incr.context_per_new = ctx;
            measure("context_per_new", std::to_string(ctx), p, seed);
        }
        // How much of its seed list a new point inherits into the instance.
        // 1.0 leaves no free slot for a random candidate, so the batch's own
        // k-NN structure never gets built -- include it so the sweep shows the
        // failure rather than only the fix.
        for (double owf : {0.0, 0.25, 0.5, 1.0}) {
            CoreSGParams p = base_params(o, d, seed);
            p.incr.strategy = InsertStrategy::Anchored;
            p.incr.owned_warm_fraction = owf;
            measure("owned_warm_fraction", std::to_string(owf), p, seed);
        }
        // Reverse edges make a directed k-NN graph navigable; 0 is forward
        // only, which is measurably cheaper and measurably worse.
        for (std::uint32_t rc : {0u, 8u, 16u, 32u}) {
            CoreSGParams p = base_params(o, d, seed);
            p.incr.strategy = InsertStrategy::Anchored;
            p.incr.search_reverse_cap = rc;
            measure("search_reverse_cap", std::to_string(rc), p, seed);
        }
        // Refinement cap policy -- the knob that trades update cost against
        // update quality, and the only one measured as binding on every data
        // set and every batch. Two families, and the comparison between them is
        // the point:
        //
        //   fractional  the shipped default. The refinement set is capped at a
        //               fraction of *n*, so it grows as the graph grows: an
        //               update costs Theta(n) however few points arrived.
        //   absolute    a fixed ceiling (`refine_max_nodes`, disabled by
        //               default). The set stops growing, so an update costs
        //               Theta(batch).
        //
        // That difference is invisible at one graph size, so per-batch rows are
        // emitted alongside the totals. Within a single run the graph doubles
        // from n/2 to n, and the fractional arms' `insert_seconds` should climb
        // across the batches while the absolute arms' stay flat. This is the
        // measurement behind the one place a streaming competitor looks better:
        // FISHDBC's per-batch cost is flat because HNSW insertion is O(log n)
        // per point, while a cap expressed as a fraction of n makes this
        // project's Theta(n). If an absolute cap holds quality at a flat slope,
        // adopting it is a one-line default change with a real result behind it.
        //
        // None of this can use `measure`. That helper inserts the whole second
        // half in one batch, so the new points alone are 50 % of the graph --
        // and since the refinement set is built new points -> first ring ->
        // second ring and the cap drops the outermost nodes first, every
        // fractional setting below 0.5 collapses to exactly 0.5 and the knob
        // appears to do nothing. The cap can only bind on the rings when the
        // batch is small, so this sweep runs the realistic shape: many small
        // batches.
        struct RefineCap {
            const char* name;
            double fraction;
            std::size_t nodes;  ///< 0 leaves the absolute ceiling disabled
        };
        const RefineCap refine_caps[] = {
            {"frac_0.05", 0.05, 0},
            {"frac_0.10", 0.10, 0},
            {"frac_0.25", 0.25, 0},   // the current default
            {"frac_0.50", 0.50, 0},
            {"frac_1.00", 1.00, 0},   // uncapped: the quality ceiling to read the rest against
            {"nodes_1000", 1.00, 1000},
            {"nodes_2500", 1.00, 2500},
            {"nodes_5000", 1.00, 5000},
            {"nodes_10000", 1.00, 10000},
        };
        for (const RefineCap& capcfg : refine_caps) {
            CoreSGParams p = base_params(o, d, seed);
            p.incr.refine_max_fraction = capcfg.fraction;
            p.incr.refine_max_nodes = capcfg.nodes;
            const std::string value = capcfg.name;

            CoreSG sg(p);
            const std::size_t base = n / 2;
            const std::size_t per_batch = (n - base) / std::max(1, o.batches);
            if (per_batch == 0) continue;
            const CoreSGStats bs = sg.build(d.values.data(), base, d.dim);

            double insert_total = 0.0, refine_total = 0.0, refine_frac_sum = 0.0;
            std::size_t cursor = base, batches = 0, capped = 0;
            while (cursor < n) {
                const std::size_t count = std::min(per_batch, n - cursor);
                const CoreSGStats st = sg.insert(d.values.data() + cursor * d.dim, count);
                insert_total += st.total_seconds;
                refine_total += st.knn_stats.refine_seconds;
                refine_frac_sum += double(st.knn_stats.refine_nodes) / double(st.n);
                capped += st.knn_stats.refine_capped ? 1 : 0;
                ++batches;

                // The slope. `n` on these rows is the graph size at that
                // moment, which is what per-batch cost has to be plotted
                // against for the Theta(n)-vs-Theta(batch) question to be
                // answerable at all.
                const long batch = static_cast<long>(batches);
                csv->row("sensitivity", d.name, st.n, d.dim, p.k_max, "refine_cap", value, seed,
                         batch, "insert_seconds", st.total_seconds);
                csv->row("sensitivity", d.name, st.n, d.dim, p.k_max, "refine_cap", value, seed,
                         batch, "refine_seconds", st.knn_stats.refine_seconds);
                csv->row("sensitivity", d.name, st.n, d.dim, p.k_max, "refine_cap", value, seed,
                         batch, "refine_nodes", double(st.knn_stats.refine_nodes));
                csv->row("sensitivity", d.name, st.n, d.dim, p.k_max, "refine_cap", value, seed,
                         batch, "seconds_per_new_point", st.total_seconds / double(count));
                cursor += count;
            }

            csv->row("sensitivity", d.name, n, d.dim, p.k_max, "refine_cap", value, seed, -1,
                     "build_seconds", bs.total_seconds);
            csv->row("sensitivity", d.name, n, d.dim, p.k_max, "refine_cap", value, seed, -1,
                     "insert_seconds", insert_total);
            csv->row("sensitivity", d.name, n, d.dim, p.k_max, "refine_cap", value, seed, -1,
                     "refine_seconds", refine_total);
            csv->row("sensitivity", d.name, n, d.dim, p.k_max, "refine_cap", value, seed, -1,
                     "refine_nodes_fraction", refine_frac_sum / double(batches));
            csv->row("sensitivity", d.name, n, d.dim, p.k_max, "refine_cap", value, seed, -1,
                     "refine_capped", double(capped) / double(batches));
            // The slope itself, as one number: last batch's cost over the
            // first's. ~1.0 means the update is Theta(batch); ~2.0 means it
            // tracked the graph doubling, i.e. Theta(n).
            csv->row("sensitivity", d.name, n, d.dim, p.k_max, "refine_cap", value, seed, -1,
                     "batches", double(batches));
            if (can_reference) {
                KnnGraph exact = brute_force_graph(ref, p.k_max, sg.pool());
                csv->row("sensitivity", d.name, n, d.dim, p.k_max, "refine_cap", value, seed, -1,
                         "recall", knn_recall(sg.knn(), exact, p.k_max).recall);
            }
        }
    }
}

// ---------------------------------------------------------------------------

void usage() {
    std::printf(
        "coresg_exp -- thesis experiment harness (long-format CSV)\n\n"
        "  --experiment  incremental|arrival|bridges|connectivity|budget|mpts|scaling|\n"
        "                drift|sensitivity|batchsize|parity|stream|all\n"
        "  --data FILE [--dim D] [--limit N] [--name LABEL]   or  --synthetic N\n"
        "  --dim D --clusters C     (synthetic only)\n"
        "  --k K --metric l2|cosine|l1\n"
        "  --seeds S --batches B --base-fraction F\n"
        "  --exact-limit N          ceiling for the O(n^2) references\n"
        "  --k-sweep 4,8,16,...     k values for `connectivity`\n"
        "  --batch-fractions a,b,c  batch sizes for `batchsize`, as fractions of n\n"
        "  --order-file F --mst-dir D --export-mpts M --seed-base S   (`stream`;\n"
        "                           driven by scripts/compare_incremental.py)\n"
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
        else if (a == "--k-sweep") o.k_sweep = next();
        else if (a == "--batch-fractions") o.batch_fractions = next();
        else if (a == "--order-file") o.order_file = next();
        else if (a == "--mst-dir") o.mst_dir = next();
        else if (a == "--export-mpts") o.export_mpts = std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--seed-base") o.seed_base = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--devices") o.device_ids = next();
        else if (a == "--search-ef") o.search_ef = std::stoul(next());
        else if (a == "--context-per-new") o.context_per_new = std::stod(next());
        else if (a == "--refine-per-new") o.refine_per_new = std::stod(next());
        else if (a == "--strategy") {
            if (!insert_strategy_from_string(next(), &o.strategy)) {
                std::fprintf(stderr, "unknown --strategy (anchored|resample)\n");
                return 2;
            }
        }
        else if (a == "--no-oom-fallback") o.oom_fallback = false;
        else if (a == "--skip-cpu-scaling") o.skip_cpu_scaling = true;
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
        if (all || e == "connectivity") exp_connectivity(o, d, &csv);
        if (all || e == "budget") exp_budget_dimension(o, d, &csv);
        if (all || e == "mpts") exp_mpts(o, d, &csv);
        if (all || e == "scaling") exp_scaling(o, d, &csv);
        if (all || e == "drift") exp_drift(o, d, &csv);
        if (all || e == "sensitivity") exp_sensitivity(o, d, &csv);
        if (all || e == "strategy") exp_strategy(o, d, &csv);
        if (all || e == "batchsize") exp_batch_size(o, d, &csv);
        if (all || e == "parity") exp_parity(o, d, &csv);
        // `stream` is deliberately NOT in `all`: it only does anything when the
        // driver script supplies an arrival order and an output directory.
        if (e == "stream") exp_stream_mst(o, d, &csv);

        std::printf("\nwrote %s\n", o.out.c_str());
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
