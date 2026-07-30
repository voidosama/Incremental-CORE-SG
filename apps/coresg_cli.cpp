// Command line driver.
//
//   coresg_cli devices
//   coresg_cli build  --data X.fvecs --kmax 32 --out graph.coresg [...]
//   coresg_cli update --in graph.coresg --data batch.fvecs --out graph.coresg
//   coresg_cli extract --in graph.coresg --mpts 2,4,8,16 --out-prefix mst
//   coresg_cli demo   [--n 50000] [--dim 16] [--batches 4]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "coresg/coresg.hpp"
#include "coresg/eval.hpp"
#include "coresg/io.hpp"
#include "coresg/resource.hpp"

using namespace coresg;

namespace {

struct Args {
    std::map<std::string, std::string> kv;
    std::string command;

    bool has(const std::string& k) const { return kv.count(k) != 0; }
    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    long num(const std::string& k, long def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::strtol(it->second.c_str(), nullptr, 10);
    }
    double real(const std::string& k, double def) const {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::strtod(it->second.c_str(), nullptr);
    }
    bool flag(const std::string& k, bool def) const {
        auto it = kv.find(k);
        if (it == kv.end()) return def;
        return it->second != "0" && it->second != "false" && it->second != "no";
    }
};

Args parse(int argc, char** argv) {
    Args a;
    if (argc > 1) a.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string tok = argv[i];
        if (tok.rfind("--", 0) != 0) continue;
        tok = tok.substr(2);
        const std::size_t eq = tok.find('=');
        if (eq != std::string::npos) {
            a.kv[tok.substr(0, eq)] = tok.substr(eq + 1);
        } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            a.kv[tok] = argv[++i];
        } else {
            a.kv[tok] = "1";
        }
    }
    return a;
}

std::vector<std::uint32_t> parse_list(const std::string& s) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(static_cast<std::uint32_t>(std::strtoul(item.c_str(), nullptr, 10)));
    }
    return out;
}

CoreSGParams params_from(const Args& a) {
    CoreSGParams p;
    p.k_max = static_cast<std::uint32_t>(a.num("kmax", 32));
    p.threads = static_cast<unsigned>(a.num("threads", 0));
    p.seed = static_cast<std::uint64_t>(a.num("seed", 42));
    p.verbose = a.flag("verbose", true);

    MetricKind m = MetricKind::L2;
    if (!metric_from_string(a.str("metric", "l2"), &m)) {
        std::fprintf(stderr, "unknown metric '%s'\n", a.str("metric").c_str());
        std::exit(2);
    }
    p.metric = m;

    MissingEdgeMode mode = MissingEdgeMode::AntiHub;
    if (!missing_edge_mode_from_string(a.str("missing", "antihub"), &mode)) {
        std::fprintf(stderr, "unknown --missing mode '%s' (antihub|exact|none)\n",
                     a.str("missing").c_str());
        std::exit(2);
    }
    p.missing_mode = mode;

    p.backend.prefer_gpu = a.flag("gpu", true);
    if (a.has("devices")) {
        for (std::uint32_t d : parse_list(a.str("devices"))) {
            p.backend.device_ids.push_back(static_cast<int>(d));
        }
    }
    p.backend.cpu_slots = static_cast<unsigned>(a.num("cpu-slots", 1));

    p.incr.instances = static_cast<std::uint32_t>(a.num("instances", 4));
    p.incr.sampling = (a.str("sampling", "partition") == "independent")
                          ? InstanceSampling::Independent
                          : InstanceSampling::Partition;
    p.incr.sample_fraction = a.real("sample-fraction", 0.0);
    p.incr.overlap_fraction = a.real("overlap", 0.05);
    p.incr.refine_second_ring = a.flag("second-ring", true);
    p.incr.refine_max_fraction = a.real("refine-max-fraction", 0.25);
    p.incr.refine_max_nodes = static_cast<std::size_t>(a.num("refine-max-nodes", 0));
    p.incr.brute_force_limit = static_cast<std::size_t>(a.num("brute-force-limit", 0));
    p.incr.verbose = p.verbose;

    p.incr.instance_nnd.k = p.k_max;
    p.incr.instance_nnd.sample = static_cast<std::uint32_t>(a.num("sample", 16));
    p.incr.instance_nnd.iters = static_cast<std::uint32_t>(a.num("iters", 10));
    p.incr.refine_nnd.k = p.k_max;
    p.incr.refine_nnd.sample = p.incr.instance_nnd.sample;
    p.incr.refine_nnd.iters = static_cast<std::uint32_t>(a.num("refine-iters", 5));

    p.hub.antihub_multiplier = a.real("antihub-multiplier", 1.0);
    p.hub.min_per_component = static_cast<std::size_t>(a.num("min-antihubs-per-component", 2));
    p.hub.store_reverse_lists = a.flag("reverse-lists", true);
    if (!score_normalization_from_string(a.str("normalization", "arithmetic"),
                                         &p.hub.normalization)) {
        std::fprintf(stderr, "unknown --normalization '%s' (arithmetic|geometric|median)\n",
                     a.str("normalization").c_str());
        std::exit(2);
    }
    p.ensure_connected = a.flag("connect", true);
    p.incremental_connectivity = a.flag("incremental-connectivity", false);
    return p;
}

RawData load(const Args& a, const std::string& key) {
    const std::string path = a.str(key);
    if (path.empty()) {
        std::fprintf(stderr, "missing --%s\n", key.c_str());
        std::exit(2);
    }
    RawData d = read_any(path, static_cast<std::uint32_t>(a.num("dim", 0)),
                         static_cast<std::size_t>(a.num("limit", 0)));
    std::printf("loaded %zu x %u from %s\n", d.n, d.dim, path.c_str());
    return d;
}

void print_stats(const CoreSGStats& s) {
    std::printf("  n                 : %zu (+%zu)\n", s.n, s.new_points);
    std::printf("  k-NN edges        : %zu\n", s.knn_edges);
    std::printf("  bridge edges      : %zu (+%zu repair)\n", s.missing_edges - s.repair_edges,
                s.repair_edges);
    std::printf("  anti-hubs selected: %zu (tagged %u, mean score %.4f)\n", s.antihubs_selected,
                static_cast<unsigned>(s.hub_stats.tagged_antihubs), s.hub_stats.mean_score);
    std::printf("  components        : %u -> %u\n",
                static_cast<unsigned>(s.components_before_repair),
                static_cast<unsigned>(s.components_after_repair));
    std::printf("  timings (s)       : knn %.3f | hubness %.3f | bridges %.3f | repair %.3f "
                "| total %.3f\n",
                s.knn_seconds, s.hubness_seconds, s.missing_seconds, s.repair_seconds,
                s.total_seconds);
    const IncrementalKnnStats& k = s.knn_stats;
    const double phase = k.instance_seconds + k.merge_seconds + k.refine_seconds;
    if (phase > 0.0) {
        std::printf("  knn phases (s)    : instances %.3f (%.0f%%, device parallel) | merge "
                    "%.3f | refine %.3f\n",
                    k.instance_seconds, 100.0 * k.instance_seconds / phase, k.merge_seconds,
                    k.refine_seconds);
    }
}

/// Validates the multi-GPU path end to end before a long run commits to it.
/// Everything here has failed at least once somewhere, or is a silent-wrong
/// mode rather than a loud one: a device below sm_70 (the local-join spin
/// locks need independent thread scheduling), instances quietly demoted to the
/// host, one device doing all the work, a device that OOMs, and results that
/// differ between backends.
int cmd_preflight(const Args& a) {
    int failures = 0;
    auto check = [&](bool ok, const char* what, const std::string& detail = "") {
        std::printf("  [%s] %s%s%s\n", ok ? " ok " : "FAIL", what,
                    detail.empty() ? "" : " -- ", detail.c_str());
        if (!ok) ++failures;
    };

    std::printf("=== devices ===\n");
    const std::vector<DeviceInfo> devs = enumerate_cuda_devices();
    check(!devs.empty(), "CUDA devices visible",
          devs.empty() ? "none; the suite will run CPU-only" : std::to_string(devs.size()));
    for (const DeviceInfo& d : devs) {
        std::printf("       %s\n", describe(d).c_str());
        check(d.cc_major >= 7, "compute capability >= 7.0 (spin locks need Volta+)",
              "device " + std::to_string(d.id) + " is sm_" + std::to_string(d.cc_major) +
                  std::to_string(d.cc_minor));
    }
    if (devs.empty()) return failures ? 1 : 0;

    const std::uint32_t k = static_cast<std::uint32_t>(a.num("kmax", 24));
    const std::uint32_t dim = static_cast<std::uint32_t>(a.num("dim", 32));
    const std::size_t n = static_cast<std::size_t>(a.num("n", 40000));
    std::printf("\n=== per-device correctness (n=%zu, dim=%u, k=%u) ===\n", n, dim, k);
    std::vector<float> raw = make_blobs(n, dim, 12, 8.0f, 1234);

    Dataset ref(dim, MetricKind::L2);
    ref.append(raw.data(), n);
    ThreadPool pool(0);
    KnnGraph exact = brute_force_graph(ref, k, &pool);

    double single_device_seconds = 0.0;
    for (const DeviceInfo& d : devs) {
        CoreSGParams p;
        p.k_max = k;
        p.backend.prefer_gpu = true;
        p.backend.device_ids.assign(1, d.id);
        p.incr.instances = 1;
        p.incr.instance_nnd.k = k;
        try {
            CoreSG sg(p);
            const CoreSGStats st = sg.build(raw.data(), n, dim);
            const double recall = knn_recall(sg.knn(), exact, k).recall;
            if (d.id == devs.front().id) single_device_seconds = st.knn_seconds;
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "recall %.4f, %.3fs, %zu/%zu jobs on device",
                          recall, st.knn_seconds, st.knn_stats.jobs_on_device,
                          st.knn_stats.jobs_on_device + st.knn_stats.jobs_on_host);
            check(recall > 0.85, ("device " + std::to_string(d.id) + " builds correctly").c_str(),
                  buf);
            check(st.knn_stats.jobs_on_device > 0,
                  ("device " + std::to_string(d.id) + " actually ran the job on the GPU").c_str(),
                  st.knn_stats.jobs_on_device ? "" : "silently demoted to the host");
        } catch (const std::exception& e) {
            check(false, ("device " + std::to_string(d.id) + " usable").c_str(), e.what());
        }
    }

    std::printf("\n=== all devices together ===\n");
    std::vector<int> all;
    for (const DeviceInfo& d : devs) all.push_back(d.id);
    // Instance count is held FIXED across the two arms. Comparing one instance
    // on one device against many instances on many devices measures different
    // amounts of work, not parallel speed-up, and reads as a slowdown.
    const std::uint32_t fixed_instances =
        std::max<std::uint32_t>(4u, static_cast<std::uint32_t>(all.size()));
    double one_device_seconds = 0.0;
    try {
        CoreSGParams p;
        p.k_max = k;
        p.backend.prefer_gpu = true;
        p.backend.device_ids.assign(1, all.front());
        p.incr.instances = fixed_instances;
        p.incr.min_base_for_split = 1000;
        p.incr.instance_nnd.k = k;
        CoreSG sg(p);
        one_device_seconds = sg.build(raw.data(), n, dim).knn_seconds;
    } catch (const std::exception& e) {
        check(false, "single-device reference run", e.what());
    }
    try {
        CoreSGParams p;
        p.k_max = k;
        p.backend.prefer_gpu = true;
        p.backend.device_ids = all;
        p.incr.instances = fixed_instances;
        p.incr.min_base_for_split = 1000;
        p.incr.instance_nnd.k = k;
        CoreSG sg(p);
        const CoreSGStats st = sg.build(raw.data(), n, dim);
        const double recall = knn_recall(sg.knn(), exact, k).recall;
        const std::size_t total =
            st.knn_stats.jobs_on_device + st.knn_stats.jobs_on_host;
        char buf[200];
        std::snprintf(buf, sizeof(buf), "recall %.4f, %.3fs, %zu/%zu jobs on device",
                      recall, st.knn_seconds, st.knn_stats.jobs_on_device, total);
        check(recall > 0.85, "multi-device build correct", buf);
        check(st.knn_stats.jobs_on_device == total,
              "every instance ran on a GPU",
              st.knn_stats.jobs_on_device == total
                  ? ""
                  : std::to_string(st.knn_stats.jobs_on_host) +
                        " demoted to the host (instances below the 512-point "
                        "threshold); raise --n or lower --instances");
        check(st.knn_stats.jobs_out_of_memory == 0, "no instance ran out of device memory",
              st.knn_stats.jobs_out_of_memory
                  ? std::to_string(st.knn_stats.jobs_out_of_memory) + " OOM"
                  : "");
        if (one_device_seconds > 0.0 && devs.size() > 1) {
            const double ratio = one_device_seconds / std::max(st.knn_seconds, 1e-9);
            const double phase = st.knn_stats.instance_seconds +
                                 st.knn_stats.merge_seconds + st.knn_stats.refine_seconds;
            const double f = phase > 0.0 ? st.knn_stats.instance_seconds / phase : 0.0;
            const double ceiling = 1.0 / ((1.0 - f) + f / double(devs.size()));
            std::printf("  [info] %u instances: %.3fs on 1 device -> %.3fs on %zu "
                        "= %.2fx (Amdahl ceiling %.2fx at f=%.2f)\n",
                        fixed_instances, one_device_seconds, st.knn_seconds, devs.size(),
                        ratio, ceiling, f);
            if (ratio < 1.0) {
                std::printf("  [warn] slower on more devices -- at this size the per-job "
                            "transfer dominates; the suite runs far larger instances\n");
            }
        }
        std::printf("  [info] largest instance: %zu nodes, %s resident\n",
                    st.knn_stats.largest_instance,
                    format_bytes(st.knn_stats.largest_instance_bytes).c_str());
    } catch (const std::exception& e) {
        check(false, "multi-device build", e.what());
    }

    std::printf("\n=== memory headroom ===\n");
    for (const DeviceMemory& m : device_memory()) {
        char buf[200];
        std::snprintf(buf, sizeof(buf), "device %d: %s used of %s", m.id,
                      format_bytes(m.used_bytes).c_str(),
                      format_bytes(m.total_bytes).c_str());
        std::printf("  [info] %s\n", buf);
    }
    std::printf("  [info] per-point device cost at k=%u, dim=%u: %.0f B "
                "-> %.1fM points per 8 GiB\n",
                k, dim, device_bytes_per_point(pad_dim(dim), k, 16),
                8.0 * (1 << 30) / device_bytes_per_point(pad_dim(dim), k, 16) / 1e6);

    std::printf("\n%s (%d failure%s)\n", failures ? "PREFLIGHT FAILED" : "preflight passed",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}

int cmd_devices() {
    std::printf("CUDA available: %s\n", cuda_available() ? "yes" : "no");
    const std::vector<DeviceInfo> devs = enumerate_cuda_devices();
    if (devs.empty()) {
        std::printf("  (no CUDA devices; the CPU backend will be used)\n");
    }
    for (const DeviceInfo& d : devs) std::printf("  %s\n", describe(d).c_str());
    return 0;
}

int cmd_build(const Args& a) {
    RawData d = load(a, "data");
    CoreSGParams p = params_from(a);
    CoreSG sg(p);
    CoreSGStats st = sg.build(d.values.data(), d.n, d.dim);
    std::printf("build:\n");
    print_stats(st);
    const std::string out = a.str("out");
    if (!out.empty()) {
        sg.save(out);
        std::printf("saved %s\n", out.c_str());
    }
    return 0;
}

int cmd_update(const Args& a) {
    CoreSGParams p = params_from(a);
    CoreSG sg(p);
    const std::string in = a.str("in");
    if (in.empty()) {
        std::fprintf(stderr, "missing --in\n");
        return 2;
    }
    sg.load(in);
    std::printf("loaded %s (n=%u, k_max=%u)\n", in.c_str(),
                static_cast<unsigned>(sg.knn().size()), sg.knn().k());

    RawData d = load(a, "data");
    CoreSGStats st = sg.insert(d.values.data(), d.n);
    std::printf("update:\n");
    print_stats(st);

    const std::string out = a.str("out", in);
    sg.save(out);
    std::printf("saved %s\n", out.c_str());
    return 0;
}

int cmd_extract(const Args& a) {
    CoreSGParams p = params_from(a);
    CoreSG sg(p);
    const std::string in = a.str("in");
    if (in.empty()) {
        std::fprintf(stderr, "missing --in\n");
        return 2;
    }
    sg.load(in);

    std::vector<std::uint32_t> values = parse_list(a.str("mpts"));
    if (values.empty()) {
        for (std::uint32_t m = 1; m <= sg.knn().k(); ++m) values.push_back(m);
    }
    const std::string prefix = a.str("out-prefix");
    for (std::uint32_t m : values) {
        if (m > sg.knn().k()) {
            std::fprintf(stderr, "skipping mpts=%u (> k_max=%u)\n", m, sg.knn().k());
            continue;
        }
        MstResult r = sg.mst(m);
        std::printf("mpts=%-3u edges=%-8zu weight=%.6f components=%u  (%.3fs)\n", m,
                    r.edges.size(), r.weight, static_cast<unsigned>(r.num_components),
                    r.seconds);
        if (!prefix.empty()) {
            write_edges_tsv(prefix + "_mpts" + std::to_string(m) + ".tsv", r.edges);
        }
    }
    return 0;
}

int cmd_demo(const Args& a) {
    const std::size_t n = static_cast<std::size_t>(a.num("n", 50000));
    const std::uint32_t dim = static_cast<std::uint32_t>(a.num("dim", 16));
    const int batches = static_cast<int>(a.num("batches", 4));
    const std::size_t clusters = static_cast<std::size_t>(a.num("clusters", 12));

    std::printf("synthetic demo: n=%zu dim=%u clusters=%zu batches=%d\n", n, dim, clusters,
                batches);
    std::vector<float> raw = make_blobs(n, dim, clusters, 8.0f, 20260728);

    CoreSGParams p = params_from(a);
    CoreSG sg(p);

    const std::size_t initial = n / 2;
    CoreSGStats st = sg.build(raw.data(), initial, dim);
    std::printf("\ninitial build (n=%zu):\n", initial);
    print_stats(st);

    const std::size_t remaining = n - initial;
    const std::size_t per_batch = batches > 0 ? remaining / batches : remaining;
    std::size_t cursor = initial;
    for (int b = 0; b < batches && cursor < n; ++b) {
        const std::size_t count = std::min(per_batch, n - cursor);
        CoreSGStats us = sg.insert(raw.data() + cursor * dim, count);
        std::printf("\nbatch %d (+%zu, n=%zu):\n", b + 1, count, us.n);
        print_stats(us);
        cursor += count;
    }

    std::printf("\nMSTs extracted from the final CORE-SG:\n");
    for (std::uint32_t m : {1u, 2u, 4u, 8u, 16u}) {
        if (m > p.k_max) break;
        MstResult r = sg.mst(m);
        std::printf("  mpts=%-3u edges=%-8zu weight=%.4f components=%u (%.3fs)\n", m,
                    r.edges.size(), r.weight, static_cast<unsigned>(r.num_components),
                    r.seconds);
    }
    return 0;
}

void usage() {
    std::printf(
        "coresg -- incremental, multi-GPU CORE-SG\n\n"
        "  coresg_cli devices\n"
        "  coresg_cli build   --data FILE [--dim D] --kmax K [--out FILE]\n"
        "  coresg_cli update  --in FILE --data FILE [--out FILE]\n"
        "  coresg_cli extract --in FILE [--mpts 2,4,8] [--out-prefix P]\n"
        "  coresg_cli demo    [--n N] [--dim D] [--batches B]\n\n"
        "common options:\n"
        "  --metric l2|cosine|l1     --missing antihub|exact|none\n"
        "  --gpu 0|1                 --devices 0,1,2,3\n"
        "  --threads T               --cpu-slots S\n"
        "  --instances I             --sampling partition|independent\n"
        "  --sample S                --iters N        --refine-iters N\n"
        "  --antihub-multiplier M    --reverse-lists 0|1\n"
        "  --connect 0|1             --brute-force-limit N\n"
        "  --normalization arithmetic|geometric|median\n"
        "  --refine-max-fraction F   --refine-max-nodes N\n");
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    try {
        if (a.command == "devices") return cmd_devices();
        if (a.command == "preflight") return cmd_preflight(a);
        if (a.command == "build") return cmd_build(a);
        if (a.command == "update") return cmd_update(a);
        if (a.command == "extract") return cmd_extract(a);
        if (a.command == "demo") return cmd_demo(a);
        usage();
        return a.command.empty() ? 1 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
