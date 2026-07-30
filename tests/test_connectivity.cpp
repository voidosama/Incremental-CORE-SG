#include "coresg/connectivity.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "coresg/coresg.hpp"
#include "coresg/hubness.hpp"
#include "coresg/io.hpp"
#include "coresg/mst.hpp"
#include "coresg/nndescent.hpp"
#include "coresg/thread_pool.hpp"
#include "test_util.hpp"

using namespace coresg;

namespace {

/// Reference: components of the materialised undirected edge list.
std::vector<idx_t> reference_labels(const KnnGraph& g, idx_t* ncomp) {
    return connected_components(g.undirected_edges(), g.size(), ncomp);
}

/// Two labellings are equal as *partitions* even if the label numbering
/// differs; the tracker promises dense labels, so compare the induced
/// partition rather than the raw integers.
bool same_partition(const std::vector<idx_t>& a, const std::vector<idx_t>& b) {
    if (a.size() != b.size()) return false;
    std::vector<idx_t> a2b(a.size(), kInvalidIdx), b2a(a.size(), kInvalidIdx);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const idx_t x = a[i], y = b[i];
        if (x >= a2b.size() || y >= b2a.size()) return false;
        if (a2b[x] == kInvalidIdx) a2b[x] = y;
        if (b2a[y] == kInvalidIdx) b2a[y] = x;
        if (a2b[x] != y || b2a[y] != x) return false;
    }
    return true;
}

void test_rebuild_matches_reference() {
    KnnGraph g(8, 2);
    // {0,1,2} - {3,4} - {5} - {6,7}
    g.try_insert(0, 1, 1.0f, false);
    g.try_insert(1, 2, 1.0f, false);
    g.try_insert(3, 4, 1.0f, false);
    g.try_insert(6, 7, 1.0f, false);

    ConnectivityTracker t;
    t.reset(8);
    ConnectivityStats st = t.rebuild(g);
    CHECK_EQ(st.components, 4u);
    CHECK(!st.incremental);

    idx_t ref_n = 0;
    const std::vector<idx_t> ref = reference_labels(g, &ref_n);
    CHECK_EQ(st.components, ref_n);
    CHECK(same_partition(t.labels(), ref));
}

void test_parallel_rebuild_is_deterministic() {
    const std::size_t n = 8000;
    const std::uint32_t k = 12;
    Dataset data(6, MetricKind::L2);
    std::vector<float> raw = make_blobs(n, 6, 20, 14.0f, 909);
    data.append(raw.data(), n);
    ThreadPool pool(8);

    KnnGraph g(static_cast<idx_t>(n), k);
    init_random(data.view(), data.metric(), &g, 4, &pool);
    NNDescentParams p;
    p.k = k;
    p.iters = 8;
    nndescent(data.view(), data.metric(), &g, p, &pool);

    ConnectivityTracker serial, parallel;
    serial.reset(static_cast<idx_t>(n));
    parallel.reset(static_cast<idx_t>(n));
    ConnectivityStats a = serial.rebuild(g, nullptr);
    ConnectivityStats b = parallel.rebuild(g, &pool);

    CHECK_EQ(a.components, b.components);
    // Union-by-index makes the outcome independent of the thread schedule, so
    // the labels must match exactly, not merely as partitions.
    bool identical = serial.labels() == parallel.labels();
    CHECK_MSG(identical, "parallel rebuild disagreed with the serial one");

    idx_t ref_n = 0;
    const std::vector<idx_t> ref = reference_labels(g, &ref_n);
    CHECK_EQ(a.components, ref_n);
    CHECK(same_partition(parallel.labels(), ref));
    std::printf("  %zu nodes, %u components, serial %.4fs vs pool %.4fs\n", n,
                static_cast<unsigned>(a.components), a.seconds, b.seconds);
}

void test_labels_with_extra_edges() {
    KnnGraph g(6, 2);
    g.try_insert(0, 1, 1.0f, false);
    g.try_insert(2, 3, 1.0f, false);
    g.try_insert(4, 5, 1.0f, false);
    ConnectivityTracker t;
    t.reset(6);
    t.rebuild(g);
    CHECK_EQ(t.components(), 3u);

    std::vector<Edge> bridges;
    bridges.emplace_back(1, 2, 5.0f);
    idx_t nc = 0;
    const std::vector<idx_t> merged = t.labels_with(bridges, &nc);
    CHECK_EQ(nc, 2u);
    CHECK(merged[0] == merged[3]);
    CHECK(merged[0] != merged[4]);
    // The tracker's own labelling must be untouched by the query.
    CHECK_EQ(t.components(), 3u);
}

/// The load-bearing property: an incremental update must produce exactly what
/// a rebuild would, batch after batch, including when it decides to fall back.
void test_incremental_matches_rebuild() {
    const std::size_t base = 6000;
    const std::size_t batch = 1200;
    const int batches = 5;
    const std::uint32_t dim = 8;
    const std::uint32_t k = 16;
    std::vector<float> all = make_blobs(base + batch * batches, dim, 25, 12.0f, 4242);

    CoreSGParams p;
    p.k_max = k;
    p.threads = 8;
    p.backend.prefer_gpu = false;
    p.incr.instances = 3;
    p.incr.min_base_for_split = 1000;
    p.incr.instance_nnd.k = k;
    p.incr.instance_nnd.iters = 10;
    p.incr.refine_nnd.k = k;
    p.incr.refine_nnd.iters = 5;
    p.incremental_connectivity = true;  // off by default; exercised here on purpose

    CoreSG sg(p);
    sg.build(all.data(), base, dim);

    std::size_t fast = 0;
    for (int b = 0; b < batches; ++b) {
        CoreSGStats st = sg.insert(all.data() + (base + b * batch) * dim, batch);
        if (st.conn_stats.incremental) ++fast;

        ConnectivityTracker fresh;
        fresh.reset(sg.knn().size());
        ConnectivityStats ref = fresh.rebuild(sg.knn(), sg.pool());

        CHECK_EQ(st.conn_stats.components, ref.components);
        bool identical = sg.connectivity().labels() == fresh.labels();
        CHECK_MSG(identical, "incremental labels differ from a rebuild at batch " +
                                 std::to_string(b));

        std::printf("  batch %d: %-11s %u comps, %zu destroyed (%zu certified, %zu dirty "
                    "comps), union edges %zu vs %zu for a rebuild (%.0f%%)\n",
                    b + 1, st.conn_stats.incremental ? "incremental" : "full rebuild",
                    static_cast<unsigned>(st.conn_stats.components),
                    st.conn_stats.destroyed_edges, st.conn_stats.certified_edges,
                    st.conn_stats.dirty_components, st.conn_stats.union_edges, ref.union_edges,
                    100.0 * double(st.conn_stats.union_edges) /
                        double(ref.union_edges ? ref.union_edges : 1));
    }
    std::printf("  fast path taken in %zu / %d batches\n", fast, batches);
    CHECK_EQ(sg.mst(k).num_components, 1u);
}

/// A destroyed edge that really is a cut must not be certified: the tracker
/// has to notice and fall back rather than report stale components.
void test_genuine_split_is_detected() {
    KnnGraph g(6, 2);
    HubnessIndex hub;
    // {0,1,2} joined to {3,4,5} by the single edge (2,3).
    g.try_insert(0, 1, 1.0f, false);
    g.try_insert(1, 2, 1.0f, false);
    g.try_insert(2, 3, 9.0f, false);
    g.try_insert(3, 4, 1.0f, false);
    g.try_insert(4, 5, 1.0f, false);

    ConnectivityTracker t;
    t.reset(6);
    t.rebuild(g);
    CHECK_EQ(t.components(), 1u);

    // Evict the bridge by hand, exactly as an insertion would.
    idx_t* ids = g.mutable_ids(2);
    dist_t* d = g.mutable_dists(2);
    for (std::uint32_t j = 0; j < g.degree(2); ++j) {
        if (ids[j] == 3) {
            ids[j] = kInvalidIdx;
            d[j] = kInfDist;
        }
    }
    std::vector<std::pair<idx_t, idx_t>> removed;
    removed.emplace_back(2, 3);
    std::vector<idx_t> changed;
    changed.push_back(2);

    ConnectivityStats st = t.update(g, removed, changed, 6, &hub, nullptr, 512);
    CHECK_EQ(st.destroyed_edges, 1u);
    CHECK_EQ(st.certified_edges, 0u);
    CHECK_MSG(!st.incremental, "an uncertifiable deletion must force a rebuild");
    CHECK_EQ(st.components, 2u);
}

}  // namespace

int main() {
    test_rebuild_matches_reference();
    test_parallel_rebuild_is_deterministic();
    test_labels_with_extra_edges();
    test_incremental_matches_rebuild();
    test_genuine_split_is_detected();
    return coresg_test::summary("connectivity");
}
