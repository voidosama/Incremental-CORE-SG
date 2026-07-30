#include "coresg/knn_graph.hpp"

#include <algorithm>
#include <vector>

#include "coresg/thread_pool.hpp"
#include "test_util.hpp"

using namespace coresg;

namespace {

void test_sorted_insertion() {
    KnnGraph g(4, 3);
    CHECK(g.try_insert(0, 1, 5.0f, true).inserted);
    CHECK(g.try_insert(0, 2, 1.0f, true).inserted);
    CHECK(g.try_insert(0, 3, 3.0f, true).inserted);
    CHECK_EQ(g.degree(0), 3u);
    CHECK_EQ(g.ids(0)[0], 2u);
    CHECK_EQ(g.ids(0)[1], 3u);
    CHECK_EQ(g.ids(0)[2], 1u);
    CHECK_NEAR(g.dists(0)[0], 1.0, 1e-6);
    CHECK_NEAR(g.dists(0)[2], 5.0, 1e-6);
}

void test_rejections_and_eviction() {
    KnnGraph g(5, 2);
    CHECK(!g.try_insert(0, 0, 1.0f, true).inserted);  // self loop
    CHECK(g.try_insert(0, 1, 2.0f, true).inserted);
    CHECK(!g.try_insert(0, 1, 1.0f, true).inserted);  // duplicate id
    CHECK(g.try_insert(0, 2, 3.0f, true).inserted);

    // Full list, worse candidate: rejected without touching anything.
    InsertResult r = g.try_insert(0, 3, 9.0f, true);
    CHECK(!r.inserted);
    CHECK_EQ(r.evicted, kInvalidIdx);

    // Full list, better candidate: evicts the last one and reports it, which
    // is what the hubness index relies on to keep reverse degrees exact.
    r = g.try_insert(0, 4, 1.0f, true);
    CHECK(r.inserted);
    CHECK_EQ(r.evicted, 2u);
    CHECK_EQ(g.ids(0)[0], 4u);
    CHECK_EQ(g.ids(0)[1], 1u);
}

void test_core_distance() {
    KnnGraph g(2, 4);
    g.try_insert(0, 1, 1.0f, false);
    // mpts is 1-based and self is excluded, per the CORE-SG definition.
    CHECK_NEAR(g.core_dist(0, 1), 1.0, 1e-6);
    // Beyond the filled range: clamp to the widest known radius.
    CHECK_NEAR(g.core_dist(0, 4), 1.0, 1e-6);
    // Empty neighbourhood.
    CHECK(g.core_dist(1, 1) == kInfDist);
}

void test_undirected_dedup() {
    KnnGraph g(3, 2);
    // A mutual pair must contribute exactly one undirected edge.
    g.try_insert(0, 1, 1.0f, false);
    g.try_insert(1, 0, 1.0f, false);
    // A one-way edge still contributes one.
    g.try_insert(2, 0, 4.0f, false);

    std::vector<Edge> e = g.undirected_edges();
    CHECK_EQ(e.size(), 2u);
    CHECK_EQ(g.undirected_edge_count(), 2u);

    std::vector<std::uint64_t> keys;
    for (const Edge& x : e) keys.push_back(edge_key(x.u, x.v));
    std::sort(keys.begin(), keys.end());
    CHECK(keys[0] == edge_key(0, 1));
    CHECK(keys[1] == edge_key(0, 2));
}

void test_grow_preserves() {
    KnnGraph g(2, 2);
    g.try_insert(0, 1, 1.0f, true);
    g.grow(6);
    CHECK_EQ(g.size(), 6u);
    CHECK_EQ(g.degree(0), 1u);
    CHECK_EQ(g.ids(0)[0], 1u);
    CHECK_EQ(g.degree(5), 0u);
    CHECK(g.try_insert(5, 3, 2.0f, true).inserted);
}

void test_concurrent_inserts() {
    // Many threads hammering the same neighbourhoods: the result must still be
    // a valid sorted, duplicate-free top-k.
    const idx_t n = 256;
    const std::uint32_t k = 16;
    KnnGraph g(n, k);
    g.ensure_locks();
    ThreadPool pool(8);
    pool.parallel_for(std::size_t(n) * 64, [&](std::size_t t, int) {
        const idx_t node = static_cast<idx_t>(t % n);
        const idx_t cand = static_cast<idx_t>((t * 2654435761u) % n);
        const dist_t d = static_cast<dist_t>((t * 7919u) % 1000) * 0.01f;
        g.try_insert_mt(node, cand, d, true);
    });
    std::string err;
    CHECK_MSG(g.validate(&err), err);
    for (idx_t v = 0; v < n; ++v) CHECK_LE(g.degree(v), k);
}

}  // namespace

int main() {
    test_sorted_insertion();
    test_rejections_and_eviction();
    test_core_distance();
    test_undirected_dedup();
    test_grow_preserves();
    test_concurrent_inserts();
    return coresg_test::summary("knn_graph");
}
