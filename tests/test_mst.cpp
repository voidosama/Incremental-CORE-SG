#include "coresg/mst.hpp"

#include <cstdio>
#include <random>
#include <vector>

#include "coresg/thread_pool.hpp"
#include "test_util.hpp"

using namespace coresg;

namespace {

void test_small_known_tree() {
    //  0 --1-- 1 --2-- 2
    //   \__5__/  \__1__/
    std::vector<Edge> e;
    e.emplace_back(0, 1, 1.0f);
    e.emplace_back(1, 2, 2.0f);
    e.emplace_back(0, 2, 5.0f);
    MstResult r = mst_kruskal(e, 3);
    CHECK_EQ(r.edges.size(), 2u);
    CHECK_NEAR(r.weight, 3.0, 1e-6);
    CHECK_EQ(r.num_components, 1u);
}

void test_forest() {
    std::vector<Edge> e;
    e.emplace_back(0, 1, 1.0f);
    e.emplace_back(2, 3, 1.0f);
    MstResult r = mst_kruskal(e, 5);
    CHECK_EQ(r.edges.size(), 2u);
    CHECK_EQ(r.num_components, 3u);  // {0,1} {2,3} {4}
    CHECK_EQ(r.component.size(), 5u);
    CHECK(r.component[0] == r.component[1]);
    CHECK(r.component[2] == r.component[3]);
    CHECK(r.component[0] != r.component[2]);
}

void test_kruskal_matches_boruvka() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> w(0.01f, 10.0f);
    const idx_t n = 3000;
    std::vector<Edge> e;
    e.reserve(n * 12);
    for (idx_t i = 1; i < n; ++i) {  // a spanning path first, so it is connected
        e.emplace_back(i, static_cast<idx_t>(rng() % i), w(rng));
    }
    for (std::size_t t = 0; t < n * 10u; ++t) {
        const idx_t a = static_cast<idx_t>(rng() % n);
        const idx_t b = static_cast<idx_t>(rng() % n);
        if (a != b) e.emplace_back(a, b, w(rng));
    }

    ThreadPool pool(8);
    MstResult k = mst_kruskal(e, n);
    MstResult b = mst_boruvka(e, n, &pool);
    CHECK_EQ(k.num_components, 1u);
    CHECK_EQ(b.num_components, 1u);
    CHECK_EQ(k.edges.size(), b.edges.size());
    // Ties can pick different edges; the total weight cannot differ.
    CHECK_NEAR(k.weight, b.weight, 1e-3);
}

void test_dense_prim_matches_kruskal() {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<float> w(0.5f, 4.0f);
    const std::size_t m = 120;
    std::vector<float> mat(m * m, 0.0f);
    std::vector<Edge> e;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i + 1; j < m; ++j) {
            const float v = w(rng);
            mat[i * m + j] = v;
            mat[j * m + i] = v;
            e.emplace_back(static_cast<idx_t>(i), static_cast<idx_t>(j), v);
        }
    }
    MstResult prim = mst_dense(mat.data(), m);
    MstResult kru = mst_kruskal(e, static_cast<idx_t>(m));
    CHECK_EQ(prim.edges.size(), m - 1);
    CHECK_NEAR(prim.weight, kru.weight, 1e-3);
    CHECK_EQ(prim.num_components, 1u);
}

void test_prim_on_disconnected_virtual_graph() {
    // Two cliques with infinite weight between them: Prim must return a
    // forest, not garbage.
    const idx_t n = 8;
    MstResult r = mst_prim_virtual(n, [n](idx_t i, float* out) {
        for (idx_t j = 0; j < n; ++j) {
            const bool same_half = (i < 4) == (j < 4);
            out[j] = same_half ? 1.0f : kInfDist;
        }
    });
    CHECK_EQ(r.num_components, 2u);
    CHECK_EQ(r.edges.size(), 6u);
}

void test_components() {
    std::vector<Edge> e;
    e.emplace_back(0, 1, 1.0f);
    e.emplace_back(3, 4, 1.0f);
    idx_t nc = 0;
    std::vector<idx_t> labels = connected_components(e, 6, &nc);
    CHECK_EQ(nc, 4u);  // {0,1} {2} {3,4} {5}
    CHECK(labels[0] == labels[1]);
    CHECK(labels[3] == labels[4]);
}

}  // namespace

int main() {
    test_small_known_tree();
    test_forest();
    test_kruskal_matches_boruvka();
    test_dense_prim_matches_kruskal();
    test_prim_on_disconnected_virtual_graph();
    test_components();
    return coresg_test::summary("mst");
}
