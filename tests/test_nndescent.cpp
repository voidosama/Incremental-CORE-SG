#include "coresg/nndescent.hpp"

#include <cstdio>

#include "coresg/dataset.hpp"
#include "coresg/eval.hpp"
#include "coresg/io.hpp"
#include "coresg/thread_pool.hpp"
#include "test_util.hpp"

using namespace coresg;

namespace {

Dataset make_data(std::size_t n, std::uint32_t dim, std::uint64_t seed) {
    Dataset d(dim, MetricKind::L2);
    std::vector<float> raw = make_blobs(n, dim, 6, 6.0f, seed);
    d.append(raw.data(), n);
    return d;
}

void test_recall_single_thread() {
    const std::size_t n = 2000;
    const std::uint32_t k = 20;
    Dataset data = make_data(n, 8, 11);

    KnnGraph g(static_cast<idx_t>(n), k);
    init_random(data.view(), data.metric(), &g, 1234);

    NNDescentParams p;
    p.k = k;
    p.sample = 12;
    p.iters = 12;
    NNDescentStats st = nndescent(data.view(), data.metric(), &g, p);

    std::string err;
    CHECK_MSG(g.validate(&err), err);

    KnnGraph exact = brute_force_graph(data, k);
    RecallResult r = knn_recall(g, exact, k);
    std::printf("  single-thread %s, rounds=%u updates=%zu\n", format_recall(r).c_str(),
                st.rounds, st.updates);
    CHECK_GE(r.recall, 0.95);
}

void test_recall_multi_thread() {
    const std::size_t n = 4000;
    const std::uint32_t k = 24;
    Dataset data = make_data(n, 12, 77);
    ThreadPool pool(8);

    KnnGraph g(static_cast<idx_t>(n), k);
    init_random(data.view(), data.metric(), &g, 99, &pool);

    NNDescentParams p;
    p.k = k;
    p.sample = 14;
    p.iters = 14;
    nndescent(data.view(), data.metric(), &g, p, &pool);

    std::string err;
    CHECK_MSG(g.validate(&err), err);

    KnnGraph exact = brute_force_graph(data, k, &pool);
    RecallResult r = knn_recall(g, exact, k);
    std::printf("  multi-thread %s\n", format_recall(r).c_str());
    // The reverse cache is deliberately race tolerant, so the multi-threaded
    // result is not bit-identical to the serial one -- only its quality is
    // guaranteed.
    CHECK_GE(r.recall, 0.95);
}

void test_restricted_active_set() {
    // A descent restricted to a node subset must leave every other
    // neighbourhood at least as good as it was, and must improve the subset.
    const std::size_t n = 1500;
    const std::uint32_t k = 16;
    Dataset data = make_data(n, 6, 5);
    ThreadPool pool(4);

    KnnGraph g(static_cast<idx_t>(n), k);
    init_random(data.view(), data.metric(), &g, 3, &pool);

    std::vector<dist_t> before(n);
    for (idx_t v = 0; v < n; ++v) before[v] = g.dists(v)[k - 1];

    std::vector<idx_t> active;
    for (idx_t v = 0; v < 200; ++v) active.push_back(v);

    NNDescentParams p;
    p.k = k;
    p.sample = 10;
    p.iters = 8;
    nndescent(data.view(), data.metric(), &g, p, &pool, &active);

    std::string err;
    CHECK_MSG(g.validate(&err), err);

    bool never_worse = true;
    double improved = 0.0;
    for (idx_t v = 0; v < n; ++v) {
        if (g.dists(v)[k - 1] > before[v] + 1e-6f) never_worse = false;
        if (v < 200 && g.dists(v)[k - 1] < before[v] - 1e-6f) improved += 1.0;
    }
    CHECK_MSG(never_worse, "a restricted descent made some neighbourhood worse");
    CHECK_GE(improved, 150.0);
}

void test_reverse_cache_bounds() {
    ReverseCache rc(10, 4);
    for (int i = 0; i < 20; ++i) rc.push_old(3, static_cast<idx_t>(i));
    std::vector<idx_t> out;
    rc.drain_old(3, &out);
    CHECK_LE(out.size(), 4u);
    // Draining empties the slot, per Algorithm 1 lines 17-18.
    rc.drain_old(3, &out);
    CHECK_EQ(out.size(), 0u);
}

}  // namespace

int main() {
    test_recall_single_thread();
    test_recall_multi_thread();
    test_restricted_active_set();
    test_reverse_cache_bounds();
    return coresg_test::summary("nndescent");
}
