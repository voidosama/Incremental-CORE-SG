#include "coresg/hubness.hpp"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <vector>

#include "coresg/dataset.hpp"
#include "coresg/io.hpp"
#include "coresg/mst.hpp"
#include "coresg/nndescent.hpp"
#include "coresg/thread_pool.hpp"
#include "test_util.hpp"

using namespace coresg;

namespace {

std::vector<std::uint32_t> brute_reverse_degrees(const KnnGraph& g) {
    std::vector<std::uint32_t> rd(g.size(), 0);
    for (idx_t v = 0; v < g.size(); ++v) {
        for (std::uint32_t j = 0; j < g.degree(v); ++j) ++rd[g.ids(v)[j]];
    }
    return rd;
}

/// The formula, written out independently of the implementation:
///   score(v) = rdeg(v) / mean_{u in NN(v)} rdeg(u)
double reference_score(const KnnGraph& g, const std::vector<std::uint32_t>& rd, idx_t v,
                       std::uint32_t k) {
    const std::uint32_t deg = g.degree(v);
    long long sum = 0;
    for (std::uint32_t j = 0; j < deg; ++j) sum += rd[g.ids(v)[j]];
    const long long own = rd[v];
    if (own > 0 && sum == 0) return double(k);
    if (own == 0 && sum > 0) return 0.0;
    if (own == 0 && sum == 0) return 1.0;
    if (deg == 0) return 1.0;
    return double(own) / (double(sum) / double(deg));
}

void test_degrees_and_scores() {
    // 4 nodes, k = 2, hand-built so the degrees are obvious.
    KnnGraph g(4, 2);
    g.try_insert(0, 1, 1.0f, false);
    g.try_insert(0, 2, 2.0f, false);
    g.try_insert(1, 0, 1.0f, false);
    g.try_insert(1, 2, 3.0f, false);
    g.try_insert(2, 0, 2.0f, false);
    g.try_insert(2, 1, 3.0f, false);
    g.try_insert(3, 0, 9.0f, false);

    HubnessIndex hub;
    hub.build(g);
    const std::vector<std::uint32_t> rd = brute_reverse_degrees(g);
    // node 0 is pointed at by 1, 2 and 3 -> the local hub.
    CHECK_EQ(hub.reverse_degree(0), 3u);
    CHECK_EQ(hub.reverse_degree(1), 2u);
    CHECK_EQ(hub.reverse_degree(2), 2u);
    CHECK_EQ(hub.reverse_degree(3), 0u);

    hub.rescore_all(g, 2, 5);
    for (idx_t v = 0; v < 4; ++v) {
        CHECK_NEAR(hub.reverse_score(v), reference_score(g, rd, v, 2), 1e-9);
    }
    CHECK(hub.reverse_score(0) > 1.0);   // hub
    CHECK(hub.reverse_score(3) == 0.0);  // no reverse neighbours at all
    CHECK(hub.is_antihub(3));
    CHECK(!hub.is_antihub(0));
}

void test_degenerate_rules() {
    // A single node with no edges anywhere: rdeg 0, neighbour sum 0 -> 1.
    KnnGraph g(3, 2);
    HubnessIndex hub;
    hub.build(g);
    hub.rescore_all(g, 2, 0);
    CHECK_NEAR(hub.reverse_score(0), 1.0, 1e-12);

    // rdeg > 0 but every neighbour has rdeg 0 -> k.
    KnnGraph h(4, 3);
    h.try_insert(0, 1, 1.0f, false);  // 1 gains a reverse neighbour
    h.try_insert(1, 2, 1.0f, false);  // 1's only neighbour is 2, rdeg(2)=1
    h.try_insert(2, 3, 1.0f, false);
    HubnessIndex hub2;
    HubnessParams p;
    p.k_for_score = 3;
    hub2.set_params(p);
    hub2.build(h);
    hub2.rescore_all(h, 2, 3);
    // node 3: rdeg(3)=1, its neighbourhood is empty -> sum == 0 -> k
    CHECK_NEAR(hub2.reverse_score(3), 3.0, 1e-12);
}

void test_incremental_matches_batch() {
    // The whole point of the observer: after a batch of edge changes, a
    // refresh() restricted to the dirty region must produce exactly the same
    // scores as rescoring every node from scratch.
    const std::size_t n = 3000;
    const std::uint32_t k = 16;
    Dataset data(6, MetricKind::L2);
    std::vector<float> raw = make_blobs(n, 6, 5, 5.0f, 21);
    data.append(raw.data(), n);
    ThreadPool pool(4);

    // Deliberately stop the first descent early, so the second one below has
    // real work to do and the observer sees a substantial amount of churn.
    KnnGraph g(static_cast<idx_t>(n), k);
    init_random(data.view(), data.metric(), &g, 5, &pool);
    NNDescentParams p;
    p.k = k;
    p.sample = 10;
    p.iters = 2;
    p.delta = 0.0;  // no early exit
    nndescent(data.view(), data.metric(), &g, p, &pool);

    HubnessIndex hub;
    hub.build(g, &pool);
    hub.rescore_all(g, data.dim(), g.undirected_edge_count(&pool), &pool);
    hub.attach(&g);

    // Perturb the graph the way an incremental update would.
    NNDescentParams p2 = p;
    p2.iters = 6;
    p2.seed = 999;
    g.set_all_new_flags();
    NNDescentStats ds = nndescent(data.view(), data.metric(), &g, p2, &pool);
    CHECK_GE(ds.updates, 100u);  // the observer must actually have seen churn

    const std::size_t edges = g.undirected_edge_count(&pool);
    HubnessStats st = hub.refresh(g, data.dim(), edges, &pool);
    std::printf("  descent made %zu updates; refresh rescored %zu / %u nodes, %zu selected\n",
                ds.updates, st.rescored, static_cast<unsigned>(n), st.selected);
    CHECK_GE(st.rescored, 1u);
    CHECK_LE(st.rescored, n);

    std::vector<double> incremental(n);
    for (idx_t v = 0; v < n; ++v) incremental[v] = hub.reverse_score(v);
    const std::vector<idx_t> sel_incremental = hub.selected();

    // Ground truth: recount everything.
    HubnessIndex fresh;
    fresh.build(g, &pool);
    fresh.rescore_all(g, data.dim(), edges, &pool);

    const std::vector<std::uint32_t> rd = brute_reverse_degrees(g);
    bool degrees_ok = true, scores_ok = true;
    for (idx_t v = 0; v < n; ++v) {
        if (hub.reverse_degree(v) != rd[v]) degrees_ok = false;
        if (std::fabs(incremental[v] - fresh.reverse_score(v)) > 1e-9) scores_ok = false;
    }
    CHECK_MSG(degrees_ok, "incrementally tracked reverse degrees drifted");
    CHECK_MSG(scores_ok, "incremental scores differ from a full rescore");

    CHECK_EQ(sel_incremental.size(), fresh.selected().size());
    bool selection_ok = sel_incremental.size() == fresh.selected().size();
    for (std::size_t i = 0; selection_ok && i < sel_incremental.size(); ++i) {
        if (sel_incremental[i] != fresh.selected()[i]) selection_ok = false;
    }
    CHECK_MSG(selection_ok, "anti-hub selection differs between incremental and batch");

    hub.detach(&g);
}

void test_selection_budget() {
    const std::size_t n = 2000;
    Dataset data(4, MetricKind::L2);
    std::vector<float> raw = make_blobs(n, 4, 4, 5.0f, 3);
    data.append(raw.data(), n);
    KnnGraph g(static_cast<idx_t>(n), 12);
    init_random(data.view(), data.metric(), &g, 1);
    NNDescentParams p;
    p.k = 12;
    p.iters = 6;
    nndescent(data.view(), data.metric(), &g, p);

    HubnessIndex hub;
    hub.build(g);
    const std::size_t edges = g.undirected_edge_count();
    HubnessStats st = hub.rescore_all(g, data.dim(), edges);

    const std::size_t expected =
        static_cast<std::size_t>(std::llround(4.0 * std::log(double(edges))));
    CHECK_EQ(st.selected, expected);

    // Selected nodes must be the lowest scoring ones.
    double worst_selected = 0.0;
    for (idx_t v : hub.selected()) worst_selected = std::max(worst_selected, hub.reverse_score(v));
    std::size_t violations = 0;
    std::vector<std::uint8_t> chosen(n, 0);
    for (idx_t v : hub.selected()) chosen[v] = 1;
    for (idx_t v = 0; v < n; ++v) {
        if (!chosen[v] && hub.reverse_score(v) < worst_selected) ++violations;
    }
    CHECK_EQ(violations, 0u);
}

/// The alternative denominators, checked against independent formulas, plus
/// the property that actually matters: they must not disturb the *ordering*
/// that anti-hub selection depends on.
void test_score_normalizations() {
    KnnGraph g(6, 3);
    // Give node 0 three neighbours with deliberately unequal reverse degrees,
    // so mean, geometric mean and median all differ.
    g.try_insert(0, 1, 1.0f, false);
    g.try_insert(0, 2, 2.0f, false);
    g.try_insert(0, 3, 3.0f, false);
    for (idx_t v = 1; v < 6; ++v) g.try_insert(v, 1, 1.0f, false);  // self loop at v=1 rejected
    g.try_insert(4, 2, 1.0f, false);
    g.try_insert(5, 2, 1.0f, false);
    // rd(1) = 5 (from 0, 2, 3, 4, 5), rd(2) = 3 (from 0, 4, 5),
    // rd(3) = 1 (from 0), rd(0) = 0.

    const std::vector<std::uint32_t> rd = brute_reverse_degrees(g);
    CHECK_EQ(rd[1], 5u);
    CHECK_EQ(rd[2], 3u);
    CHECK_EQ(rd[3], 1u);

    // Node 0 has reverse degree 0 and a non-zero neighbourhood total, so the
    // degenerate rule pins it at 0 whatever the denominator is.
    for (ScoreNormalization norm : {ScoreNormalization::ArithmeticMean,
                                    ScoreNormalization::GeometricMean,
                                    ScoreNormalization::Median}) {
        HubnessParams p;
        p.normalization = norm;
        p.k_for_score = 3;
        HubnessIndex hub(p);
        hub.build(g);
        hub.rescore_all(g, 2, 6);
        CHECK_NEAR(hub.reverse_score(0), 0.0, 1e-12);
    }

    // Node 4's neighbourhood is {1, 2}: reverse degrees 5 and 2, and rd(4)=0
    // -> still the degenerate rule. Use node 5 with an injected edge instead
    // so the denominator is actually exercised.
    KnnGraph h(4, 2);
    h.try_insert(0, 1, 1.0f, false);
    h.try_insert(0, 2, 2.0f, false);
    h.try_insert(1, 0, 1.0f, false);  // rd(0) = 1
    h.try_insert(2, 1, 1.0f, false);  // rd(1) = 2
    h.try_insert(3, 1, 1.0f, false);  // rd(1) = 3 total, rd(2) = 1
    const std::vector<std::uint32_t> rh = brute_reverse_degrees(h);
    // NN(0) = {1, 2} with reverse degrees {3, 1}; rd(0) = 1.
    CHECK_EQ(rh[1], 3u);
    CHECK_EQ(rh[2], 1u);
    CHECK_EQ(rh[0], 1u);

    struct Case {
        ScoreNormalization norm;
        double denom;
    };
    const Case cases[] = {
        {ScoreNormalization::ArithmeticMean, (3.0 + 1.0) / 2.0},               // 2.0
        {ScoreNormalization::GeometricMean, std::exp((std::log(4.0) + std::log(2.0)) / 2.0) - 1.0},
        {ScoreNormalization::Median, 1.0},  // lower median of {1, 3}
    };
    for (const Case& c : cases) {
        HubnessParams p;
        p.normalization = c.norm;
        p.k_for_score = 2;
        HubnessIndex hub(p);
        hub.build(h);
        hub.rescore_all(h, 2, 3);
        CHECK_NEAR(hub.reverse_score(0), 1.0 / c.denom, 1e-9);
    }
}

/// Whichever denominator is used, the selected set must stay largely the same:
/// the bridges depend on the ranking, not on where the neutral point sits.
void test_normalization_selection_agreement() {
    const std::size_t n = 4000;
    const std::uint32_t k = 16;
    Dataset data(8, MetricKind::L2);
    std::vector<float> raw = make_blobs(n, 8, 6, 6.0f, 3131);
    data.append(raw.data(), n);
    ThreadPool pool(4);

    KnnGraph g(static_cast<idx_t>(n), k);
    init_random(data.view(), data.metric(), &g, 17, &pool);
    NNDescentParams p;
    p.k = k;
    p.sample = 12;
    p.iters = 10;
    nndescent(data.view(), data.metric(), &g, p, &pool);
    const std::size_t edges = g.undirected_edge_count(&pool);

    std::vector<idx_t> reference;
    for (ScoreNormalization norm : {ScoreNormalization::ArithmeticMean,
                                    ScoreNormalization::GeometricMean,
                                    ScoreNormalization::Median}) {
        HubnessParams hp;
        hp.normalization = norm;
        hp.k_for_score = k;
        HubnessIndex hub(hp);
        hub.build(g, &pool);
        HubnessStats st = hub.rescore_all(g, data.dim(), edges, &pool);

        std::vector<idx_t> sel = hub.selected();
        std::sort(sel.begin(), sel.end());
        if (reference.empty()) {
            reference = sel;
        } else {
            std::vector<idx_t> common;
            std::set_intersection(sel.begin(), sel.end(), reference.begin(), reference.end(),
                                  std::back_inserter(common));
            const double overlap = double(common.size()) / double(reference.size());
            std::printf("  %-10s tagged %.1f%%, selected %zu, overlap with arithmetic %.1f%%\n",
                        to_string(norm), 100.0 * hub.antihub_fraction(), st.selected,
                        100.0 * overlap);
            CHECK_GE(overlap, 0.85);
        }
    }
}

/// The per-component quota must guarantee coverage even where the global
/// budget cannot: a small, uniform, detached component has *no* node scoring
/// below 1 (everyone's reverse degree equals their neighbours', so everyone is
/// exactly neutral), so a score-ranked budget can never reach it.
void test_component_quota() {
    // Two components of 4 mutually-linked nodes each, k = 3. Inside each, every
    // node lists the other three, so every reverse degree is 3 and every score
    // is exactly 1 -- nothing is tagged an anti-hub anywhere.
    const std::uint32_t k = 3;
    KnnGraph g(8, k);
    for (idx_t base : {idx_t(0), idx_t(4)}) {
        for (idx_t a = 0; a < 4; ++a) {
            for (idx_t b = 0; b < 4; ++b) {
                if (a != b) g.try_insert(base + a, base + b, 1.0f + float(a), false);
            }
        }
    }

    HubnessParams p;
    p.k_for_score = k;
    p.min_selected = 2;
    p.max_selected = 2;  // force the budget to be too small to cover both

    HubnessIndex plain(p);
    plain.build(g);
    plain.rescore_all(g, 2, 12);
    CHECK_EQ(plain.antihub_fraction(), 0.0);  // uniform -> nobody scores below 1

    // Budget alone: both picks can land in the same component.
    std::vector<idx_t> comp(8);
    for (idx_t v = 0; v < 8; ++v) comp[v] = (v < 4) ? 0u : 1u;

    HubnessParams q = p;
    q.min_per_component = 2;
    HubnessIndex quota(q);
    quota.build(g);
    quota.set_components(&comp, 2);
    HubnessStats st = quota.rescore_all(g, 2, 12);

    std::size_t from[2] = {0, 0};
    for (idx_t v : quota.selected()) ++from[comp[v]];
    std::printf("  quota: %zu selected (%zu budget + %zu quota), %zu from each component\n",
                st.selected, st.budget_selected, st.quota_selected, from[0]);
    CHECK_GE(from[0], 2u);
    CHECK_GE(from[1], 2u);
    CHECK_EQ(st.covered_components, 2u);
}

/// The quota must never shrink the selection, and must be a no-op when the
/// budget already covers every component.
void test_quota_is_a_superset() {
    const std::size_t n = 3000;
    const std::uint32_t k = 16;
    Dataset data(6, MetricKind::L2);
    std::vector<float> raw = make_blobs(n, 6, 4, 6.0f, 66);
    data.append(raw.data(), n);
    ThreadPool pool(4);

    KnnGraph g(static_cast<idx_t>(n), k);
    init_random(data.view(), data.metric(), &g, 2, &pool);
    NNDescentParams np;
    np.k = k;
    np.iters = 10;
    nndescent(data.view(), data.metric(), &g, np, &pool);
    const std::size_t edges = g.undirected_edge_count(&pool);

    idx_t nc = 0;
    const std::vector<idx_t> comp = connected_components(g.undirected_edges(&pool),
                                                        static_cast<idx_t>(n), &nc);

    HubnessParams p;
    p.k_for_score = k;
    HubnessIndex plain(p);
    plain.build(g, &pool);
    const HubnessStats a = plain.rescore_all(g, data.dim(), edges, &pool);

    HubnessParams q = p;
    q.min_per_component = 2;
    HubnessIndex quota(q);
    quota.build(g, &pool);
    quota.set_components(&comp, nc);
    const HubnessStats b = quota.rescore_all(g, data.dim(), edges, &pool);

    CHECK_GE(b.selected, a.selected);
    CHECK_EQ(b.budget_selected, a.selected);
    CHECK_EQ(b.covered_components, static_cast<std::size_t>(nc));
    std::printf("  %u components: budget picks %zu, quota picks %zu (+%zu)\n",
                static_cast<unsigned>(nc), a.selected, b.selected, b.quota_selected);
}

}  // namespace

int main() {
    test_degrees_and_scores();
    test_component_quota();
    test_quota_is_a_superset();
    test_score_normalizations();
    test_normalization_selection_agreement();
    test_degenerate_rules();
    test_incremental_matches_batch();
    test_selection_budget();
    return coresg_test::summary("hubness");
}
