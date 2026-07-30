#include "coresg/intrinsic_dim.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <vector>

#include "coresg/thread_pool.hpp"

namespace coresg {

const char* to_string(BudgetDimension b) {
    switch (b) {
        case BudgetDimension::Ambient: return "ambient";
        case BudgetDimension::Intrinsic: return "intrinsic";
    }
    return "?";
}

bool budget_dimension_from_string(const std::string& s, BudgetDimension* out) {
    if (s == "ambient") { *out = BudgetDimension::Ambient; return true; }
    if (s == "intrinsic" || s == "lid") { *out = BudgetDimension::Intrinsic; return true; }
    return false;
}

IntrinsicDimStats estimate_intrinsic_dimension(const KnnGraph& g, std::uint32_t k,
                                               ThreadPool* pool, std::uint32_t ambient_dim) {
    const auto t0 = std::chrono::steady_clock::now();
    IntrinsicDimStats st;
    const idx_t n = g.size();
    if (n == 0 || g.k() < 3) return st;

    if (k == 0) k = std::min(g.k(), kDefaultIdK);
    k = std::min(k, g.k());
    if (k < 3) k = std::min<std::uint32_t>(3, g.k());
    st.k_used = k;

    std::vector<double> local(n, 0.0);
    std::vector<std::uint8_t> ok(n, 0);

    auto body = [&](std::size_t lo, std::size_t hi, int) {
        for (std::size_t i = lo; i < hi; ++i) {
            const idx_t v = static_cast<idx_t>(i);
            const std::uint32_t deg = std::min(k, g.degree(v));
            if (deg < 3) continue;
            const dist_t* r = g.dists(v);
            const double rk = static_cast<double>(r[deg - 1]);
            // A zero outer radius means duplicates fill the neighbourhood;
            // there is no scale to measure and the log would diverge. Real
            // data hits this (integer-valued SIFT, near-duplicate documents),
            // so it is skipped rather than clamped.
            if (!(rk > 0.0)) continue;

            double sum = 0.0;
            std::uint32_t count = 0;
            for (std::uint32_t j = 0; j + 1 < deg; ++j) {
                const double rj = static_cast<double>(r[j]);
                if (!(rj > 0.0)) continue;  // duplicate of v itself
                sum += std::log(rk / rj);
                ++count;
            }
            if (count == 0 || !(sum > 0.0)) continue;  // all radii identical
            double m = static_cast<double>(count) / sum;
            // An intrinsic dimensionality above the ambient one is impossible;
            // values above it come from collapsed neighbourhoods, so clamp
            // before aggregating instead of letting them skew the summaries.
            if (ambient_dim > 0) m = std::min(m, static_cast<double>(ambient_dim));
            if (!(m >= 1.0)) m = 1.0;
            local[v] = m;
            ok[v] = 1;
        }
    };
    if (pool && pool->size() > 1) {
        pool->parallel_ranges(n, body);
    } else {
        body(0, n, 0);
    }

    std::vector<double> good;
    good.reserve(n);
    double sum = 0.0, inv_sum = 0.0;
    for (idx_t v = 0; v < n; ++v) {
        if (!ok[v]) continue;
        good.push_back(local[v]);
        sum += local[v];
        inv_sum += 1.0 / local[v];
    }
    st.usable = good.size();
    st.degenerate = static_cast<std::size_t>(n) - good.size();
    if (good.empty()) return st;

    std::sort(good.begin(), good.end());
    auto quantile = [&](double f) {
        const std::size_t idx =
            std::min(good.size() - 1, static_cast<std::size_t>(f * double(good.size())));
        return good[idx];
    };
    st.mean = sum / double(good.size());
    st.median = quantile(0.50);
    st.harmonic = double(good.size()) / inv_sum;
    st.p10 = quantile(0.10);
    st.p90 = quantile(0.90);
    st.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

std::string format_intrinsic_dim(const IntrinsicDimStats& s) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << "ID " << s.value() << " [harmonic] (median " << s.median << ", mean " << s.mean
        << ", p10-p90 " << s.p10 << "-" << s.p90 << "), k=" << s.k_used << ", "
        << s.usable_fraction() * 100.0 << "% usable"
        << (s.reliable() ? "" : "  <-- UNRELIABLE, fall back to ambient");
    return oss.str();
}

}  // namespace coresg
