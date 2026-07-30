#pragma once
// Distance metrics.
//
// Two things matter for the rest of the pipeline:
//
//  1. The value stored in the k-NN graph must be the *true* metric value, not
//     a monotone surrogate. Mutual reachability distances are compared against
//     core distances coming from possibly different code paths (CPU merge of a
//     GPU-built instance, for example), so the units have to agree everywhere.
//     Search kernels are still free to rank with a surrogate internally and
//     call `finalize()` once at the end -- that is what the CUDA backend does.
//
//  2. Density estimates only make sense for a dissimilarity that is >= 0 and
//     grows with separation. L2/L1/cosine qualify; raw inner product does not,
//     and is only offered for k-NN experiments (`requires_metric()` reports
//     this, and CoreSG refuses to build with it).

#include <cmath>
#include <cstdint>
#include <string>

#if defined(__AVX2__)
#include <immintrin.h>
#define CORESG_HAS_AVX2 1
#else
#define CORESG_HAS_AVX2 0
#endif

#include "coresg/types.hpp"

namespace coresg {

enum class MetricKind : std::uint8_t {
    L2 = 0,          // Euclidean
    Cosine = 1,      // 1 - cos(a,b); vectors are L2-normalised on load
    L1 = 2,          // Manhattan
    InnerProduct = 3 // -<a,b>; not a metric, k-NN experiments only
};

inline const char* metric_name(MetricKind m) {
    switch (m) {
        case MetricKind::L2: return "l2";
        case MetricKind::Cosine: return "cosine";
        case MetricKind::L1: return "l1";
        case MetricKind::InnerProduct: return "ip";
    }
    return "?";
}

inline bool metric_from_string(const std::string& s, MetricKind* out) {
    if (s == "l2" || s == "euclidean") { *out = MetricKind::L2; return true; }
    if (s == "cosine" || s == "angular") { *out = MetricKind::Cosine; return true; }
    if (s == "l1" || s == "manhattan") { *out = MetricKind::L1; return true; }
    if (s == "ip" || s == "dot") { *out = MetricKind::InnerProduct; return true; }
    return false;
}

/// True when the metric is a non-negative dissimilarity, i.e. when mutual
/// reachability distances and therefore CORE-SG are well defined.
inline bool metric_is_dissimilarity(MetricKind m) {
    return m != MetricKind::InnerProduct;
}

/// Whether the data set has to be L2-normalised at load time.
inline bool metric_needs_normalisation(MetricKind m) {
    return m == MetricKind::Cosine;
}

// ---------------------------------------------------------------------------
// Low-level kernels. `*_rank` returns a monotone surrogate (cheap), `finalize`
// converts it to the true metric value.
// ---------------------------------------------------------------------------

inline float l2sqr_scalar(const float* a, const float* b, std::uint32_t d) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    std::uint32_t i = 0;
    for (; i + 4 <= d; i += 4) {
        const float d0 = a[i] - b[i];
        const float d1 = a[i + 1] - b[i + 1];
        const float d2 = a[i + 2] - b[i + 2];
        const float d3 = a[i + 3] - b[i + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    float s = s0 + s1 + s2 + s3;
    for (; i < d; ++i) {
        const float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

inline float dot_scalar(const float* a, const float* b, std::uint32_t d) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    std::uint32_t i = 0;
    for (; i + 4 <= d; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float s = s0 + s1 + s2 + s3;
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
}

inline float l1_scalar(const float* a, const float* b, std::uint32_t d) {
    float s = 0.f;
    for (std::uint32_t i = 0; i < d; ++i) s += std::fabs(a[i] - b[i]);
    return s;
}

#if CORESG_HAS_AVX2
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline float l2sqr_avx2(const float* a, const float* b, std::uint32_t d) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::uint32_t i = 0;
    for (; i + 16 <= d; i += 16) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    for (; i + 8 <= d; i += 8) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    }
    float s = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < d; ++i) {
        const float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

inline float dot_avx2(const float* a, const float* b, std::uint32_t d) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::uint32_t i = 0;
    for (; i + 16 <= d; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= d; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    float s = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
}
#endif  // CORESG_HAS_AVX2

inline float l2sqr(const float* a, const float* b, std::uint32_t d) {
#if CORESG_HAS_AVX2
    return l2sqr_avx2(a, b, d);
#else
    return l2sqr_scalar(a, b, d);
#endif
}

inline float dotp(const float* a, const float* b, std::uint32_t d) {
#if CORESG_HAS_AVX2
    return dot_avx2(a, b, d);
#else
    return dot_scalar(a, b, d);
#endif
}

// ---------------------------------------------------------------------------
// Compile-time metric operators. The NN-Descent inner loops are templated on
// these so the metric switch never reaches the hot path.
// ---------------------------------------------------------------------------

template <MetricKind K>
struct MetricOp;

// `rank` is the cheap monotone surrogate, `finalize` maps it to the stored
// metric value, and `to_rank` maps back. The inverse is what lets a local join
// reject a pair against a stored k-th distance without paying for the
// finalisation (a square root per pair, which dominates in low dimensions).

template <>
struct MetricOp<MetricKind::L2> {
    static constexpr bool rank_is_final = false;
    static float rank(const float* a, const float* b, std::uint32_t d) { return l2sqr(a, b, d); }
    static float finalize(float r) { return std::sqrt(r); }
    static float to_rank(float v) { return v * v; }
};

template <>
struct MetricOp<MetricKind::Cosine> {
    // Data is L2-normalised on load, so ||a-b||^2 = 2 - 2<a,b> and the cosine
    // distance is exactly half the squared Euclidean distance. Ranking with
    // the squared distance is therefore free and exact.
    static constexpr bool rank_is_final = false;
    static float rank(const float* a, const float* b, std::uint32_t d) { return l2sqr(a, b, d); }
    static float finalize(float r) { return 0.5f * r; }
    static float to_rank(float v) { return 2.0f * v; }
};

template <>
struct MetricOp<MetricKind::L1> {
    static constexpr bool rank_is_final = true;
    static float rank(const float* a, const float* b, std::uint32_t d) { return l1_scalar(a, b, d); }
    static float finalize(float r) { return r; }
    static float to_rank(float v) { return v; }
};

template <>
struct MetricOp<MetricKind::InnerProduct> {
    static constexpr bool rank_is_final = true;
    static float rank(const float* a, const float* b, std::uint32_t d) { return -dotp(a, b, d); }
    static float finalize(float r) { return r; }
    static float to_rank(float v) { return v; }
};

/// Runtime -> compile-time dispatch: `dispatch_metric(m, [](auto op){ ... })`.
template <typename F>
auto dispatch_metric(MetricKind m, F&& f) -> decltype(f(MetricOp<MetricKind::L2>{})) {
    switch (m) {
        case MetricKind::L2: return f(MetricOp<MetricKind::L2>{});
        case MetricKind::Cosine: return f(MetricOp<MetricKind::Cosine>{});
        case MetricKind::L1: return f(MetricOp<MetricKind::L1>{});
        case MetricKind::InnerProduct: return f(MetricOp<MetricKind::InnerProduct>{});
    }
    return f(MetricOp<MetricKind::L2>{});
}

/// Convenience for cold paths (evaluation, brute force references).
inline dist_t metric_distance(MetricKind m, const float* a, const float* b, std::uint32_t d) {
    return dispatch_metric(m, [&](auto op) {
        using Op = decltype(op);
        return static_cast<dist_t>(Op::finalize(Op::rank(a, b, d)));
    });
}

}  // namespace coresg
