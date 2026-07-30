#pragma once
// Intrinsic dimensionality, estimated from the k-NN graph that already exists.
//
// Why this is here. The anti-hub budget is d * ln(|E|), and `d` was taken to
// be the ambient dimensionality. Measured on real data, that choice breaks:
// the bridge MST costs m^2*d/2 with m proportional to d, so the whole step
// grows as d^3 -- 0.1 G coordinate ops at d=84 but 128 G at d=960 -- and at
// d=1000 with n=11314 the budget exceeds n outright, so the "approximation"
// selects every point and quietly becomes the exact MST it was meant to
// replace.
//
// Ambient dimensionality is also the wrong quantity on its own terms. What the
// budget is trying to express is roughly "how many directions can a bridge
// leave a cluster in", and that is governed by the dimensionality of the
// manifold the data actually occupies, not by the width of the vector it is
// stored in. A 960-column GIST descriptor does not fill 960 independent
// directions.
//
// The estimator. Levina & Bickel's maximum-likelihood estimator (equivalently
// the Hill estimator, and the same quantity the Dynamic NN-Descent paper calls
// LID). For a point x with sorted neighbour radii r_1 <= ... <= r_k:
//
//     m(x) = count / sum_j ln(r_k / r_j)     over j = 1 .. k-1
//
// It needs nothing but the sorted distances the k-NN graph already stores, so
// it costs one O(n*k) pass with no distance computations at all.
//
// Aggregation. The harmonic mean -- averaging 1/m rather than m -- is both the
// theoretically correct pooling (MacKay & Ghahramani showed averaging the
// estimates themselves is biased, averaging their inverses is not) and, as it
// turns out, the only one that survives real data. On TF-IDF text the distances
// concentrate so hard that r_1 is nearly r_k, the denominator collapses towards
// zero, and individual estimates run to 1e7 and beyond; the median inherits
// that, while the harmonic mean is unmoved because an exploded estimate
// contributes ~0 to a sum of inverses. Mean and median are still reported, as
// diagnostics rather than as the answer.
//
// Reliability. Duplicate and all-zero rows (very common in sparse text) leave
// a point with no usable radius at all. When too few points yield an estimate
// the result is flagged unreliable, and the caller is expected to fall back to
// ambient dimensionality rather than trust it.

#include <cstdint>
#include <string>

#include "coresg/knn_graph.hpp"
#include "coresg/types.hpp"

namespace coresg {

class ThreadPool;

struct IntrinsicDimStats {
    double mean = 0.0;       ///< arithmetic mean of the local estimates
    double median = 0.0;     ///< robust default
    double harmonic = 0.0;   ///< MacKay-Ghahramani: averages the inverse
    double p10 = 0.0;
    double p90 = 0.0;
    std::size_t usable = 0;      ///< points that yielded an estimate
    std::size_t degenerate = 0;  ///< skipped: zero radius, or all radii equal
    std::uint32_t k_used = 0;
    double seconds = 0.0;

    /// Fraction of points that produced an estimate.
    double usable_fraction() const {
        const std::size_t total = usable + degenerate;
        return total ? double(usable) / double(total) : 0.0;
    }

    /// Whether the estimate should be trusted. Too many degenerate points
    /// means the neighbourhoods carry no scale information.
    bool reliable(double min_usable_fraction = 0.5) const {
        return usable > 0 && usable_fraction() >= min_usable_fraction && harmonic > 0.0;
    }

    /// The value the budget should use. See the header note on aggregation.
    double value() const { return harmonic > 0.0 ? harmonic : mean; }
};

/// `k` = 0 uses min(graph k, `kDefaultIdK`). The estimator is biased upward for
/// large k (distant neighbours stop being locally uniform) and noisy for very
/// small k, so it is deliberately not tied to k_max.
constexpr std::uint32_t kDefaultIdK = 20;

/// `ambient_dim` > 0 clamps each local estimate to [1, ambient_dim]; an
/// intrinsic dimensionality above the ambient one is impossible by definition,
/// so this bounds the damage from collapsed neighbourhoods before aggregation.
IntrinsicDimStats estimate_intrinsic_dimension(const KnnGraph& g, std::uint32_t k = 0,
                                               ThreadPool* pool = nullptr,
                                               std::uint32_t ambient_dim = 0);

std::string format_intrinsic_dim(const IntrinsicDimStats& s);

/// Which dimensionality term feeds the anti-hub budget d * ln(|E|).
enum class BudgetDimension : std::uint8_t {
    Ambient = 0,    ///< the specified rule: the data's column count
    Intrinsic = 1,  ///< the estimated manifold dimensionality
};

const char* to_string(BudgetDimension b);
bool budget_dimension_from_string(const std::string& s, BudgetDimension* out);

}  // namespace coresg
