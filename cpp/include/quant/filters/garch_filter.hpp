#pragma once

#include <span>
#include <vector>

namespace quant::filters {

/// Floor applied to every conditional variance the filter produces; matches
/// the Python reference implementation.
inline constexpr double kVarianceFloor = 1e-12;

/// Fixed parameters produced by a fitted GARCH(p,q) model.
/// `alpha.size() == p`, `beta.size() == q`. `mu` and `backcast` are the
/// training-time sample mean and initial conditional variance, respectively.
/// All values are expressed in the caller's scaled-returns space (typically
/// returns × 100, per arch-library convention).
struct GarchParams {
    double omega = 0.0;
    std::vector<double> alpha;
    std::vector<double> beta;
    double mu = 0.0;
    double backcast;
};

/// Run the GARCH(p,q) recursion on a scaled-returns series, producing the
/// conditional variance path. For indices where `t - i - 1 < 0`, the
/// `backcast` value substitutes for the missing past squared error or
/// past conditional variance. Each variance is floored at 1e-12.
///
/// Mirrors `GARCHPredictor._manual_garch_filter` in the Python codebase;
/// both must agree to 1e-12 rtol on identical inputs.
[[nodiscard]] std::vector<double> garch_filter(
    std::span<const double> scaled_returns,
    const GarchParams& params);

}  // namespace quant::filters
