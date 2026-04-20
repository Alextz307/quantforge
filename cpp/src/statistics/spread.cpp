#include "quant/statistics/spread.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "quant/indicators/detail/rolling.hpp"

namespace quant::statistics {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

// TODO(Phase 6): allocate output buffers from a caller-provided arena or
// out-param to eliminate per-call heap allocation in walk-forward loops.
std::vector<double> SpreadCalculator::compute_spread(
    std::span<const double> a,
    std::span<const double> b,
    double hedge_ratio)
{
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "SpreadCalculator::compute_spread: a and b must have the same length");
    }
    const auto n = a.size();
    std::vector<double> out(n);
    // TODO(Phase 6): verify auto-vectorization via -Rpass=loop-vectorize;
    // straight-line fp64 fma is a prime SIMD target.
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a[i] - hedge_ratio * b[i];
    }
    return out;
}

// TODO(Phase 6): fuse rolling_mean + rolling_std + z-score into a single
// Welford pass to eliminate two temporary vectors.
// NaN semantics diverge from pandas — see the header docstring.
std::vector<double> SpreadCalculator::compute_zscore(
    std::span<const double> spread,
    int window)
{
    if (window < 2) {
        throw std::invalid_argument(
            "SpreadCalculator::compute_zscore: window must be >= 2");
    }
    const auto n = spread.size();
    std::vector<double> out(n, kNaN);
    if (static_cast<int>(n) < window) {
        return out;
    }

    const auto mean = detail::rolling_mean(spread, window);
    const auto std_dev = detail::rolling_std(spread, window, /*ddof=*/1);
    const auto start = static_cast<std::size_t>(window - 1);
    for (std::size_t i = start; i < n; ++i) {
        if (std_dev[i] > 0.0) {
            out[i] = (spread[i] - mean[i]) / std_dev[i];
        }
    }
    return out;
}

}  // namespace quant::statistics
