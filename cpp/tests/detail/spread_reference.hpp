#pragma once

// Independent reference implementation of spread z-score - composes
// rolling_mean + rolling_std + a divide pass. Exists only as a parity
// fixed point for the production kernel.

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "quant/indicators/detail/rolling.hpp"

namespace quant::tests::detail {

inline std::vector<double> compute_zscore_reference(
    std::span<const double> spread,
    int window)
{
    if (window < 2) {
        throw std::invalid_argument(
            "compute_zscore_reference: window must be >= 2");
    }
    const auto n = spread.size();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> out(n, nan);
    if (static_cast<int>(n) < window) return out;

    const auto mean = quant::detail::rolling_mean(spread, window);
    const auto std_dev = quant::detail::rolling_std(spread, window, /*ddof=*/1);
    const auto start = static_cast<std::size_t>(window - 1);
    for (std::size_t i = start; i < n; ++i) {
        if (std_dev[i] > 0.0) {
            out[i] = (spread[i] - mean[i]) / std_dev[i];
        }
    }
    return out;
}

}  // namespace quant::tests::detail
