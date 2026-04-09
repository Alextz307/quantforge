#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "quant/indicators/detail/constants.hpp"
#include "quant/indicators/detail/rolling.hpp"

namespace quant::detail {

/// Shared annualization for rolling variance estimators.
/// Takes per-bar variance proxy values and returns annualized volatility:
///   result[i] = sqrt(max(0, rolling_mean(daily_values, window)[i])) * sqrt(252)
/// First (window - 1) values are NaN. Reuses the vector returned by rolling_mean
/// to avoid an extra allocation.
/// TODO(Phase 6): Callers allocate a full temporary vector (gk_daily/pk_daily)
/// before calling this. Consider a streaming API where per-bar values are
/// computed on the fly during the rolling mean accumulation, eliminating
/// the N-element temporary entirely.
[[nodiscard]] inline std::vector<double> annualize_rolling_variance(
    std::span<const double> daily_values,
    int window)
{
    auto result = rolling_mean(daily_values, window);
    const auto n = static_cast<int>(result.size());

    for (int i = window - 1; i < n; ++i) {
        result[i] = std::sqrt(std::max(0.0, result[i])) * kSqrt252;
    }

    return result;
}

/// Validate that all four OHLC arrays have equal length.
/// @throws std::invalid_argument if lengths differ.
inline void validate_ohlc_lengths(
    std::span<const double> open,
    std::span<const double> high,
    std::span<const double> low,
    std::span<const double> close,
    const std::string& estimator_name)
{
    if (high.size() != open.size() || low.size() != open.size()
        || close.size() != open.size()) {
        throw std::invalid_argument(
            estimator_name + ": all input arrays must have equal length");
    }
}

/// Validate that all OHLC prices are positive and high >= low at every index.
/// @throws std::invalid_argument on the first invalid value.
inline void validate_ohlc_prices(
    std::span<const double> open,
    std::span<const double> high,
    std::span<const double> low,
    std::span<const double> close,
    const std::string& estimator_name)
{
    const auto n = static_cast<int>(open.size());
    for (int i = 0; i < n; ++i) {
        if (open[i] <= 0.0 || high[i] <= 0.0 || low[i] <= 0.0 || close[i] <= 0.0) {
            throw std::invalid_argument(
                estimator_name + ": all prices must be positive, got non-positive value at index "
                + std::to_string(i));
        }
        if (high[i] < low[i]) {
            throw std::invalid_argument(
                estimator_name + ": high must be >= low at index " + std::to_string(i));
        }
    }
}

}  // namespace quant::detail
