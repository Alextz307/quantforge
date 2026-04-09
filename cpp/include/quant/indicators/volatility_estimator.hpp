#pragma once

#include <span>
#include <string>
#include <vector>

namespace quant {

/// Abstract base class for OHLC-based volatility estimators.
/// Unlike IIndicator (single price array), these require all four OHLC arrays.
class IVolatilityEstimator {
public:
    virtual ~IVolatilityEstimator() = default;

    /// Compute annualized volatility from OHLC data.
    /// @return Vector of same length as input, NaN-padded for warmup period.
    [[nodiscard]] virtual std::vector<double> compute(
        std::span<const double> open,
        std::span<const double> high,
        std::span<const double> low,
        std::span<const double> close) const = 0;

    /// Minimum number of bars before output is valid.
    [[nodiscard]] virtual int warmup_period() const noexcept = 0;

    /// Human-readable estimator name.
    [[nodiscard]] virtual std::string name() const = 0;
};

}  // namespace quant
