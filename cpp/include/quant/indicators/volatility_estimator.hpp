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

    /// Compute annualized volatility into a caller-owned buffer. ``out.size()``
    /// must equal ``open.size()``; leading warmup slots are filled with NaN.
    /// Derived classes implement this; the allocating convenience below
    /// forwards here so inference loops can reuse a scratch buffer.
    virtual void compute(
        std::span<const double> open,
        std::span<const double> high,
        std::span<const double> low,
        std::span<const double> close,
        std::span<double> out) const = 0;

    /// Allocating convenience: size + forward to the out-param overload.
    [[nodiscard]] std::vector<double> compute(
        std::span<const double> open,
        std::span<const double> high,
        std::span<const double> low,
        std::span<const double> close) const {
        std::vector<double> out(open.size());
        compute(open, high, low, close, out);
        return out;
    }

    /// Minimum number of bars before output is valid.
    [[nodiscard]] virtual int warmup_period() const noexcept = 0;

    /// Human-readable estimator name.
    [[nodiscard]] virtual std::string name() const = 0;
};

}  // namespace quant
