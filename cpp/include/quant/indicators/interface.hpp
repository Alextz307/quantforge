#pragma once

#include <span>
#include <string>
#include <vector>

namespace quant {

/// Abstract base class for single-input indicators.
/// Computes indicator values from a single price array (typically close prices).
/// Output vector has the same length as input, with NaN for warmup values.
class IIndicator {
public:
    virtual ~IIndicator() = default;

    /// Compute indicator values from price data.
    /// @param prices Input price array (e.g., close prices).
    /// @return Vector of same length as input, NaN-padded for warmup period.
    /// TODO(Phase 6): Add an output-buffer overload
    ///   void compute(span<const double> prices, span<double> output) const
    /// to let callers pre-allocate once and reuse across calls in the backtest loop.
    [[nodiscard]] virtual std::vector<double> compute(
        std::span<const double> prices) const = 0;

    /// Minimum number of bars before output is valid.
    [[nodiscard]] virtual int warmup_period() const noexcept = 0;

    /// Human-readable indicator name (e.g., "RSI(14)").
    [[nodiscard]] virtual std::string name() const = 0;
};

}  // namespace quant
