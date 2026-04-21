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

    /// Compute indicator values into a caller-owned buffer. ``out.size()``
    /// must equal ``prices.size()``; leading warmup slots are filled with NaN.
    /// Derived classes implement this; the allocating convenience below
    /// forwards here so inference loops can reuse a scratch buffer.
    virtual void compute(
        std::span<const double> prices,
        std::span<double> out) const = 0;

    /// Allocating convenience: size + forward to the out-param overload.
    /// Kept non-virtual so there is a single implementation path.
    [[nodiscard]] std::vector<double> compute(
        std::span<const double> prices) const {
        std::vector<double> out(prices.size());
        compute(prices, out);
        return out;
    }

    /// Minimum number of bars before output is valid.
    [[nodiscard]] virtual int warmup_period() const noexcept = 0;

    /// Human-readable indicator name (e.g., "RSI(14)").
    [[nodiscard]] virtual std::string name() const = 0;
};

}  // namespace quant
