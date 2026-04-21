#pragma once

#include <string>
#include <vector>

#include "quant/indicators/interface.hpp"

namespace quant {

/// Full MACD output: line, signal, and histogram.
struct MACDResult {
    std::vector<double> macd_line;
    std::vector<double> signal_line;
    std::vector<double> histogram;
};

/// Moving Average Convergence Divergence indicator.
/// compute() returns the MACD line; compute_all() returns all three outputs.
class MACD final : public IIndicator {
public:
    explicit MACD(int fast_period = 12, int slow_period = 26, int signal_period = 9);

    using IIndicator::compute;  // allocating overload from base

    /// Writes MACD line into ``out`` (same size as ``prices``).
    void compute(
        std::span<const double> prices,
        std::span<double> out) const override;

    [[nodiscard]] int warmup_period() const noexcept override;
    [[nodiscard]] std::string name() const override;

    /// Returns MACD line, signal line, and histogram. Allocating convenience.
    [[nodiscard]] MACDResult compute_all(std::span<const double> prices) const;

    /// Writes MACD line, signal line, and histogram into ``out``. The three
    /// vectors of ``out`` must each have size ``prices.size()``; caller
    /// reuses the same ``MACDResult`` across calls to amortize allocation.
    void compute_all(
        std::span<const double> prices,
        MACDResult& out) const;

private:
    int fast_period_;
    int slow_period_;
    int signal_period_;
};

}  // namespace quant
