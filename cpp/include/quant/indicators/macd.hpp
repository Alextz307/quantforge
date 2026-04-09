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

    /// Returns the MACD line only.
    [[nodiscard]] std::vector<double> compute(
        std::span<const double> prices) const override;

    [[nodiscard]] int warmup_period() const noexcept override;
    [[nodiscard]] std::string name() const override;

    /// Returns MACD line, signal line, and histogram.
    [[nodiscard]] MACDResult compute_all(std::span<const double> prices) const;

private:
    int fast_period_;
    int slow_period_;
    int signal_period_;
};

}  // namespace quant
