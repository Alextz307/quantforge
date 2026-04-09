#pragma once

#include <string>
#include <vector>

#include "quant/indicators/interface.hpp"

namespace quant {

/// Full Bollinger Bands output: upper, middle, and lower bands.
struct BollingerResult {
    std::vector<double> upper;
    std::vector<double> mid;
    std::vector<double> lower;
};

/// Bollinger Bands indicator.
/// compute() returns the middle band (SMA); compute_all() returns all three bands.
class BollingerBands final : public IIndicator {
public:
    explicit BollingerBands(int period = 20, double num_std = 2.0);

    /// Returns the middle band (SMA) only.
    [[nodiscard]] std::vector<double> compute(
        std::span<const double> prices) const override;

    [[nodiscard]] int warmup_period() const noexcept override;
    [[nodiscard]] std::string name() const override;

    /// Returns upper, middle, and lower bands.
    [[nodiscard]] BollingerResult compute_all(std::span<const double> prices) const;

private:
    int period_;
    double num_std_;
};

}  // namespace quant
