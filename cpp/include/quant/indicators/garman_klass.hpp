#pragma once

#include <string>

#include "quant/indicators/volatility_estimator.hpp"

namespace quant {

/// Garman-Klass range-based volatility estimator.
/// More efficient than close-to-close volatility.
/// Formula: GK_daily[t] = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
/// Output: annualized = sqrt(rolling_mean(GK_daily, window)) * sqrt(252)
class GarmanKlass final : public IVolatilityEstimator {
public:
    explicit GarmanKlass(int window = 22);

    [[nodiscard]] std::vector<double> compute(
        std::span<const double> open,
        std::span<const double> high,
        std::span<const double> low,
        std::span<const double> close) const override;

    [[nodiscard]] int warmup_period() const noexcept override;
    [[nodiscard]] std::string name() const override;

private:
    int window_;
};

}  // namespace quant
