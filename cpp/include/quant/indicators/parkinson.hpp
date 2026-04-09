#pragma once

#include <string>

#include "quant/indicators/volatility_estimator.hpp"

namespace quant {

/// Parkinson high-low range volatility estimator.
/// Uses only high and low prices (open and close are ignored).
/// Formula: PK_daily[t] = (1 / (4 * ln(2))) * ln(H/L)^2
/// Output: annualized = sqrt(rolling_mean(PK_daily, window)) * sqrt(252)
class Parkinson final : public IVolatilityEstimator {
public:
    explicit Parkinson(int window = 22);

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
