#pragma once

#include <string>

#include "quant/indicators/interface.hpp"

namespace quant {

/// Relative Strength Index using Wilder's smoothing.
/// Single-pass O(n) algorithm. Warmup period = period.
class RSI final : public IIndicator {
public:
    explicit RSI(int period = 14);

    using IIndicator::compute;  // allocating overload from base

    void compute(
        std::span<const double> prices,
        std::span<double> out) const override;

    [[nodiscard]] int warmup_period() const noexcept override;
    [[nodiscard]] std::string name() const override;

private:
    int period_;
};

}  // namespace quant
