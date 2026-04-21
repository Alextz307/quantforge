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

    using IIndicator::compute;  // allocating overload from base

    /// Writes the middle band (SMA) into ``out`` (same size as ``prices``).
    void compute(
        std::span<const double> prices,
        std::span<double> out) const override;

    [[nodiscard]] int warmup_period() const noexcept override;
    [[nodiscard]] std::string name() const override;

    /// Returns upper, middle, and lower bands. Allocating convenience.
    [[nodiscard]] BollingerResult compute_all(std::span<const double> prices) const;

    /// Writes upper/mid/lower bands into ``out``. The three vectors of
    /// ``out`` must each have size ``prices.size()``.
    void compute_all(
        std::span<const double> prices,
        BollingerResult& out) const;

private:
    int period_;
    double num_std_;
};

}  // namespace quant
