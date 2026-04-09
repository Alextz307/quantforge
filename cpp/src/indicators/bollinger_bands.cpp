#include "quant/indicators/bollinger_bands.hpp"

#include <limits>
#include <stdexcept>

#include "quant/indicators/detail/rolling.hpp"

namespace quant {

BollingerBands::BollingerBands(int period, double num_std)
    : period_(period)
    , num_std_(num_std)
{
    if (period < 1) {
        throw std::invalid_argument("BollingerBands period must be >= 1, got "
                                    + std::to_string(period));
    }
    if (num_std < 0.0) {
        throw std::invalid_argument("BollingerBands num_std must be >= 0");
    }
}

// TODO(Phase 6): rolling_mean and rolling_std both iterate the data independently.
// Welford's algorithm already computes both mean and variance in one pass — a fused
// rolling_mean_std() helper would halve the memory traffic for compute_all().
BollingerResult BollingerBands::compute_all(std::span<const double> prices) const {
    auto mid = detail::rolling_mean(prices, period_);
    auto std_dev = detail::rolling_std(prices, period_);

    const auto n = prices.size();
    BollingerResult result;
    result.mid = std::move(mid);
    result.upper.resize(n, std::numeric_limits<double>::quiet_NaN());
    result.lower.resize(n, std::numeric_limits<double>::quiet_NaN());

    for (size_t i = static_cast<size_t>(period_ - 1); i < n; ++i) {
        result.upper[i] = result.mid[i] + num_std_ * std_dev[i];
        result.lower[i] = result.mid[i] - num_std_ * std_dev[i];
    }

    return result;
}

std::vector<double> BollingerBands::compute(std::span<const double> prices) const {
    return detail::rolling_mean(prices, period_);
}

int BollingerBands::warmup_period() const noexcept {
    return period_ - 1;
}

std::string BollingerBands::name() const {
    return "BB(" + std::to_string(period_) + "," + std::to_string(num_std_) + ")";
}

}  // namespace quant
