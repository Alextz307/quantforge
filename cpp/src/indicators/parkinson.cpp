#include "quant/indicators/parkinson.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "quant/indicators/detail/volatility_utils.hpp"

namespace quant {

namespace {
// 1 / (4 * ln(2)) ≈ 0.3607
constexpr double kPKCoeff = 1.0 / (4.0 * std::numbers::ln2);
}  // namespace

Parkinson::Parkinson(int window) : window_(window) {
    if (window < 1) {
        throw std::invalid_argument("Parkinson window must be >= 1, got "
                                    + std::to_string(window));
    }
}

std::vector<double> Parkinson::compute(
    std::span<const double> open,
    std::span<const double> high,
    std::span<const double> low,
    std::span<const double> close) const
{
    const auto n = static_cast<int>(high.size());
    detail::validate_ohlc_lengths(open, high, low, close, "Parkinson");

    if (n == 0) return {};

    detail::validate_ohlc_prices(open, high, low, close, "Parkinson");

    // Per-bar Parkinson variance proxy (uses only high/low)
    std::vector<double> pk_daily(high.size());
    for (int i = 0; i < n; ++i) {
        double log_hl = std::log(high[i] / low[i]);
        pk_daily[i] = kPKCoeff * log_hl * log_hl;
    }

    return detail::annualize_rolling_variance(pk_daily, window_);
}

int Parkinson::warmup_period() const noexcept {
    return window_ - 1;
}

std::string Parkinson::name() const {
    return "Parkinson(" + std::to_string(window_) + ")";
}

}  // namespace quant
