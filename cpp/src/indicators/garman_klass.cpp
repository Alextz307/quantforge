#include "quant/indicators/garman_klass.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "quant/indicators/detail/volatility_utils.hpp"

namespace quant {

namespace {
// 2 * ln(2) - 1 ≈ 0.3863
constexpr double kGKCoeff = 2.0 * std::numbers::ln2 - 1.0;
}  // namespace

GarmanKlass::GarmanKlass(int window) : window_(window) {
    if (window < 1) {
        throw std::invalid_argument("GarmanKlass window must be >= 1, got "
                                    + std::to_string(window));
    }
}

std::vector<double> GarmanKlass::compute(
    std::span<const double> open,
    std::span<const double> high,
    std::span<const double> low,
    std::span<const double> close) const
{
    const auto n = static_cast<int>(open.size());
    detail::validate_ohlc_lengths(open, high, low, close, "GarmanKlass");

    if (n == 0) return {};

    detail::validate_ohlc_prices(open, high, low, close, "GarmanKlass");

    // Per-bar Garman-Klass variance proxy.
    // Note: individual gk_daily[i] can be negative when the close-to-open
    // movement dominates the high-low range (0.5*ln(H/L)^2 < (2ln2-1)*ln(C/O)^2).
    // The rolling mean over a reasonable window is almost always positive with
    // real market data. If it is negative, annualize_rolling_variance clamps to 0
    // via max(0, ...) before sqrt — this is conservative but avoids NaN propagation.
    std::vector<double> gk_daily(open.size());
    for (int i = 0; i < n; ++i) {
        double log_hl = std::log(high[i] / low[i]);
        double log_co = std::log(close[i] / open[i]);
        gk_daily[i] = 0.5 * log_hl * log_hl - kGKCoeff * log_co * log_co;
    }

    return detail::annualize_rolling_variance(gk_daily, window_);
}

int GarmanKlass::warmup_period() const noexcept {
    return window_ - 1;
}

std::string GarmanKlass::name() const {
    return "GarmanKlass(" + std::to_string(window_) + ")";
}

}  // namespace quant
