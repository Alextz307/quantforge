#include "quant/indicators/garman_klass.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>

#include "quant/core/validation.hpp"
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

void GarmanKlass::compute(
    std::span<const double> open,
    std::span<const double> high,
    std::span<const double> low,
    std::span<const double> close,
    std::span<double> out) const
{
    detail::check_out_size(open.size(), out.size(), "GarmanKlass::compute");
    detail::validate_ohlc_lengths(open, high, low, close, "GarmanKlass");

    const auto n = static_cast<int>(open.size());
    if (n == 0) return;

    detail::validate_ohlc_prices(open, high, low, close, "GarmanKlass");

    // Local N-element temp is unavoidable: the rolling accumulator needs
    // every bar's proxy available at window-slide time. Individual values
    // can be negative when close-to-open dominates the high-low range;
    // annualize_rolling_variance clamps to 0 via max(0, ...) before sqrt.
    std::vector<double> gk_daily(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        double log_hl = std::log(high[i] / low[i]);
        double log_co = std::log(close[i] / open[i]);
        gk_daily[i] = 0.5 * log_hl * log_hl - kGKCoeff * log_co * log_co;
    }
    detail::annualize_rolling_variance(gk_daily, window_, out);
}

int GarmanKlass::warmup_period() const noexcept {
    return window_ - 1;
}

std::string GarmanKlass::name() const {
    return "GarmanKlass(" + std::to_string(window_) + ")";
}

}  // namespace quant
