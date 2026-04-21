#include "quant/strategies/adaptive_bollinger.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include "quant/core/validation.hpp"
#include "quant/indicators/detail/rolling.hpp"
#include "quant/strategies/state_machines.hpp"

namespace quant::strategies {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

AdaptiveBollingerStrategy::AdaptiveBollingerStrategy(Config config)
    : config_(config)
{
    if (config_.band_window < 2) {
        throw std::invalid_argument(
            "AdaptiveBollingerStrategy: band_window must be >= 2");
    }
    if (config_.trend_window < 2) {
        throw std::invalid_argument(
            "AdaptiveBollingerStrategy: trend_window must be >= 2");
    }
    if (config_.k <= 0.0) {
        throw std::invalid_argument(
            "AdaptiveBollingerStrategy: k must be > 0");
    }
}

std::vector<double> AdaptiveBollingerStrategy::generate_signals(
    std::span<const double> close,
    std::span<const double> cond_vol) const
{
    std::vector<double> out(close.size());
    Buffer scratch;
    generate_signals(close, cond_vol, out, scratch);
    return out;
}

void AdaptiveBollingerStrategy::generate_signals(
    std::span<const double> close,
    std::span<const double> cond_vol,
    std::span<double> out) const
{
    Buffer scratch;
    generate_signals(close, cond_vol, out, scratch);
}

void AdaptiveBollingerStrategy::generate_signals(
    std::span<const double> close,
    std::span<const double> cond_vol,
    std::span<double> out,
    Buffer& scratch) const
{
    if (close.size() != cond_vol.size()) {
        throw std::invalid_argument(
            "AdaptiveBollingerStrategy::generate_signals: close and cond_vol "
            "must have the same length");
    }
    detail::check_out_size(
        close.size(), out.size(), "AdaptiveBollingerStrategy::generate_signals");

    const auto n = close.size();
    scratch.mid.resize(n);
    scratch.trend_ma.resize(n);
    scratch.upper.resize(n);
    scratch.lower.resize(n);
    detail::rolling_mean(close, config_.band_window, scratch.mid);
    detail::rolling_mean(close, config_.trend_window, scratch.trend_ma);

    std::fill(scratch.upper.begin(), scratch.upper.end(), kNaN);
    std::fill(scratch.lower.begin(), scratch.lower.end(), kNaN);
    for (std::size_t i = 0; i < n; ++i) {
        const double m = scratch.mid[i];
        const double v = cond_vol[i];
        if (!std::isnan(m) && !std::isnan(v)) {
            const double half = config_.k * v;
            scratch.upper[i] = m + half;
            scratch.lower[i] = m - half;
        }
    }
    run_mean_reversion_state_machine(
        close, scratch.mid, scratch.upper, scratch.lower, scratch.trend_ma, out);
}

std::string AdaptiveBollingerStrategy::name() const {
    return "AdaptiveBollinger";
}

int AdaptiveBollingerStrategy::required_warmup() const noexcept {
    return std::max(config_.band_window, config_.trend_window);
}

}  // namespace quant::strategies
