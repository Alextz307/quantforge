#include "quant/strategies/adaptive_bollinger.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

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

// TODO(Phase 6): allocate output buffers from a caller-provided arena or
// out-param to eliminate per-call heap allocation in walk-forward loops.
std::vector<double> AdaptiveBollingerStrategy::generate_signals(
    std::span<const double> close,
    std::span<const double> cond_vol) const
{
    if (close.size() != cond_vol.size()) {
        throw std::invalid_argument(
            "AdaptiveBollingerStrategy::generate_signals: close and cond_vol "
            "must have the same length");
    }
    const auto n = close.size();
    const auto mid = detail::rolling_mean(close, config_.band_window);
    const auto trend_ma = detail::rolling_mean(close, config_.trend_window);

    std::vector<double> upper(n, kNaN);
    std::vector<double> lower(n, kNaN);
    for (std::size_t i = 0; i < n; ++i) {
        const double m = mid[i];
        const double v = cond_vol[i];
        if (!std::isnan(m) && !std::isnan(v)) {
            const double half = config_.k * v;
            upper[i] = m + half;
            lower[i] = m - half;
        }
    }
    return run_mean_reversion_state_machine(close, mid, upper, lower, trend_ma);
}

std::string AdaptiveBollingerStrategy::name() const {
    return "AdaptiveBollinger";
}

int AdaptiveBollingerStrategy::required_warmup() const noexcept {
    return std::max(config_.band_window, config_.trend_window);
}

}  // namespace quant::strategies
