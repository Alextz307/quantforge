#include "quant/strategies/pairs_trading.hpp"

#include <stdexcept>
#include <string>

#include "quant/statistics/spread.hpp"
#include "quant/strategies/state_machines.hpp"

namespace quant::strategies {

PairsTradingStrategy::PairsTradingStrategy(Config config)
    : config_(config)
{
    if (config_.entry_zscore <= 0.0 || config_.exit_zscore < 0.0
        || config_.stop_loss_zscore <= 0.0) {
        throw std::invalid_argument(
            "PairsTradingStrategy: z-score thresholds must be positive");
    }
    if (config_.exit_zscore >= config_.entry_zscore) {
        throw std::invalid_argument(
            "PairsTradingStrategy: exit_zscore must be < entry_zscore");
    }
    if (config_.stop_loss_zscore <= config_.entry_zscore) {
        throw std::invalid_argument(
            "PairsTradingStrategy: stop_loss_zscore must be > entry_zscore");
    }
    if (config_.zscore_lookback < 2) {
        throw std::invalid_argument(
            "PairsTradingStrategy: zscore_lookback must be >= 2");
    }
}

// TODO(Phase 6): allocate output buffers from a caller-provided arena or
// out-param to eliminate per-call heap allocation in walk-forward loops.
std::vector<double> PairsTradingStrategy::generate_signals(
    std::span<const double> prices_a,
    std::span<const double> prices_b,
    const statistics::CointegrationParams& coint) const
{
    const auto spread = statistics::SpreadCalculator::compute_spread(
        prices_a, prices_b, coint.hedge_ratio);
    const auto zscore = statistics::SpreadCalculator::compute_zscore(
        spread, config_.zscore_lookback);
    return run_pairs_state_machine(
        zscore, config_.entry_zscore, config_.exit_zscore, config_.stop_loss_zscore);
}

std::string PairsTradingStrategy::name() const {
    return "PairsTrading";
}

int PairsTradingStrategy::required_warmup() const noexcept {
    return config_.zscore_lookback;
}

}  // namespace quant::strategies
