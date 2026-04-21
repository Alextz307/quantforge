#include "quant/strategies/pairs_trading.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "quant/core/validation.hpp"
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

std::vector<double> PairsTradingStrategy::generate_signals(
    std::span<const double> prices_a,
    std::span<const double> prices_b,
    const statistics::CointegrationParams& coint) const
{
    std::vector<double> out(prices_a.size());
    Buffer scratch;
    generate_signals(prices_a, prices_b, coint, out, scratch);
    return out;
}

void PairsTradingStrategy::generate_signals(
    std::span<const double> prices_a,
    std::span<const double> prices_b,
    const statistics::CointegrationParams& coint,
    std::span<double> out) const
{
    Buffer scratch;
    generate_signals(prices_a, prices_b, coint, out, scratch);
}

void PairsTradingStrategy::generate_signals(
    std::span<const double> prices_a,
    std::span<const double> prices_b,
    const statistics::CointegrationParams& coint,
    std::span<double> out,
    Buffer& scratch) const
{
    if (prices_a.size() != prices_b.size()) {
        throw std::invalid_argument(
            "PairsTradingStrategy::generate_signals: prices_a and prices_b "
            "must have the same length");
    }
    detail::check_out_size(
        prices_a.size(), out.size(), "PairsTradingStrategy::generate_signals");

    const auto n = prices_a.size();
    scratch.spread.resize(n);
    scratch.zscore.resize(n);
    statistics::SpreadCalculator::compute_spread(
        prices_a, prices_b, coint.hedge_ratio, scratch.spread);
    statistics::SpreadCalculator::compute_zscore(
        scratch.spread, config_.zscore_lookback, scratch.zscore);
    run_pairs_state_machine(
        scratch.zscore, config_.entry_zscore, config_.exit_zscore,
        config_.stop_loss_zscore, out);
}

std::string PairsTradingStrategy::name() const {
    return "PairsTrading";
}

int PairsTradingStrategy::required_warmup() const noexcept {
    return config_.zscore_lookback;
}

}  // namespace quant::strategies
