#include <quant/engine/backtest_engine.hpp>

#include <cmath>
#include <stdexcept>

namespace quant {

namespace {

constexpr double kMinOrderShares = 1e-12;  // below this, treat as no trade

}  // namespace

BacktestResult BacktestEngine::run(
    std::span<const Bar> bars,
    std::span<const double> signals
) const {
    if (bars.size() != signals.size()) {
        throw std::invalid_argument(
            "BacktestEngine::run: bars.size() must equal signals.size()"
        );
    }

    const size_t n = bars.size();
    BacktestResult result;
    result.equity_curve.reserve(n);

    double cash = config_.initial_capital;
    double shares = 0.0;
    int trade_count = 0;

    if (n == 0) {
        return result;  // BacktestResult is default-initialized (zeros)
    }

    // Bar 0: no prior signal → no fill. Equity = starting cash.
    result.equity_curve.push_back(cash);

    // TODO(Phase 6): hoist config_.allow_short into a loop-local bool to avoid
    // per-iteration field access (tiny, but the branch on line ~44 is the most
    // predictable candidate in the hot path).
    // TODO(Phase 6): when no trade fires, pre_fill_equity below recomputes
    // `cash + shares * bars[i-1].close` — the same expression the previous
    // iteration used to build equity_curve[i-1]. Carry it across instead.
    // TODO(Phase 6): profile the switch inside SlippageConfig::apply. If
    // production runs rarely change slippage mid-backtest, a specialized
    // template instantiation or branch hint could cut one indirection per trade.
    for (size_t i = 1; i < n; ++i) {
        const double raw_signal = signals[i - 1];
        double target_leverage = std::isnan(raw_signal) ? 0.0 : raw_signal;
        if (!config_.allow_short && target_leverage < 0.0) {
            target_leverage = 0.0;
        }

        const double pre_fill_equity = cash + shares * bars[i - 1].close;
        const double theoretical_price = bars[i].open;
        const double target_notional = target_leverage * pre_fill_equity;
        const double target_shares = (theoretical_price > 0.0)
            ? target_notional / theoretical_price
            : shares;
        const double delta_shares = target_shares - shares;

        if (std::abs(delta_shares) > kMinOrderShares) {
            const double fill_price = config_.slippage.apply(
                theoretical_price, delta_shares, bars[i].volume);
            const double trade_notional = delta_shares * fill_price;
            const double commission =
                std::abs(trade_notional) * config_.transaction_fee_rate;
            cash -= trade_notional + commission;
            shares = target_shares;
            ++trade_count;
        }

        const double equity = cash + shares * bars[i].close;
        result.equity_curve.push_back(equity);
    }

    const double final_equity = result.equity_curve.back();
    result.total_return = (config_.initial_capital > 0.0)
        ? (final_equity / config_.initial_capital) - 1.0
        : 0.0;
    result.trade_count = trade_count;

    return result;
}

}  // namespace quant
