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
    return run(bars, signals, config_.slippage);
}

BacktestResult BacktestEngine::run(
    std::span<const Bar> bars,
    std::span<const double> signals,
    const SlippageConfig& slippage_override
) const {
    BacktestResult out;
    run(bars, signals, slippage_override, out);
    return out;
}

void BacktestEngine::run(
    std::span<const Bar> bars,
    std::span<const double> signals,
    BacktestResult& out
) const {
    run(bars, signals, config_.slippage, out);
}

void BacktestEngine::run(
    std::span<const Bar> bars,
    std::span<const double> signals,
    const SlippageConfig& slippage_override,
    BacktestResult& out
) const {
    if (bars.size() != signals.size()) {
        throw std::invalid_argument(
            "BacktestEngine::run: bars.size() must equal signals.size()"
        );
    }

    const size_t n = bars.size();
    // Preserve equity_curve capacity across scenarios by moving it out, then
    // restoring after a default-reset of ``out``. Any new BacktestResult
    // field gets reset for free via this swap, so the buffer-reuse path
    // can't silently drift out of sync with the struct's defaults.
    auto capacity_saver = std::move(out.equity_curve);
    capacity_saver.clear();
    out = BacktestResult{};
    out.equity_curve = std::move(capacity_saver);
    out.equity_curve.reserve(n);

    if (n == 0) {
        return;
    }

    const bool allow_short = config_.allow_short;
    double cash = config_.initial_capital;
    double shares = 0.0;
    int trade_count = 0;

    // Bar 0: no prior signal → no fill. Equity = starting cash.
    double equity = cash;
    out.equity_curve.push_back(equity);

    for (size_t i = 1; i < n; ++i) {
        const double raw_signal = signals[i - 1];
        double target_leverage = std::isnan(raw_signal) ? 0.0 : raw_signal;
        if (!allow_short && target_leverage < 0.0) {
            target_leverage = 0.0;
        }

        // Carry forward the previous iteration's equity_curve tail instead
        // of recomputing ``cash + shares * bars[i - 1].close``.
        const double pre_fill_equity = equity;
        const double theoretical_price = bars[i].open;
        const double target_notional = target_leverage * pre_fill_equity;
        const double target_shares = (theoretical_price > 0.0)
            ? target_notional / theoretical_price
            : shares;
        const double delta_shares = target_shares - shares;

        if (std::abs(delta_shares) > kMinOrderShares) {
            const double fill_price = slippage_override.apply(
                theoretical_price, delta_shares, bars[i].volume);
            const double trade_notional = delta_shares * fill_price;
            const double commission =
                std::abs(trade_notional) * config_.transaction_fee_rate;
            cash -= trade_notional + commission;
            shares = target_shares;
            ++trade_count;
        }

        equity = cash + shares * bars[i].close;
        out.equity_curve.push_back(equity);
    }

    const double final_equity = out.equity_curve.back();
    out.total_return = (config_.initial_capital > 0.0)
        ? (final_equity / config_.initial_capital) - 1.0
        : 0.0;
    out.trade_count = trade_count;
}

}  // namespace quant
