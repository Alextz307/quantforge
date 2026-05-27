#include <quant/engine/backtest_engine.hpp>

#include <cmath>
#include <stdexcept>

namespace quant {

namespace {

constexpr double kMinOrderShares = 1e-12;  // below this, treat as no trade

/// Cash flow + new share count produced by one leg's fill at a single bar.
/// ``cash_delta`` is signed (negative for a buy) and includes commission.
struct FillOutcome {
    double new_shares;
    double cash_delta;
    bool traded;
};

[[nodiscard]] FillOutcome try_fill_leg(
    double current_shares,
    double target_shares,
    double price,
    double volume,
    const SlippageConfig& slippage,
    double transaction_fee_rate
) {
    const double delta_shares = target_shares - current_shares;
    if (std::abs(delta_shares) <= kMinOrderShares) {
        return FillOutcome{current_shares, 0.0, false};
    }
    const double fill_price = slippage.apply(price, delta_shares, volume);
    const double trade_notional = delta_shares * fill_price;
    const double commission = std::abs(trade_notional) * transaction_fee_rate;
    return FillOutcome{target_shares, -(trade_notional + commission), true};
}

[[nodiscard]] double size_target_shares(
    double target_leverage,
    double pre_fill_equity,
    double price,
    double current_shares
) {
    if (price <= 0.0) {
        return current_shares;
    }
    const double target_notional = target_leverage * pre_fill_equity;
    return target_notional / price;
}

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

    double equity = cash;
    out.equity_curve.push_back(equity);

    for (size_t i = 1; i < n; ++i) {
        const double raw_signal = signals[i - 1];
        double target_leverage = std::isnan(raw_signal) ? 0.0 : raw_signal;
        if (!allow_short && target_leverage < 0.0) {
            target_leverage = 0.0;
        }

        // Carry forward the previous iteration's equity tail instead of
        // recomputing ``cash + shares * bars[i - 1].close``.
        const double pre_fill_equity = equity;
        const double theoretical_price = bars[i].open;
        const double target_shares = size_target_shares(
            target_leverage, pre_fill_equity, theoretical_price, shares);
        const FillOutcome fill = try_fill_leg(
            shares, target_shares, theoretical_price, bars[i].volume,
            slippage_override, config_.transaction_fee_rate);
        if (fill.traded) {
            cash += fill.cash_delta;
            shares = fill.new_shares;
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

BacktestResult BacktestEngine::run_pairs(
    std::span<const Bar> bars_a,
    std::span<const Bar> bars_b,
    std::span<const double> signals,
    double hedge_ratio
) const {
    return run_pairs(bars_a, bars_b, signals, hedge_ratio, config_.slippage);
}

BacktestResult BacktestEngine::run_pairs(
    std::span<const Bar> bars_a,
    std::span<const Bar> bars_b,
    std::span<const double> signals,
    double hedge_ratio,
    const SlippageConfig& slippage_override
) const {
    BacktestResult out;
    run_pairs(bars_a, bars_b, signals, hedge_ratio, slippage_override, out);
    return out;
}

void BacktestEngine::run_pairs(
    std::span<const Bar> bars_a,
    std::span<const Bar> bars_b,
    std::span<const double> signals,
    double hedge_ratio,
    BacktestResult& out
) const {
    run_pairs(bars_a, bars_b, signals, hedge_ratio, config_.slippage, out);
}

void BacktestEngine::run_pairs(
    std::span<const Bar> bars_a,
    std::span<const Bar> bars_b,
    std::span<const double> signals,
    double hedge_ratio,
    const SlippageConfig& slippage_override,
    BacktestResult& out
) const {
    if (bars_a.size() != bars_b.size() || bars_a.size() != signals.size()) {
        throw std::invalid_argument(
            "BacktestEngine::run_pairs: bars_a, bars_b, and signals must "
            "have the same length"
        );
    }

    const size_t n = bars_a.size();
    auto capacity_saver = std::move(out.equity_curve);
    capacity_saver.clear();
    out = BacktestResult{};
    out.equity_curve = std::move(capacity_saver);
    out.equity_curve.reserve(n);

    if (n == 0) {
        return;
    }

    double cash = config_.initial_capital;
    double shares_a = 0.0;
    double shares_b = 0.0;
    int trade_count = 0;

    double equity = cash;
    out.equity_curve.push_back(equity);

    for (size_t i = 1; i < n; ++i) {
        const double raw_signal = signals[i - 1];
        // allow_short is intentionally ignored: a pairs trade is
        // dollar-neutral by construction (long one leg, short the other).
        const double leg_a_target = std::isnan(raw_signal) ? 0.0 : raw_signal;
        const double leg_b_target = -hedge_ratio * leg_a_target;

        const double pre_fill_equity = equity;
        const double price_a = bars_a[i].open;
        const double price_b = bars_b[i].open;
        const double target_shares_a = size_target_shares(
            leg_a_target, pre_fill_equity, price_a, shares_a);
        const double target_shares_b = size_target_shares(
            leg_b_target, pre_fill_equity, price_b, shares_b);

        const FillOutcome fill_a = try_fill_leg(
            shares_a, target_shares_a, price_a, bars_a[i].volume,
            slippage_override, config_.transaction_fee_rate);
        const FillOutcome fill_b = try_fill_leg(
            shares_b, target_shares_b, price_b, bars_b[i].volume,
            slippage_override, config_.transaction_fee_rate);

        if (fill_a.traded) {
            cash += fill_a.cash_delta;
            shares_a = fill_a.new_shares;
        }
        if (fill_b.traded) {
            cash += fill_b.cash_delta;
            shares_b = fill_b.new_shares;
        }
        if (fill_a.traded || fill_b.traded) {
            ++trade_count;
        }

        equity = cash + shares_a * bars_a[i].close + shares_b * bars_b[i].close;
        out.equity_curve.push_back(equity);
    }

    const double final_equity = out.equity_curve.back();
    out.total_return = (config_.initial_capital > 0.0)
        ? (final_equity / config_.initial_capital) - 1.0
        : 0.0;
    out.trade_count = trade_count;
}

}  // namespace quant
