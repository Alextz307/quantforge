#pragma once

#include <quant/core/types.hpp>
#include <quant/engine/slippage.hpp>

#include <span>

namespace quant {

/// Event-driven bar-level backtest engine.
///
/// Signals at bar t determine the target leverage held from bar t+1's open
/// onward. Fills happen at bars[t+1].open with slippage applied per
/// SlippageConfig. Commission is charged on |Δposition_notional| *
/// transaction_fee_rate.
///
/// Conventions:
///   - NaN signals map to 0 (explicitly flat; matches strategy warmup intent).
///   - allow_short=false clips negative targets to 0 before sizing the fill.
///   - equity_curve[0] == initial_capital (no fill possible on the first bar).
///   - Fills are sized against the previous bar's close mark-to-market equity,
///     using bars[t+1].open as the sizing price (actual fill price, after
///     slippage, may differ slightly — standard zipline/backtrader convention).
///
/// Metrics boundary: this engine populates `equity_curve`, `total_return`, and
/// `trade_count`. Statistical metrics (sharpe / sortino / max_drawdown /
/// win_rate / annualized_*) are filled by MetricsCalculator (see
/// quant/metrics/performance.hpp) — default-zero here.
class BacktestEngine final {
public:
    struct Config {
        double initial_capital{10000.0};
        double transaction_fee_rate{0.001};
        SlippageConfig slippage{};
        bool allow_short{true};
    };

    explicit BacktestEngine(Config config) noexcept : config_{config} {}

    /// Throws std::invalid_argument when bars.size() != signals.size().
    [[nodiscard]] BacktestResult run(
        std::span<const Bar> bars,
        std::span<const double> signals
    ) const;

private:
    Config config_;
};

}  // namespace quant
