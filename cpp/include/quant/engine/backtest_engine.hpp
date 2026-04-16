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
    // TODO(Phase 6): offer a result-buffer-taking overload so HPO sweeps can
    // reuse a scratch BacktestResult (or a separate equity_curve buffer) across
    // thousands of run() calls instead of allocating a fresh N-element vector
    // per scenario. Mirrors the same Phase 6 candidate on
    // MetricsCalculator::equity_to_returns.
    [[nodiscard]] BacktestResult run(
        std::span<const Bar> bars,
        std::span<const double> signals
    ) const;

    /// Run with a slippage override — swaps in the given model instead of
    /// `config_.slippage`. Lets a single engine drive scenario sweeps without
    /// reconstructing it per scenario.
    [[nodiscard]] BacktestResult run(
        std::span<const Bar> bars,
        std::span<const double> signals,
        const SlippageConfig& slippage_override
    ) const;

private:
    Config config_;
};

}  // namespace quant
