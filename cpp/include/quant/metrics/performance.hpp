#pragma once

#include <span>
#include <vector>

namespace quant {

/// Statistical performance metrics populated by `MetricsCalculator::compute`.
/// All metrics use simple returns r[i] = equity[i] / equity[i-1] - 1.
/// Degenerate inputs (empty, single point, zero variance) return 0 rather
/// than NaN so callers can chain without extra guards.
struct PerformanceMetrics {
    double annualized_return{};
    double annualized_volatility{};
    double sharpe_ratio{};
    double sortino_ratio{};
    double max_drawdown{};
    double calmar_ratio{};
    double win_rate{};
};

/// Pure static methods computing performance metrics from an equity curve or
/// a return series. All building blocks are individually exposed for targeted
/// testing and so Python bindings can pick the subset they need.
class MetricsCalculator final {
public:
    /// Compute the full metric suite from an equity curve.
    /// @param equity_curve          Series starting at initial capital.
    /// @param annualization_factor  Bars per year (see annualization_factor()).
    /// @param risk_free_rate        Per-period rf; subtracted from returns.
    [[nodiscard]] static PerformanceMetrics compute(
        std::span<const double> equity_curve,
        int annualization_factor,
        double risk_free_rate = 0.0
    );

    /// Convert an equity curve to simple returns. Length = equity.size() - 1.
    // TODO(Phase 6): offer a buffer-taking overload so HPO sweeps can reuse a
    // scratch vector across thousands of `compute()` calls instead of
    // allocating a fresh return series per scenario.
    [[nodiscard]] static std::vector<double> equity_to_returns(
        std::span<const double> equity_curve);

    [[nodiscard]] static double sharpe_ratio(
        std::span<const double> returns,
        int annualization_factor,
        double risk_free_rate = 0.0
    ) noexcept;

    /// Downside-only Sortino (empyrical convention): denominator is
    /// sqrt(mean(min(0, excess)^2)) over the full return series, not only
    /// the negative subset.
    [[nodiscard]] static double sortino_ratio(
        std::span<const double> returns,
        int annualization_factor,
        double risk_free_rate = 0.0
    ) noexcept;

    /// Peak-to-trough fraction of the equity curve. Zero or negative.
    [[nodiscard]] static double max_drawdown(
        std::span<const double> equity_curve) noexcept;

    /// Fraction of strictly positive returns out of non-zero returns.
    [[nodiscard]] static double win_rate(
        std::span<const double> returns) noexcept;

    /// Geometric annualized return from the first and last equity points.
    [[nodiscard]] static double annualized_return(
        std::span<const double> equity_curve,
        int annualization_factor
    ) noexcept;

    [[nodiscard]] static double annualized_volatility(
        std::span<const double> returns,
        int annualization_factor
    ) noexcept;
};

}  // namespace quant
