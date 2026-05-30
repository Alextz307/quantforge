#pragma once

// Independent reference composition of the per-metric helpers - gives the
// parity tests a fixed point to compare MetricsCalculator::compute
// against.

#include <cmath>
#include <span>

#include "quant/metrics/performance.hpp"

namespace quant::tests::detail {

inline PerformanceMetrics metrics_compute_reference(
    std::span<const double> equity_curve,
    int annualization_factor,
    double risk_free_rate = 0.0)
{
    PerformanceMetrics metrics;
    if (equity_curve.size() < 2) return metrics;
    const auto returns = MetricsCalculator::equity_to_returns(equity_curve);
    const std::span<const double> rs{returns};

    metrics.annualized_return =
        MetricsCalculator::annualized_return(equity_curve, annualization_factor);
    metrics.annualized_volatility =
        MetricsCalculator::annualized_volatility(rs, annualization_factor);
    metrics.sharpe_ratio =
        MetricsCalculator::sharpe_ratio(rs, annualization_factor, risk_free_rate);
    metrics.sortino_ratio =
        MetricsCalculator::sortino_ratio(rs, annualization_factor, risk_free_rate);
    metrics.max_drawdown = MetricsCalculator::max_drawdown(equity_curve);
    metrics.win_rate = MetricsCalculator::win_rate(rs);
    metrics.calmar_ratio = (metrics.max_drawdown < 0.0)
        ? metrics.annualized_return / std::abs(metrics.max_drawdown)
        : 0.0;
    return metrics;
}

}  // namespace quant::tests::detail
