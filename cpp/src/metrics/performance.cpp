#include <quant/metrics/performance.hpp>

#include <cmath>

namespace quant {

namespace {

// Minimum observations for sample variance (ddof=1). Matches pandas default.
constexpr size_t kMinObsForSampleStd = 2;

struct MeanStd {
    double mean;
    double sample_std;  // ddof=1
};

/// Single-pass Welford mean + sample std (ddof=1). Matches the rolling_std
/// helper in detail/rolling.hpp; avoids catastrophic cancellation from naive
/// sum-of-squares formulas. Returns {0, 0} when n < 2.
[[nodiscard]] MeanStd welford_mean_std(std::span<const double> xs) noexcept {
    const size_t n = xs.size();
    if (n < kMinObsForSampleStd) {
        return {0.0, 0.0};
    }
    double mean = 0.0;
    double m2 = 0.0;
    size_t i = 0;
    for (double x : xs) {
        ++i;
        const double delta = x - mean;
        mean += delta / static_cast<double>(i);
        m2 += delta * (x - mean);
    }
    const double variance = m2 / static_cast<double>(n - 1);
    return {mean, std::sqrt(variance < 0.0 ? 0.0 : variance)};
}

}  // namespace

std::vector<double> MetricsCalculator::equity_to_returns(
    std::span<const double> equity_curve) {
    std::vector<double> returns;
    if (equity_curve.size() < 2) {
        return returns;
    }
    returns.reserve(equity_curve.size() - 1);
    for (size_t i = 1; i < equity_curve.size(); ++i) {
        const double prev = equity_curve[i - 1];
        const double next = equity_curve[i];
        returns.push_back((prev > 0.0) ? (next / prev - 1.0) : 0.0);
    }
    return returns;
}

double MetricsCalculator::sharpe_ratio(
    std::span<const double> returns,
    int annualization_factor,
    double risk_free_rate
) noexcept {
    if (returns.size() < kMinObsForSampleStd) {
        return 0.0;
    }
    // std(r - rf) = std(r) for constant rf (shift-invariance), so one Welford
    // pass yields both the mean and the denominator.
    const auto [mean, sd] = welford_mean_std(returns);
    if (sd <= 0.0) {
        return 0.0;
    }
    return ((mean - risk_free_rate) / sd)
        * std::sqrt(static_cast<double>(annualization_factor));
}

double MetricsCalculator::sortino_ratio(
    std::span<const double> returns,
    int annualization_factor,
    double risk_free_rate
) noexcept {
    const size_t n = returns.size();
    if (n < kMinObsForSampleStd) {
        return 0.0;
    }
    double sum_excess = 0.0;
    double sum_sq_downside = 0.0;
    for (double r : returns) {
        const double excess = r - risk_free_rate;
        sum_excess += excess;
        if (excess < 0.0) {
            sum_sq_downside += excess * excess;
        }
    }
    const double m = sum_excess / static_cast<double>(n);
    const double downside_var = sum_sq_downside / static_cast<double>(n);
    if (downside_var <= 0.0) {
        return 0.0;
    }
    const double downside_std = std::sqrt(downside_var);
    return (m / downside_std) * std::sqrt(static_cast<double>(annualization_factor));
}

double MetricsCalculator::max_drawdown(
    std::span<const double> equity_curve) noexcept {
    if (equity_curve.size() < 2) {
        return 0.0;
    }
    double peak = equity_curve[0];
    double max_dd = 0.0;
    for (double e : equity_curve) {
        if (e > peak) {
            peak = e;
        }
        if (peak > 0.0) {
            const double dd = (e - peak) / peak;
            if (dd < max_dd) {
                max_dd = dd;
            }
        }
    }
    return max_dd;
}

double MetricsCalculator::win_rate(
    std::span<const double> returns) noexcept {
    size_t positives = 0;
    size_t non_zero = 0;
    for (double r : returns) {
        if (r > 0.0) {
            ++positives;
        }
        if (r != 0.0) {
            ++non_zero;
        }
    }
    if (non_zero == 0) {
        return 0.0;
    }
    return static_cast<double>(positives) / static_cast<double>(non_zero);
}

double MetricsCalculator::annualized_return(
    std::span<const double> equity_curve,
    int annualization_factor
) noexcept {
    if (equity_curve.size() < 2) {
        return 0.0;
    }
    const double initial = equity_curve.front();
    const double final_value = equity_curve.back();
    if (initial <= 0.0 || final_value <= 0.0) {
        return 0.0;
    }
    const double n_periods = static_cast<double>(equity_curve.size() - 1);
    const double growth = final_value / initial;
    return std::pow(growth,
                    static_cast<double>(annualization_factor) / n_periods) - 1.0;
}

double MetricsCalculator::annualized_volatility(
    std::span<const double> returns,
    int annualization_factor
) noexcept {
    const auto [_, sd] = welford_mean_std(returns);
    return sd * std::sqrt(static_cast<double>(annualization_factor));
}

// TODO(Phase 6): compute() walks the return series 4× (Welford for sharpe,
// Welford for ann_vol, sortino, win_rate) plus 1× over equity_curve for
// max_drawdown. Profile and, if hot, fuse into a single streaming pass that
// derives returns on the fly and accumulates mean/variance/downside/peak/
// win-count together.
PerformanceMetrics MetricsCalculator::compute(
    std::span<const double> equity_curve,
    int annualization_factor,
    double risk_free_rate
) {
    PerformanceMetrics metrics;
    if (equity_curve.size() < 2) {
        return metrics;
    }
    const auto returns = equity_to_returns(equity_curve);
    const std::span<const double> rs{returns};

    metrics.annualized_return =
        annualized_return(equity_curve, annualization_factor);
    metrics.annualized_volatility =
        annualized_volatility(rs, annualization_factor);
    metrics.sharpe_ratio =
        sharpe_ratio(rs, annualization_factor, risk_free_rate);
    metrics.sortino_ratio =
        sortino_ratio(rs, annualization_factor, risk_free_rate);
    metrics.max_drawdown = max_drawdown(equity_curve);
    metrics.win_rate = win_rate(rs);
    metrics.calmar_ratio = (metrics.max_drawdown < 0.0)
        ? metrics.annualized_return / std::abs(metrics.max_drawdown)
        : 0.0;
    return metrics;
}

}  // namespace quant
