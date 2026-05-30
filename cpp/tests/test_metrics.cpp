#include <cmath>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "quant/core/types.hpp"
#include "quant/metrics/performance.hpp"

namespace quant {
namespace {

constexpr int kYear252 = kTradingDaysPerYear;
constexpr int kYear2 = 2;
constexpr int kYear4 = 4;

constexpr double kExactTol = 1e-12;
constexpr double kFpTol = 1e-9;

constexpr double kInitialEquity = 100.0;

constexpr double kDrawdownPeak = 110.0;
constexpr double kDrawdownTrough = 80.0;

TEST(MetricsCalculatorTest, EmptyEquityAllMetricsZero) {
    const std::vector<double> equity{};
    const auto m = MetricsCalculator::compute(equity, kYear252);
    EXPECT_DOUBLE_EQ(m.annualized_return, 0.0);
    EXPECT_DOUBLE_EQ(m.annualized_volatility, 0.0);
    EXPECT_DOUBLE_EQ(m.sharpe_ratio, 0.0);
    EXPECT_DOUBLE_EQ(m.sortino_ratio, 0.0);
    EXPECT_DOUBLE_EQ(m.max_drawdown, 0.0);
    EXPECT_DOUBLE_EQ(m.win_rate, 0.0);
    EXPECT_DOUBLE_EQ(m.calmar_ratio, 0.0);
}

TEST(MetricsCalculatorTest, SingleElementEquityAllMetricsZero) {
    const std::vector<double> equity{kInitialEquity};
    const auto m = MetricsCalculator::compute(equity, kYear252);
    EXPECT_DOUBLE_EQ(m.sharpe_ratio, 0.0);
    EXPECT_DOUBLE_EQ(m.max_drawdown, 0.0);
    EXPECT_DOUBLE_EQ(m.annualized_return, 0.0);
}

TEST(MetricsCalculatorTest, EquityToReturnsSimple) {
    const std::vector<double> equity{100.0, 110.0, 99.0};
    const auto returns = MetricsCalculator::equity_to_returns(equity);
    ASSERT_EQ(returns.size(), 2U);
    EXPECT_NEAR(returns[0], 0.1, kExactTol);
    EXPECT_NEAR(returns[1], -0.1, kExactTol);
}

TEST(MetricsCalculatorTest, EquityToReturnsNonPositivePrevGivesZero) {
    // prev <= 0 breaks the ratio; guard emits 0 rather than NaN / Inf.
    const std::vector<double> equity{0.0, 100.0, 110.0};
    const auto returns = MetricsCalculator::equity_to_returns(equity);
    ASSERT_EQ(returns.size(), 2U);
    EXPECT_DOUBLE_EQ(returns[0], 0.0);
    EXPECT_NEAR(returns[1], 0.1, kExactTol);
}

TEST(MetricsCalculatorTest, MaxDrawdownPeakToTrough) {
    const std::vector<double> equity{100.0, 110.0, 90.0, 95.0, 80.0, 85.0};
    const double expected =
        (kDrawdownTrough - kDrawdownPeak) / kDrawdownPeak;  // ~= -0.2727
    EXPECT_NEAR(MetricsCalculator::max_drawdown(equity), expected, kExactTol);
}

TEST(MetricsCalculatorTest, MaxDrawdownMonotonicIncreaseZero) {
    const std::vector<double> equity{100.0, 110.0, 120.0, 130.0};
    EXPECT_DOUBLE_EQ(MetricsCalculator::max_drawdown(equity), 0.0);
}

TEST(MetricsCalculatorTest, MaxDrawdownFlatZero) {
    const std::vector<double> equity{100.0, 100.0, 100.0};
    EXPECT_DOUBLE_EQ(MetricsCalculator::max_drawdown(equity), 0.0);
}

TEST(MetricsCalculatorTest, WinRateMixed) {
    // 2 positives out of 4 non-zero returns.
    const std::vector<double> returns{0.1, -0.1, 0.0, 0.2, -0.05};
    EXPECT_DOUBLE_EQ(MetricsCalculator::win_rate(returns), 0.5);
}

TEST(MetricsCalculatorTest, WinRateAllPositive) {
    const std::vector<double> returns{0.01, 0.02, 0.03};
    EXPECT_DOUBLE_EQ(MetricsCalculator::win_rate(returns), 1.0);
}

TEST(MetricsCalculatorTest, WinRateAllZerosSafeZero) {
    const std::vector<double> returns{0.0, 0.0, 0.0};
    EXPECT_DOUBLE_EQ(MetricsCalculator::win_rate(returns), 0.0);
}

TEST(MetricsCalculatorTest, AnnualizedReturnGeometric) {
    // Equity doubled over one period with ann=2 -> 2^2 - 1 = 3.0.
    const std::vector<double> equity{100.0, 200.0};
    EXPECT_NEAR(MetricsCalculator::annualized_return(equity, kYear2),
                3.0, kExactTol);
}

TEST(MetricsCalculatorTest, AnnualizedReturnFlatZero) {
    const std::vector<double> equity{100.0, 100.0, 100.0};
    EXPECT_DOUBLE_EQ(MetricsCalculator::annualized_return(equity, kYear252),
                     0.0);
}

TEST(MetricsCalculatorTest, AnnualizedVolatilityScalesBySqrtFactor) {
    // returns = [0.01, 0.02, 0.03] -> sample_std = 0.01 exactly.
    const std::vector<double> returns{0.01, 0.02, 0.03};
    const double expected = 0.01 * std::sqrt(static_cast<double>(kYear4));
    EXPECT_NEAR(MetricsCalculator::annualized_volatility(returns, kYear4),
                expected, kFpTol);
}

TEST(MetricsCalculatorTest, SharpeRiskFreeLowersRatio) {
    const std::vector<double> returns{0.01, 0.02, 0.03};
    const double with_zero_rf =
        MetricsCalculator::sharpe_ratio(returns, kYear252, /*rf=*/0.0);
    const double with_rf =
        MetricsCalculator::sharpe_ratio(returns, kYear252, /*rf=*/0.01);
    EXPECT_GT(with_zero_rf, 0.0);
    EXPECT_GT(with_rf, 0.0);
    EXPECT_LT(with_rf, with_zero_rf);
}

TEST(MetricsCalculatorTest, SharpeZeroVarianceSafeZero) {
    // Flat returns -> std = 0 -> 0 rather than NaN/Inf.
    const std::vector<double> returns{0.01, 0.01, 0.01};
    EXPECT_DOUBLE_EQ(MetricsCalculator::sharpe_ratio(returns, kYear252), 0.0);
}

TEST(MetricsCalculatorTest, SharpeSingleObservationZero) {
    const std::vector<double> returns{0.05};
    EXPECT_DOUBLE_EQ(MetricsCalculator::sharpe_ratio(returns, kYear252), 0.0);
}

TEST(MetricsCalculatorTest, SharpeHandComputed) {
    // returns = [0.01, 0.02, 0.03] -> mean=0.02, sample_std=0.01 exactly
    // (symmetric around the mean, deviations +/-0.01, sum_sq=2e-4, var=1e-4).
    const std::vector<double> returns{0.01, 0.02, 0.03};
    const double mean_val = 0.02;
    const double sd = 0.01;
    const double expected =
        (mean_val / sd) * std::sqrt(static_cast<double>(kYear252));
    EXPECT_NEAR(MetricsCalculator::sharpe_ratio(returns, kYear252),
                expected, kFpTol);
}

TEST(MetricsCalculatorTest, SortinoOnlyPositiveReturnsZero) {
    // No downside -> downside_var = 0 -> safe zero.
    const std::vector<double> returns{0.01, 0.02, 0.03};
    EXPECT_DOUBLE_EQ(MetricsCalculator::sortino_ratio(returns, kYear252), 0.0);
}

TEST(MetricsCalculatorTest, SortinoHandComputed) {
    // returns = [0.1, -0.05, 0.2, -0.02, 0.03]
    //   mean = 0.052; negative squares = 0.0025 + 0.0004 = 0.0029
    //   downside_var = 0.0029 / 5 = 0.00058
    //   downside_std = sqrt(0.00058)
    //   sortino = (0.052 / downside_std) * sqrt(252)
    const std::vector<double> returns{0.1, -0.05, 0.2, -0.02, 0.03};
    const double mean_val = 0.052;
    const double downside_std = std::sqrt(0.00058);
    const double expected =
        (mean_val / downside_std) * std::sqrt(static_cast<double>(kYear252));
    EXPECT_NEAR(MetricsCalculator::sortino_ratio(returns, kYear252),
                expected, kFpTol);
}

TEST(MetricsCalculatorTest, SortinoAllNegativeReturnsIsNegative) {
    // All downside -> numerator negative, denominator positive -> negative sortino.
    const std::vector<double> returns{-0.01, -0.02, -0.03};
    EXPECT_LT(MetricsCalculator::sortino_ratio(returns, kYear252), 0.0);
}

TEST(MetricsCalculatorTest, ComputeFlatEquityAllZero) {
    const std::vector<double> equity{100.0, 100.0, 100.0, 100.0};
    const auto m = MetricsCalculator::compute(equity, kYear252);
    EXPECT_DOUBLE_EQ(m.annualized_return, 0.0);
    EXPECT_DOUBLE_EQ(m.annualized_volatility, 0.0);
    EXPECT_DOUBLE_EQ(m.sharpe_ratio, 0.0);
    EXPECT_DOUBLE_EQ(m.sortino_ratio, 0.0);
    EXPECT_DOUBLE_EQ(m.max_drawdown, 0.0);
    EXPECT_DOUBLE_EQ(m.win_rate, 0.0);
    EXPECT_DOUBLE_EQ(m.calmar_ratio, 0.0);
}

TEST(MetricsCalculatorTest, ComputeMatchesIndividualMethods) {
    // Curve mixes drop, recovery, and modest final loss so every metric is
    // non-trivial. compute() is a thin aggregator - verify its fields match
    // what the individual static methods return on the same inputs.
    const std::vector<double> equity{
        100.0, 110.0, 90.0, 95.0, 80.0, 85.0, 120.0, 115.0};
    const auto returns = MetricsCalculator::equity_to_returns(equity);
    const std::span<const double> rs{returns};

    const auto m = MetricsCalculator::compute(equity, kYear252);
    EXPECT_NEAR(m.annualized_return,
                MetricsCalculator::annualized_return(equity, kYear252),
                kFpTol);
    EXPECT_NEAR(m.annualized_volatility,
                MetricsCalculator::annualized_volatility(rs, kYear252),
                kFpTol);
    EXPECT_NEAR(m.sharpe_ratio,
                MetricsCalculator::sharpe_ratio(rs, kYear252), kFpTol);
    EXPECT_NEAR(m.sortino_ratio,
                MetricsCalculator::sortino_ratio(rs, kYear252), kFpTol);
    EXPECT_NEAR(m.max_drawdown,
                MetricsCalculator::max_drawdown(equity), kFpTol);
    EXPECT_NEAR(m.win_rate, MetricsCalculator::win_rate(rs), kFpTol);

    const double expected_calmar = (m.max_drawdown < 0.0)
        ? m.annualized_return / std::abs(m.max_drawdown) : 0.0;
    EXPECT_NEAR(m.calmar_ratio, expected_calmar, kFpTol);
}

TEST(MetricsCalculatorTest, ComputeRiskFreeLowersSharpeAndSortino) {
    // Same equity curve, two different rf values: rf>0 shrinks the excess-
    // return numerator, so both risk-adjusted ratios fall.
    const std::vector<double> equity{
        100.0, 101.0, 102.5, 101.5, 103.0, 104.0};
    const double rf = 0.005;  // per-bar risk-free rate (well below avg return)
    const auto zero_rf = MetricsCalculator::compute(equity, kYear252, 0.0);
    const auto with_rf = MetricsCalculator::compute(equity, kYear252, rf);
    EXPECT_GT(zero_rf.sharpe_ratio, 0.0);
    EXPECT_LT(with_rf.sharpe_ratio, zero_rf.sharpe_ratio);
    EXPECT_LT(with_rf.sortino_ratio, zero_rf.sortino_ratio);
    // rf is a Sharpe/Sortino input only - cash-flow metrics stay put.
    EXPECT_DOUBLE_EQ(with_rf.max_drawdown, zero_rf.max_drawdown);
    EXPECT_DOUBLE_EQ(with_rf.annualized_return, zero_rf.annualized_return);
}

TEST(MetricsCalculatorTest, ComputeCalmarUsesDrawdownMagnitude) {
    // Sanity: calmar = annualized_return / |max_drawdown|.
    const std::vector<double> equity{100.0, 110.0, 80.0, 85.0};
    const auto m = MetricsCalculator::compute(equity, kYear4);
    ASSERT_LT(m.max_drawdown, 0.0);
    const double expected = m.annualized_return / std::abs(m.max_drawdown);
    EXPECT_NEAR(m.calmar_ratio, expected, kFpTol);
}

}  // namespace
}  // namespace quant
