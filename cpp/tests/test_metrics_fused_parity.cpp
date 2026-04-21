#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "detail/metrics_reference.hpp"
#include "detail/parity_helpers.hpp"
#include "quant/metrics/performance.hpp"

namespace {

using quant::MetricsCalculator;
using quant::PerformanceMetrics;
using quant::tests::detail::geometric_random_walk;
using quant::tests::detail::metrics_compute_reference;

constexpr unsigned kSeed = 0xD15EA5Eu;
constexpr double kStartEquity = 100.0;
constexpr double kReturnStdDev = 0.015;
constexpr std::size_t kSmallN = 1000;
constexpr std::size_t kLargeN = 10000;
constexpr int kAnnFactor = 252;
constexpr double kRiskFreeRate = 0.0002;

std::vector<double> equity(std::size_t n) {
    return geometric_random_walk(n, kSeed, kStartEquity, kReturnStdDev);
}

void expect_metrics_eq(const PerformanceMetrics& a, const PerformanceMetrics& b) {
    EXPECT_EQ(a.annualized_return, b.annualized_return);
    EXPECT_EQ(a.annualized_volatility, b.annualized_volatility);
    EXPECT_EQ(a.sharpe_ratio, b.sharpe_ratio);
    EXPECT_EQ(a.sortino_ratio, b.sortino_ratio);
    EXPECT_EQ(a.max_drawdown, b.max_drawdown);
    EXPECT_EQ(a.calmar_ratio, b.calmar_ratio);
    EXPECT_EQ(a.win_rate, b.win_rate);
}

TEST(MetricsFusedParity, MatchesReferenceSmall) {
    const auto eq = equity(kSmallN);
    expect_metrics_eq(
        MetricsCalculator::compute(eq, kAnnFactor, kRiskFreeRate),
        metrics_compute_reference(eq, kAnnFactor, kRiskFreeRate));
}

TEST(MetricsFusedParity, MatchesReferenceLarge) {
    const auto eq = equity(kLargeN);
    expect_metrics_eq(
        MetricsCalculator::compute(eq, kAnnFactor, kRiskFreeRate),
        metrics_compute_reference(eq, kAnnFactor, kRiskFreeRate));
}

TEST(MetricsFusedParity, MatchesReferenceZeroRiskFree) {
    const auto eq = equity(kLargeN);
    expect_metrics_eq(
        MetricsCalculator::compute(eq, kAnnFactor, 0.0),
        metrics_compute_reference(eq, kAnnFactor, 0.0));
}

TEST(MetricsFusedParity, ShortSeriesReturnsDefaults) {
    const std::vector<double> two_point = {100.0, 101.0};
    expect_metrics_eq(
        MetricsCalculator::compute(two_point, kAnnFactor),
        metrics_compute_reference(two_point, kAnnFactor));
}

}  // namespace
