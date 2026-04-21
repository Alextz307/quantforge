#include <cstddef>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "detail/parity_helpers.hpp"
#include "quant/indicators/bollinger_bands.hpp"
#include "quant/indicators/detail/rolling.hpp"

namespace {

using quant::BollingerBands;
using quant::detail::rolling_mean;
using quant::detail::rolling_mean_std;
using quant::detail::rolling_std;
using quant::tests::detail::expect_array_eq;
using quant::tests::detail::geometric_random_walk;

constexpr unsigned kSeed = 0xBEEFu;
constexpr double kStartPrice = 100.0;
constexpr double kReturnStdDev = 0.01;
constexpr std::size_t kSmallN = 1000;
constexpr std::size_t kLargeN = 10000;
constexpr int kWindow = 20;
constexpr double kNumStd = 2.0;

std::vector<double> prices(std::size_t n) {
    return geometric_random_walk(n, kSeed, kStartPrice, kReturnStdDev);
}

TEST(RollingMeanStdParity, MatchesIndependentHelpersSmall) {
    const auto p = prices(kSmallN);
    std::vector<double> mean_fused(kSmallN);
    std::vector<double> std_fused(kSmallN);
    rolling_mean_std(p, kWindow, mean_fused, std_fused);
    expect_array_eq(mean_fused, rolling_mean(p, kWindow));
    expect_array_eq(std_fused, rolling_std(p, kWindow));
}

TEST(RollingMeanStdParity, MatchesIndependentHelpersLarge) {
    const auto p = prices(kLargeN);
    std::vector<double> mean_fused(kLargeN);
    std::vector<double> std_fused(kLargeN);
    rolling_mean_std(p, kWindow, mean_fused, std_fused);
    expect_array_eq(mean_fused, rolling_mean(p, kWindow));
    expect_array_eq(std_fused, rolling_std(p, kWindow));
}

TEST(BollingerFusedParity, ComputeAllBitIdentical) {
    const auto p = prices(kLargeN);
    BollingerBands bb(kWindow, kNumStd);
    const auto fused = bb.compute_all(p);

    const auto mean_ref = rolling_mean(p, kWindow);
    const auto std_ref = rolling_std(p, kWindow);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> upper_ref(kLargeN, nan);
    std::vector<double> lower_ref(kLargeN, nan);
    for (std::size_t i = static_cast<std::size_t>(kWindow - 1); i < kLargeN; ++i) {
        upper_ref[i] = mean_ref[i] + kNumStd * std_ref[i];
        lower_ref[i] = mean_ref[i] - kNumStd * std_ref[i];
    }

    expect_array_eq(fused.mid, mean_ref);
    expect_array_eq(fused.upper, upper_ref);
    expect_array_eq(fused.lower, lower_ref);
}

}  // namespace
