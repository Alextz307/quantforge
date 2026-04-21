#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "detail/parity_helpers.hpp"
#include "detail/spread_reference.hpp"
#include "quant/statistics/spread.hpp"

namespace {

using quant::statistics::SpreadCalculator;
using quant::tests::detail::additive_random_walk;
using quant::tests::detail::compute_zscore_reference;
using quant::tests::detail::expect_array_eq;

constexpr unsigned kSeed = 0xFACADEu;
constexpr double kStartPrice = 50.0;
constexpr double kSpreadStdDev = 0.02;
constexpr std::size_t kSmallN = 1000;
constexpr std::size_t kLargeN = 10000;
constexpr int kWindow = 30;
constexpr double kShortSeriesFill = 1.0;

std::vector<double> spread(std::size_t n) {
    return additive_random_walk(n, kSeed, kStartPrice, kSpreadStdDev);
}

TEST(SpreadZScoreFusedParity, MatchesReferenceSmall) {
    const auto s = spread(kSmallN);
    expect_array_eq(
        SpreadCalculator::compute_zscore(s, kWindow),
        compute_zscore_reference(s, kWindow));
}

TEST(SpreadZScoreFusedParity, MatchesReferenceLarge) {
    const auto s = spread(kLargeN);
    expect_array_eq(
        SpreadCalculator::compute_zscore(s, kWindow),
        compute_zscore_reference(s, kWindow));
}

TEST(SpreadZScoreFusedParity, ShortSeriesAllNaN) {
    const std::vector<double> s(kWindow - 1, kShortSeriesFill);
    expect_array_eq(
        SpreadCalculator::compute_zscore(s, kWindow),
        compute_zscore_reference(s, kWindow));
}

}  // namespace
