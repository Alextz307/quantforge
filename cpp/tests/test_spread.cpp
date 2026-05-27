#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quant/statistics/spread.hpp"

namespace quant::statistics {
namespace {

constexpr double kTol = 1e-12;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

TEST(SpreadCalculator, ComputeSpreadSubtractsHedgedLeg) {
    const std::vector<double> a = {10.0, 20.0, 30.0};
    const std::vector<double> b = {1.0, 2.0, 3.0};
    const auto out = SpreadCalculator::compute_spread(a, b, 2.0);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[0], 8.0, kTol);
    EXPECT_NEAR(out[1], 16.0, kTol);
    EXPECT_NEAR(out[2], 24.0, kTol);
}

TEST(SpreadCalculator, ComputeSpreadLengthMismatchThrows) {
    const std::vector<double> a(5, 1.0);
    const std::vector<double> b(4, 1.0);
    EXPECT_THROW((void)SpreadCalculator::compute_spread(a, b, 1.0), std::invalid_argument);
}

TEST(SpreadCalculator, ZScoreLeadingWindowIsNaN) {
    const std::vector<double> spread = {1.0, 2.0, 3.0, 4.0, 5.0};
    const auto out = SpreadCalculator::compute_zscore(spread, 3);
    ASSERT_EQ(out.size(), spread.size());
    EXPECT_TRUE(std::isnan(out[0]));
    EXPECT_TRUE(std::isnan(out[1]));
    EXPECT_FALSE(std::isnan(out[2]));
}

TEST(SpreadCalculator, ZScoreMatchesHandComputedValue) {
    const std::vector<double> spread = {1.0, 2.0, 3.0, 4.0, 5.0};
    const auto out = SpreadCalculator::compute_zscore(spread, 3);
    EXPECT_NEAR(out[2], 1.0, kTol);
    EXPECT_NEAR(out[3], 1.0, kTol);
    EXPECT_NEAR(out[4], 1.0, kTol);
}

TEST(SpreadCalculator, ZScoreConstantSeriesEmitsNaN) {
    const std::vector<double> spread(5, 42.0);
    const auto out = SpreadCalculator::compute_zscore(spread, 3);
    for (std::size_t i = 2; i < out.size(); ++i) {
        EXPECT_TRUE(std::isnan(out[i])) << "position " << i << " should be NaN";
    }
}

TEST(SpreadCalculator, ZScoreWindowTooLargeReturnsAllNaN) {
    const std::vector<double> spread = {1.0, 2.0};
    const auto out = SpreadCalculator::compute_zscore(spread, 5);
    ASSERT_EQ(out.size(), spread.size());
    for (const double v : out) {
        EXPECT_TRUE(std::isnan(v));
    }
}

TEST(SpreadCalculator, ZScoreWindowBelowTwoThrows) {
    const std::vector<double> spread = {1.0, 2.0, 3.0};
    EXPECT_THROW((void)SpreadCalculator::compute_zscore(spread, 1), std::invalid_argument);
    EXPECT_THROW((void)SpreadCalculator::compute_zscore(spread, 0), std::invalid_argument);
}

TEST(SpreadCalculator, ZScoreWindowTwoIsSmallestLegalValue) {
    const std::vector<double> spread = {1.0, 3.0, 5.0};
    const auto out = SpreadCalculator::compute_zscore(spread, 2);
    ASSERT_EQ(out.size(), spread.size());
    EXPECT_TRUE(std::isnan(out[0]));
    const double expected = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(out[1], expected, kTol);
    EXPECT_NEAR(out[2], expected, kTol);
}

TEST(SpreadCalculator, ZScoreNaNInInputPoisonsSubsequentOutputs) {
    // Documented semantics: Welford has no recovery from NaN in the input —
    // once a NaN enters the accumulator, every subsequent output is NaN
    // (unlike pandas' rolling(w).std() which resumes when NaN exits the
    // window). Pin the behavior so future refactors don't silently change
    // it without updating the header docstring.
    const std::vector<double> spread = {1.0, 2.0, kNaN, 4.0, 5.0, 6.0};
    const auto out = SpreadCalculator::compute_zscore(spread, 2);
    ASSERT_EQ(out.size(), spread.size());
    EXPECT_TRUE(std::isnan(out[0]));
    EXPECT_FALSE(std::isnan(out[1]));
    // Every output from the NaN onward is NaN, even after NaN has slid out.
    for (std::size_t i = 2; i < out.size(); ++i) {
        EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at position " << i;
    }
}

}  // namespace
}  // namespace quant::statistics
