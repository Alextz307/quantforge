#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "quant/filters/garch_filter.hpp"

namespace quant::filters {
namespace {

constexpr double kOmega = 0.05;
constexpr double kAlpha1 = 0.10;
constexpr double kBeta1 = 0.85;
constexpr double kMu = 0.0;
constexpr double kBackcast = 1.0;
constexpr double kExactTolerance = 1e-12;

constexpr double kConstantReturn = 0.5;
constexpr std::size_t kConstantSeriesLen = 5;

constexpr std::size_t kLongStabilitySeriesLen = 5000;
constexpr std::uint_fast32_t kRandomSeed = 1234;
constexpr double kReturnStdDev = 1.0;

TEST(GarchFilter, ConstantReturnsHandVerified) {
    // GARCH(1,1) hand check: r=0.5, mu=0, backcast=1.
    // sigma2[0] = omega + (alpha+beta)*backcast = 1.00
    // sigma2[t>=1] = 0.075 + 0.85 * sigma2[t-1]
    std::vector<double> r(kConstantSeriesLen, kConstantReturn);
    const GarchParams params{kOmega, {kAlpha1}, {kBeta1}, kMu, kBackcast};

    const auto sigma2 = garch_filter(r, params);

    ASSERT_EQ(sigma2.size(), kConstantSeriesLen);
    EXPECT_NEAR(sigma2[0], 1.00, kExactTolerance);
    EXPECT_NEAR(sigma2[1], 0.075 + 0.85 * sigma2[0], kExactTolerance);
    EXPECT_NEAR(sigma2[2], 0.075 + 0.85 * sigma2[1], kExactTolerance);
    EXPECT_NEAR(sigma2[3], 0.075 + 0.85 * sigma2[2], kExactTolerance);
    EXPECT_NEAR(sigma2[4], 0.075 + 0.85 * sigma2[3], kExactTolerance);
}

TEST(GarchFilter, EmptyInputReturnsEmpty) {
    const GarchParams params{kOmega, {kAlpha1}, {kBeta1}, kMu, kBackcast};
    const auto sigma2 = garch_filter(std::span<const double>{}, params);
    EXPECT_TRUE(sigma2.empty());
}

TEST(GarchFilter, SingleBarUsesBackcast) {
    std::vector<double> r{kConstantReturn};
    const GarchParams params{kOmega, {kAlpha1}, {kBeta1}, kMu, kBackcast};
    const auto sigma2 = garch_filter(r, params);

    ASSERT_EQ(sigma2.size(), 1u);
    EXPECT_NEAR(sigma2[0], kOmega + kAlpha1 * kBackcast + kBeta1 * kBackcast,
                kExactTolerance);
}

TEST(GarchFilter, VarianceFloorFiresForPathologicalParams) {
    // Filter must floor at kVarianceFloor when all params are zero.
    std::vector<double> r(kConstantSeriesLen, 0.0);
    const GarchParams params{0.0, {0.0}, {0.0}, 0.0, 0.0};
    const auto sigma2 = garch_filter(r, params);
    for (const double v : sigma2) {
        EXPECT_EQ(v, kVarianceFloor);
    }
}

TEST(GarchFilter, EmptyAlphaUsesOnlyBetaAndOmega) {
    std::vector<double> r(kConstantSeriesLen, kConstantReturn);
    const GarchParams params{kOmega, {}, {kBeta1}, kMu, kBackcast};
    const auto sigma2 = garch_filter(r, params);

    ASSERT_EQ(sigma2.size(), kConstantSeriesLen);
    EXPECT_NEAR(sigma2[0], kOmega + kBeta1 * kBackcast, kExactTolerance);
    EXPECT_NEAR(sigma2[1], kOmega + kBeta1 * sigma2[0], kExactTolerance);
    EXPECT_NEAR(sigma2[2], kOmega + kBeta1 * sigma2[1], kExactTolerance);
}

TEST(GarchFilter, EmptyBetaUsesOnlyAlphaAndOmega) {
    std::vector<double> r(kConstantSeriesLen, kConstantReturn);
    const GarchParams params{kOmega, {kAlpha1}, {}, kMu, kBackcast};
    const auto sigma2 = garch_filter(r, params);

    ASSERT_EQ(sigma2.size(), kConstantSeriesLen);
    EXPECT_NEAR(sigma2[0], kOmega + kAlpha1 * kBackcast, kExactTolerance);
    const double e2 = (kConstantReturn - kMu) * (kConstantReturn - kMu);
    EXPECT_NEAR(sigma2[1], kOmega + kAlpha1 * e2, kExactTolerance);
    EXPECT_NEAR(sigma2[2], kOmega + kAlpha1 * e2, kExactTolerance);
}

TEST(GarchFilter, LongSeriesRemainsFinite) {
    std::mt19937 gen(kRandomSeed);
    std::normal_distribution<double> dist(0.0, kReturnStdDev);
    std::vector<double> r(kLongStabilitySeriesLen);
    for (auto& x : r) x = dist(gen);

    const GarchParams params{kOmega, {kAlpha1}, {kBeta1}, kMu, kBackcast};
    const auto sigma2 = garch_filter(r, params);
    for (const double v : sigma2) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_GT(v, 0.0);
    }
}

}  // namespace
}  // namespace quant::filters
