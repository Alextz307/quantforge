#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quant/strategies/state_machines.hpp"

namespace quant::strategies {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Pairs thresholds shared across the z-score tests.
constexpr double kEntryZ = 2.0;
constexpr double kExitZ = 0.5;
constexpr double kStopLossZ = 3.0;

// ───── Mean-reversion state machine ─────

TEST(MeanReversionStateMachine, FlatWhenAllBandsNaN) {
    const std::vector<double> close(5, 100.0);
    const std::vector<double> nans(5, kNaN);
    const auto out = run_mean_reversion_state_machine(close, nans, nans, nans, nans);
    ASSERT_EQ(out.size(), close.size());
    for (const double v : out) EXPECT_TRUE(std::isnan(v));
}

TEST(MeanReversionStateMachine, BullishLongEntryAndMidExit) {
    // Bull regime: close > trend_ma throughout.
    // Bar 0: close below lower → enter long.
    // Bar 1: close between lower and mid → hold long.
    // Bar 2: close at mid → exit to flat.
    const std::vector<double> close    = {90.0, 95.0, 100.0};
    const std::vector<double> mid      = {100.0, 100.0, 100.0};
    const std::vector<double> upper    = {110.0, 110.0, 110.0};
    const std::vector<double> lower    = {92.0, 92.0, 92.0};
    const std::vector<double> trend_ma = {80.0, 80.0, 80.0};

    const auto out = run_mean_reversion_state_machine(close, mid, upper, lower, trend_ma);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], 1.0);
    EXPECT_EQ(out[1], 1.0);
    EXPECT_EQ(out[2], 0.0);
}

TEST(MeanReversionStateMachine, BearishShortEntryAndMidExit) {
    // Bear regime: close < trend_ma throughout.
    const std::vector<double> close    = {115.0, 105.0, 100.0};
    const std::vector<double> mid      = {100.0, 100.0, 100.0};
    const std::vector<double> upper    = {108.0, 108.0, 108.0};
    const std::vector<double> lower    = {90.0, 90.0, 90.0};
    const std::vector<double> trend_ma = {120.0, 120.0, 120.0};

    const auto out = run_mean_reversion_state_machine(close, mid, upper, lower, trend_ma);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], -1.0);
    EXPECT_EQ(out[1], -1.0);
    EXPECT_EQ(out[2], 0.0);
}

TEST(MeanReversionStateMachine, NaNBandSkipsBarAndHoldsPosition) {
    // Enter long at bar 0, NaN band at bar 1 (skip + NaN out, position held),
    // resume at bar 2 still below mid → still long.
    const std::vector<double> close    = {90.0, 95.0, 96.0};
    const std::vector<double> mid      = {100.0, 100.0, 100.0};
    const std::vector<double> upper    = {110.0, kNaN, 110.0};
    const std::vector<double> lower    = {92.0, 92.0, 92.0};
    const std::vector<double> trend_ma = {80.0, 80.0, 80.0};

    const auto out = run_mean_reversion_state_machine(close, mid, upper, lower, trend_ma);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], 1.0);
    EXPECT_TRUE(std::isnan(out[1]));
    EXPECT_EQ(out[2], 1.0);
}

TEST(MeanReversionStateMachine, LengthMismatchThrows) {
    const std::vector<double> a(5, 1.0);
    const std::vector<double> b(4, 1.0);
    EXPECT_THROW((void)run_mean_reversion_state_machine(a, b, a, a, a), std::invalid_argument);
    EXPECT_THROW((void)run_mean_reversion_state_machine(a, a, b, a, a), std::invalid_argument);
    EXPECT_THROW((void)run_mean_reversion_state_machine(a, a, a, b, a), std::invalid_argument);
    EXPECT_THROW((void)run_mean_reversion_state_machine(a, a, a, a, b), std::invalid_argument);
}

// ───── Pairs state machine ─────

TEST(PairsStateMachine, EntryShortAndExit) {
    // z timeline: below entry, above entry (short), back inside exit (flat),
    // still inside exit (flat).
    const std::vector<double> z = {0.0, 2.5, 0.3, 0.1};
    const auto out = run_pairs_state_machine(z, kEntryZ, kExitZ, kStopLossZ);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(out[0], 0.0);
    EXPECT_EQ(out[1], -1.0);
    EXPECT_EQ(out[2], 0.0);
    EXPECT_EQ(out[3], 0.0);
}

TEST(PairsStateMachine, EntryLongAndExit) {
    const std::vector<double> z = {0.0, -2.5, -0.3, 0.1};
    const auto out = run_pairs_state_machine(z, kEntryZ, kExitZ, kStopLossZ);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(out[0], 0.0);
    EXPECT_EQ(out[1], 1.0);
    EXPECT_EQ(out[2], 0.0);
    EXPECT_EQ(out[3], 0.0);
}

TEST(PairsStateMachine, StopLossForcesFlat) {
    // Enter short on bar 0; z blows past stop-loss on bar 1 → flat.
    // Bar 2: |z|=1.0 is inside the entry band, so the state holds at flat.
    const std::vector<double> z = {2.5, 3.5, 1.0};
    const auto out = run_pairs_state_machine(z, kEntryZ, kExitZ, kStopLossZ);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], -1.0);
    EXPECT_EQ(out[1], 0.0);
    EXPECT_EQ(out[2], 0.0);
}

TEST(PairsStateMachine, NaNSkipsBarAndHoldsPosition) {
    // Enter short on bar 0, NaN at bar 1 holds short + emits NaN,
    // exit on bar 2.
    const std::vector<double> z = {2.5, kNaN, 0.3};
    const auto out = run_pairs_state_machine(z, kEntryZ, kExitZ, kStopLossZ);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], -1.0);
    EXPECT_TRUE(std::isnan(out[1]));
    EXPECT_EQ(out[2], 0.0);
}

}  // namespace
}  // namespace quant::strategies
