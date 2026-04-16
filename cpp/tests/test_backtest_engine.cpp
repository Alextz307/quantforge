#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quant/core/types.hpp"
#include "quant/engine/backtest_engine.hpp"
#include "quant/engine/slippage.hpp"

namespace quant {
namespace {

// ───── Synthetic bar constants ─────
constexpr int64_t kBaseTimestampS = 1'000'000;
constexpr int64_t kSecondsPerBar = 86'400;
constexpr double kSampleVolume = 1'000'000.0;
constexpr double kTinyOrderVolume = 100.0;

// ───── Engine defaults ─────
constexpr double kInitialCapital = 10'000.0;
constexpr double kZeroFeeRate = 0.0;
constexpr double kTenBpsFeeRate = 0.001;
constexpr double kFloatTolerance = 1e-9;

// ───── Slippage test parameters ─────
constexpr double kFixedSlippageBps = 10.0;   // 10 bp = 0.1%
constexpr double kVolumeImpactCoeff = 5000.0;

// ───── Synthetic price levels reused across directional tests ─────
constexpr double kPriceAt80 = 80.0;
constexpr double kPriceAt90 = 90.0;
constexpr double kPriceAt100 = 100.0;
constexpr double kPriceAt110 = 110.0;
constexpr double kPriceAt120 = 120.0;

// ───── Helpers ─────
Bar make_bar(size_t idx, double open, double high, double low, double close,
             double volume = kSampleVolume) {
    return Bar{
        .timestamp_epoch_s = kBaseTimestampS + static_cast<int64_t>(idx) * kSecondsPerBar,
        .open = open,
        .high = high,
        .low = low,
        .close = close,
        .volume = volume,
    };
}

std::vector<Bar> constant_price_series(size_t n, double price = kPriceAt100) {
    std::vector<Bar> bars;
    bars.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        bars.push_back(make_bar(i, price, price, price, price));
    }
    return bars;
}

BacktestEngine::Config make_config(
    double fee_rate,
    SlippageConfig slippage,
    bool allow_short = true
) {
    return BacktestEngine::Config{
        .initial_capital = kInitialCapital,
        .transaction_fee_rate = fee_rate,
        .slippage = slippage,
        .allow_short = allow_short,
    };
}

BacktestEngine::Config zero_friction_config(bool allow_short = true) {
    return make_config(
        kZeroFeeRate,
        SlippageConfig{SlippageModel::NoSlippage, 0.0, 0.0},
        allow_short);
}

// ═══════════════════════════════════════════════════════════════
// Input validation & trivial shapes
// ═══════════════════════════════════════════════════════════════

TEST(BacktestEngineTest, FewerSignalsThanBarsThrows) {
    BacktestEngine engine(zero_friction_config());
    const auto bars = constant_price_series(3);
    const std::vector<double> signals{0.0, 0.0};
    EXPECT_THROW(static_cast<void>(engine.run(bars, signals)), std::invalid_argument);
}

TEST(BacktestEngineTest, MoreSignalsThanBarsThrows) {
    BacktestEngine engine(zero_friction_config());
    const auto bars = constant_price_series(2);
    const std::vector<double> signals{0.0, 0.0, 0.0};
    EXPECT_THROW(static_cast<void>(engine.run(bars, signals)), std::invalid_argument);
}

TEST(BacktestEngineTest, EmptySeriesReturnsDefault) {
    BacktestEngine engine(zero_friction_config());
    const std::vector<Bar> bars{};
    const std::vector<double> signals{};
    const auto result = engine.run(bars, signals);
    EXPECT_TRUE(result.equity_curve.empty());
    EXPECT_EQ(result.trade_count, 0);
    EXPECT_DOUBLE_EQ(result.total_return, 0.0);
}

TEST(BacktestEngineTest, SingleBarNoFillPossible) {
    BacktestEngine engine(zero_friction_config());
    const auto bars = constant_price_series(1);
    const std::vector<double> signals{1.0};  // no bar t+1 to fill into
    const auto result = engine.run(bars, signals);
    ASSERT_EQ(result.equity_curve.size(), 1U);
    EXPECT_DOUBLE_EQ(result.equity_curve[0], kInitialCapital);
    EXPECT_EQ(result.trade_count, 0);
}

// ═══════════════════════════════════════════════════════════════
// Signal semantics: flat, NaN, allow_short
// ═══════════════════════════════════════════════════════════════

TEST(BacktestEngineTest, FlatSignalNoTrades) {
    BacktestEngine engine(zero_friction_config());
    const auto bars = constant_price_series(5);
    const std::vector<double> signals(5, 0.0);
    const auto result = engine.run(bars, signals);
    EXPECT_EQ(result.trade_count, 0);
    for (double v : result.equity_curve) {
        EXPECT_DOUBLE_EQ(v, kInitialCapital);
    }
}

TEST(BacktestEngineTest, NanSignalTreatedAsFlat) {
    BacktestEngine engine(zero_friction_config());
    const auto bars = constant_price_series(4);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> signals{nan, nan, nan, nan};
    const auto result = engine.run(bars, signals);
    EXPECT_EQ(result.trade_count, 0);
    EXPECT_DOUBLE_EQ(result.equity_curve.back(), kInitialCapital);
}

TEST(BacktestEngineTest, AllowShortFalseClipsNegativeSignal) {
    BacktestEngine engine(zero_friction_config(/*allow_short=*/false));
    const auto bars = constant_price_series(3);
    const std::vector<double> signals{-1.0, -1.0, -1.0};
    const auto result = engine.run(bars, signals);
    EXPECT_EQ(result.trade_count, 0);
    EXPECT_DOUBLE_EQ(result.equity_curve.back(), kInitialCapital);
}

TEST(BacktestEngineTest, ShortProfitsOnPriceDrop) {
    BacktestEngine engine(zero_friction_config(/*allow_short=*/true));
    // Enter short at bar 1 open (90); bar 2 drops to 80 → MTM gain.
    const std::vector<Bar> bars{
        make_bar(0, kPriceAt100, kPriceAt100, kPriceAt100, kPriceAt100),
        make_bar(1, kPriceAt90, kPriceAt90, kPriceAt90, kPriceAt90),
        make_bar(2, kPriceAt80, kPriceAt80, kPriceAt80, kPriceAt80),
    };
    const std::vector<double> signals{-1.0, -1.0, -1.0};
    const auto result = engine.run(bars, signals);
    EXPECT_GT(result.equity_curve.back(), kInitialCapital);
    // Shorted (kInitialCapital / kPriceAt90) shares at 90; bar 2 at 80.
    // Rebalancing to maintain leverage is equity-neutral at the instant of
    // fill, so MTM gain at bar 2 close = shares * (entry - close).
    const double expected_gain =
        (kInitialCapital / kPriceAt90) * (kPriceAt90 - kPriceAt80);
    EXPECT_NEAR(result.equity_curve.back() - kInitialCapital,
                expected_gain, kFloatTolerance);
}

// ═══════════════════════════════════════════════════════════════
// Position sizing & mark-to-market
// ═══════════════════════════════════════════════════════════════

TEST(BacktestEngineTest, AlwaysLongTracksPrice) {
    BacktestEngine engine(zero_friction_config());
    const std::vector<Bar> bars{
        make_bar(0, kPriceAt100, kPriceAt100, kPriceAt100, kPriceAt100),
        make_bar(1, kPriceAt110, kPriceAt120, kPriceAt110, kPriceAt120),
    };
    const std::vector<double> signals{1.0, 1.0};
    const auto result = engine.run(bars, signals);
    // Buy at bar 1 open (110) with 10,000 → 90.909... shares.
    // MTM at bar 1 close (120): 90.909... * 120 = 10909.090909...
    const double expected_shares = kInitialCapital / kPriceAt110;
    const double expected_equity = expected_shares * kPriceAt120;
    EXPECT_NEAR(result.equity_curve.back(), expected_equity, kFloatTolerance);
    EXPECT_EQ(result.trade_count, 1);
    EXPECT_NEAR(result.total_return,
                (expected_equity / kInitialCapital) - 1.0,
                kFloatTolerance);
}

TEST(BacktestEngineTest, TradeCountReflectsDistinctPositionChanges) {
    BacktestEngine engine(zero_friction_config());
    const auto bars = constant_price_series(5);
    // 0 → +1 → +1 → -1 → 0 : three distinct position changes.
    const std::vector<double> signals{1.0, 1.0, -1.0, 0.0, 0.0};
    const auto result = engine.run(bars, signals);
    EXPECT_EQ(result.trade_count, 3);
}

// ═══════════════════════════════════════════════════════════════
// Slippage
// ═══════════════════════════════════════════════════════════════

TEST(BacktestEngineTest, FixedSlippageRaisesBuyFillPrice) {
    BacktestEngine engine(make_config(
        kZeroFeeRate,
        SlippageConfig{SlippageModel::Fixed, kFixedSlippageBps, 0.0}));
    const std::vector<Bar> bars{
        make_bar(0, kPriceAt100, kPriceAt100, kPriceAt100, kPriceAt100),
        make_bar(1, kPriceAt110, kPriceAt110, kPriceAt110, kPriceAt110),
    };
    const std::vector<double> signals{1.0, 0.0};
    const auto result = engine.run(bars, signals);
    // Sizing uses theoretical_open: target_shares = 10000 / 110.
    // Fill happens at 110 * (1 + 10bp) = 110.11, so cash goes slightly
    // negative (target_shares * 110.11 > initial_capital). At bar 1 close
    // (110), equity = cash + shares * close = 10000 - bps_fraction * 10000.
    const double bps_fraction = kFixedSlippageBps / kBpsPerUnit;
    const double expected_equity = kInitialCapital * (1.0 - bps_fraction);
    EXPECT_NEAR(result.equity_curve.back(), expected_equity, kFloatTolerance);
}

TEST(BacktestEngineTest, VolumeScaledSlippageIncreasesWithOrderSize) {
    // Two runs with identical bars/signals but different volume on the fill bar.
    // Smaller volume → larger slippage impact → lower final equity after a long entry.
    const std::vector<double> signals{1.0, 0.0};
    const std::vector<Bar> bars_large_volume{
        make_bar(0, kPriceAt100, kPriceAt100, kPriceAt100, kPriceAt100),
        make_bar(1, kPriceAt110, kPriceAt110, kPriceAt110, kPriceAt110,
                 kSampleVolume),
    };
    const std::vector<Bar> bars_small_volume{
        make_bar(0, kPriceAt100, kPriceAt100, kPriceAt100, kPriceAt100),
        make_bar(1, kPriceAt110, kPriceAt110, kPriceAt110, kPriceAt110,
                 kTinyOrderVolume),
    };
    BacktestEngine engine(make_config(
        kZeroFeeRate,
        SlippageConfig{SlippageModel::VolumeScaled, 0.0, kVolumeImpactCoeff}));
    const auto result_large = engine.run(bars_large_volume, signals);
    const auto result_small = engine.run(bars_small_volume, signals);
    EXPECT_LT(result_small.equity_curve.back(), result_large.equity_curve.back());
}

// ═══════════════════════════════════════════════════════════════
// Commission
// ═══════════════════════════════════════════════════════════════

TEST(BacktestEngineTest, CommissionBleedsAcrossAlternatingTrades) {
    BacktestEngine engine(make_config(
        kTenBpsFeeRate,
        SlippageConfig{SlippageModel::NoSlippage, 0.0, 0.0}));
    const auto bars = constant_price_series(5);  // flat prices → no price PnL
    // 0 → +1 → -1 → +1 → -1: 4 position changes.
    const std::vector<double> signals{1.0, -1.0, 1.0, -1.0, 0.0};
    const auto result = engine.run(bars, signals);
    EXPECT_GT(result.trade_count, 0);
    EXPECT_LT(result.equity_curve.back(), kInitialCapital);
}

// ═══════════════════════════════════════════════════════════════
// Metrics boundary: statistical fields default-zero in Batch A
// ═══════════════════════════════════════════════════════════════

TEST(BacktestEngineTest, StatisticalMetricsDefaultZero) {
    BacktestEngine engine(zero_friction_config());
    const auto bars = constant_price_series(3);
    const std::vector<double> signals{1.0, 1.0, 0.0};
    const auto result = engine.run(bars, signals);
    // Engine owns cash-flow metrics; MetricsCalculator (Batch B) owns these.
    EXPECT_DOUBLE_EQ(result.sharpe_ratio, 0.0);
    EXPECT_DOUBLE_EQ(result.sortino_ratio, 0.0);
    EXPECT_DOUBLE_EQ(result.max_drawdown, 0.0);
    EXPECT_DOUBLE_EQ(result.win_rate, 0.0);
    EXPECT_DOUBLE_EQ(result.annualized_return, 0.0);
    EXPECT_DOUBLE_EQ(result.annualized_volatility, 0.0);
}

}  // namespace
}  // namespace quant
