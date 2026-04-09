#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include <gtest/gtest.h>

#include "quant/indicators/bollinger_bands.hpp"
#include "quant/indicators/detail/rolling.hpp"
#include "quant/indicators/garman_klass.hpp"
#include "quant/indicators/macd.hpp"
#include "quant/indicators/parkinson.hpp"
#include "quant/indicators/rsi.hpp"

namespace quant {
namespace {

// ═══════════════════════════════════════════════════════════════
// RSI Tests
// ═══════════════════════════════════════════════════════════════

TEST(RSITest, WarmupIsNaN) {
    // 20 prices, period=14 → first 14 values should be NaN
    std::vector<double> prices(20, 100.0);
    for (int i = 0; i < 20; ++i) prices[i] = 100.0 + i * 0.5;

    RSI rsi(14);
    auto result = rsi.compute(prices);

    EXPECT_EQ(result.size(), prices.size());
    for (int i = 0; i < 14; ++i) {
        EXPECT_TRUE(std::isnan(result[i])) << "Index " << i << " should be NaN";
    }
    for (size_t i = 14; i < result.size(); ++i) {
        EXPECT_FALSE(std::isnan(result[i])) << "Index " << i << " should not be NaN";
    }
}

TEST(RSITest, MonotonicallyIncreasingApproaches100) {
    // Prices always go up → RSI should be very high
    std::vector<double> prices(100);
    for (int i = 0; i < 100; ++i) prices[i] = 100.0 + i;

    RSI rsi(14);
    auto result = rsi.compute(prices);

    // First valid RSI (index 14) should be 100 (all gains, no losses in seed)
    EXPECT_DOUBLE_EQ(result[14], 100.0);
    // Subsequent values should remain 100 (all gains)
    EXPECT_DOUBLE_EQ(result[50], 100.0);
}

TEST(RSITest, MonotonicallyDecreasingApproaches0) {
    std::vector<double> prices(100);
    for (int i = 0; i < 100; ++i) prices[i] = 200.0 - i;

    RSI rsi(14);
    auto result = rsi.compute(prices);

    EXPECT_DOUBLE_EQ(result[14], 0.0);
    EXPECT_DOUBLE_EQ(result[50], 0.0);
}

TEST(RSITest, ConstantPricesGive50) {
    std::vector<double> prices(30, 100.0);

    RSI rsi(14);
    auto result = rsi.compute(prices);

    EXPECT_DOUBLE_EQ(result[14], 50.0);
    EXPECT_DOUBLE_EQ(result[20], 50.0);
}

TEST(RSITest, KnownReferenceValue) {
    // Hand-computed RSI(3) for prices [10, 11, 12, 11, 13, 14, 12, 15]
    // Deltas: +1, +1, -1, +2, +1, -2, +3
    // Seed (first 3 deltas): avg_gain = (1+1+0)/3 = 0.6667, avg_loss = (0+0+1)/3 = 0.3333
    // RSI[3] = 100 - 100/(1 + 0.6667/0.3333) = 100 - 100/3 = 66.667
    std::vector<double> prices = {10, 11, 12, 11, 13, 14, 12, 15};
    RSI rsi(3);
    auto result = rsi.compute(prices);

    EXPECT_NEAR(result[3], 66.6667, 0.001);

    // RSI[4]: delta=+2, gain=2, loss=0
    // avg_gain = (0.6667*2 + 2)/3 = 3.3333/3 = 1.1111
    // avg_loss = (0.3333*2 + 0)/3 = 0.6667/3 = 0.2222
    // RS = 5.0, RSI = 100 - 100/6 = 83.333
    EXPECT_NEAR(result[4], 83.3333, 0.001);
}

TEST(RSITest, EmptyInput) {
    RSI rsi(14);
    auto result = rsi.compute(std::span<const double>{});
    EXPECT_TRUE(result.empty());
}

TEST(RSITest, TooFewPrices) {
    std::vector<double> prices = {100.0, 101.0, 102.0};
    RSI rsi(14);
    auto result = rsi.compute(prices);
    EXPECT_EQ(result.size(), 3u);
    for (const auto& v : result) {
        EXPECT_TRUE(std::isnan(v));
    }
}

TEST(RSITest, NameFormat) {
    RSI rsi(14);
    EXPECT_EQ(rsi.name(), "RSI(14)");
}

TEST(RSITest, WarmupPeriod) {
    RSI rsi(14);
    EXPECT_EQ(rsi.warmup_period(), 14);
}

TEST(RSITest, InvalidPeriod) {
    EXPECT_THROW(auto r = RSI(0), std::invalid_argument);
    EXPECT_THROW(auto r = RSI(-1), std::invalid_argument);
}

TEST(RSITest, Period1) {
    // RSI(1): every bar is either 100 (up), 0 (down), or 50 (flat)
    std::vector<double> prices = {100, 102, 101, 103};
    RSI rsi(1);
    auto result = rsi.compute(prices);

    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_DOUBLE_EQ(result[1], 100.0);  // up
    EXPECT_DOUBLE_EQ(result[2], 0.0);    // down
    EXPECT_DOUBLE_EQ(result[3], 100.0);  // up
}

TEST(RSITest, RangeIsBounded) {
    // RSI should always be in [0, 100]
    std::vector<double> prices = {100, 105, 95, 110, 85, 120, 80, 115, 90, 105,
                                  95, 110, 85, 120, 80, 115, 90, 105, 95, 110};
    RSI rsi(5);
    auto result = rsi.compute(prices);
    for (size_t i = 5; i < result.size(); ++i) {
        EXPECT_GE(result[i], 0.0);
        EXPECT_LE(result[i], 100.0);
    }
}

// ═══════════════════════════════════════════════════════════════
// MACD Tests
// ═══════════════════════════════════════════════════════════════

TEST(MACDTest, OutputSameLength) {
    std::vector<double> prices(50);
    for (int i = 0; i < 50; ++i) prices[i] = 100.0 + i;

    MACD macd;
    auto result = macd.compute(prices);
    EXPECT_EQ(result.size(), prices.size());

    auto full = macd.compute_all(prices);
    EXPECT_EQ(full.macd_line.size(), prices.size());
    EXPECT_EQ(full.signal_line.size(), prices.size());
    EXPECT_EQ(full.histogram.size(), prices.size());
}

TEST(MACDTest, WarmupNaN) {
    std::vector<double> prices(50);
    for (int i = 0; i < 50; ++i) prices[i] = 100.0 + i * 0.5;

    MACD macd(12, 26, 9);
    auto result = macd.compute(prices);

    // First slow_period - 1 = 25 values should be NaN
    for (int i = 0; i < 25; ++i) {
        EXPECT_TRUE(std::isnan(result[i])) << "MACD line index " << i;
    }
    EXPECT_FALSE(std::isnan(result[25]));
}

TEST(MACDTest, HistogramIsLineMinusSignal) {
    std::vector<double> prices(60);
    for (int i = 0; i < 60; ++i) prices[i] = 100.0 + std::sin(i * 0.3) * 5.0;

    MACD macd;
    auto full = macd.compute_all(prices);

    for (size_t i = 0; i < full.histogram.size(); ++i) {
        if (!std::isnan(full.histogram[i])) {
            EXPECT_NEAR(full.histogram[i],
                        full.macd_line[i] - full.signal_line[i], 1e-10)
                << "at index " << i;
        }
    }
}

TEST(MACDTest, ConstantPricesMACDIsZero) {
    std::vector<double> prices(50, 100.0);

    MACD macd;
    auto full = macd.compute_all(prices);

    for (size_t i = 25; i < full.macd_line.size(); ++i) {
        EXPECT_NEAR(full.macd_line[i], 0.0, 1e-10);
    }
}

TEST(MACDTest, EmptyInput) {
    MACD macd;
    auto result = macd.compute(std::span<const double>{});
    EXPECT_TRUE(result.empty());
}

TEST(MACDTest, NameFormat) {
    MACD macd(12, 26, 9);
    EXPECT_EQ(macd.name(), "MACD(12,26,9)");
}

TEST(MACDTest, WarmupPeriod) {
    MACD macd(12, 26, 9);
    EXPECT_EQ(macd.warmup_period(), 25);
}

TEST(MACDTest, SignalLineWarmup) {
    // Signal line valid at slow_period-1 + signal_period-1 = 25+8 = 33
    std::vector<double> prices(50);
    for (int i = 0; i < 50; ++i) prices[i] = 100.0 + i * 0.5;

    MACD macd(12, 26, 9);
    auto full = macd.compute_all(prices);

    // Signal and histogram should be NaN before index 33
    for (int i = 0; i < 33; ++i) {
        EXPECT_TRUE(std::isnan(full.signal_line[i])) << "signal at " << i;
        EXPECT_TRUE(std::isnan(full.histogram[i])) << "histogram at " << i;
    }
    EXPECT_FALSE(std::isnan(full.signal_line[33]));
    EXPECT_FALSE(std::isnan(full.histogram[33]));
}

TEST(MACDTest, InvalidPeriods) {
    EXPECT_THROW(auto m = MACD(0, 26, 9), std::invalid_argument);
    EXPECT_THROW(auto m = MACD(12, 12, 9), std::invalid_argument);  // fast >= slow
    EXPECT_THROW(auto m = MACD(30, 26, 9), std::invalid_argument);  // fast > slow
}

// ═══════════════════════════════════════════════════════════════
// Bollinger Bands Tests
// ═══════════════════════════════════════════════════════════════

TEST(BollingerTest, UpperGeqMidGeqLower) {
    std::vector<double> prices = {100, 102, 98, 105, 97, 103, 99, 106, 94, 101,
                                  100, 102, 98, 105, 97, 103, 99, 106, 94, 101,
                                  100, 102, 98, 105, 97};

    BollingerBands bb(5, 2.0);
    auto result = bb.compute_all(prices);

    for (size_t i = 4; i < result.mid.size(); ++i) {
        EXPECT_GE(result.upper[i], result.mid[i])
            << "upper >= mid at index " << i;
        EXPECT_GE(result.mid[i], result.lower[i])
            << "mid >= lower at index " << i;
    }
}

TEST(BollingerTest, WarmupNaN) {
    std::vector<double> prices(30, 100.0);

    BollingerBands bb(20, 2.0);
    auto result = bb.compute_all(prices);

    for (int i = 0; i < 19; ++i) {
        EXPECT_TRUE(std::isnan(result.mid[i])) << "mid at " << i;
        EXPECT_TRUE(std::isnan(result.upper[i])) << "upper at " << i;
        EXPECT_TRUE(std::isnan(result.lower[i])) << "lower at " << i;
    }
    EXPECT_FALSE(std::isnan(result.mid[19]));
}

TEST(BollingerTest, ConstantPricesBandsEqual) {
    std::vector<double> prices(30, 50.0);

    BollingerBands bb(5, 2.0);
    auto result = bb.compute_all(prices);

    for (size_t i = 4; i < result.mid.size(); ++i) {
        EXPECT_DOUBLE_EQ(result.upper[i], 50.0);
        EXPECT_DOUBLE_EQ(result.mid[i], 50.0);
        EXPECT_DOUBLE_EQ(result.lower[i], 50.0);
    }
}

TEST(BollingerTest, ComputeReturnsMidBand) {
    std::vector<double> prices = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    BollingerBands bb(5, 2.0);
    auto mid_only = bb.compute(prices);
    auto full = bb.compute_all(prices);

    for (size_t i = 0; i < mid_only.size(); ++i) {
        if (std::isnan(mid_only[i])) {
            EXPECT_TRUE(std::isnan(full.mid[i]));
        } else {
            EXPECT_DOUBLE_EQ(mid_only[i], full.mid[i]);
        }
    }
}

TEST(BollingerTest, KnownSMAReference) {
    // SMA(5) of [10, 11, 12, 13, 14] = 12.0
    std::vector<double> prices = {10, 11, 12, 13, 14, 15};

    BollingerBands bb(5, 2.0);
    auto result = bb.compute_all(prices);

    EXPECT_NEAR(result.mid[4], 12.0, 1e-10);
    EXPECT_NEAR(result.mid[5], 13.0, 1e-10);
}

TEST(BollingerTest, EmptyInput) {
    BollingerBands bb;
    auto result = bb.compute(std::span<const double>{});
    EXPECT_TRUE(result.empty());
}

TEST(BollingerTest, NameFormat) {
    BollingerBands bb(20, 2.0);
    EXPECT_EQ(bb.name().substr(0, 3), "BB(");
}

TEST(BollingerTest, WarmupPeriod) {
    BollingerBands bb(20, 2.0);
    EXPECT_EQ(bb.warmup_period(), 19);
}

TEST(BollingerTest, ZeroStdBandsEqualMid) {
    std::vector<double> prices = {100, 102, 98, 105, 97, 103, 99, 106, 94, 101};
    BollingerBands bb(5, 0.0);
    auto result = bb.compute_all(prices);

    for (size_t i = 4; i < result.mid.size(); ++i) {
        EXPECT_DOUBLE_EQ(result.upper[i], result.mid[i]);
        EXPECT_DOUBLE_EQ(result.lower[i], result.mid[i]);
    }
}

TEST(BollingerTest, InvalidPeriod) {
    EXPECT_THROW(auto b = BollingerBands(0, 2.0), std::invalid_argument);
    EXPECT_THROW(auto b = BollingerBands(20, -1.0), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════
// Garman-Klass Tests
// ═══════════════════════════════════════════════════════════════

TEST(GarmanKlassTest, WarmupNaN) {
    std::vector<double> o = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    std::vector<double> h = {102, 103, 104, 105, 106, 107, 108, 109, 110, 111};
    std::vector<double> l = {99, 100, 101, 102, 103, 104, 105, 106, 107, 108};
    std::vector<double> c = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110};

    GarmanKlass gk(5);
    auto result = gk.compute(o, h, l, c);
    EXPECT_EQ(result.size(), 10u);

    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(result[i])) << "index " << i;
    }
    for (size_t i = 4; i < result.size(); ++i) {
        EXPECT_FALSE(std::isnan(result[i])) << "index " << i;
    }
}

TEST(GarmanKlassTest, KnownReference) {
    // Single bar: H=110, L=90, O=100, C=105
    // GK_daily = 0.5 * ln(110/90)^2 - (2ln2-1) * ln(105/100)^2
    // ln(110/90) = ln(1.2222) ≈ 0.20067
    // ln(105/100) = ln(1.05) ≈ 0.04879
    // GK = 0.5 * 0.04027 - 0.3863 * 0.002381 = 0.020133 - 0.000920 = 0.019213
    // With window=1: annualized = sqrt(0.019213) * sqrt(252) ≈ 0.13862 * 15.8745 ≈ 2.200
    std::vector<double> o = {100};
    std::vector<double> h = {110};
    std::vector<double> l = {90};
    std::vector<double> c = {105};

    GarmanKlass gk(1);
    auto result = gk.compute(o, h, l, c);
    EXPECT_EQ(result.size(), 1u);

    // Compute expected precisely
    double log_hl = std::log(110.0 / 90.0);
    double log_co = std::log(105.0 / 100.0);
    double gk_daily = 0.5 * log_hl * log_hl
                    - (2.0 * std::numbers::ln2 - 1.0) * log_co * log_co;
    double expected = std::sqrt(gk_daily) * std::sqrt(252.0);
    EXPECT_NEAR(result[0], expected, 1e-10);
}

TEST(GarmanKlassTest, ConstantOHLCGivesZero) {
    // When H==L==O==C, all log ratios are 0
    std::vector<double> prices(10, 100.0);
    GarmanKlass gk(3);
    auto result = gk.compute(prices, prices, prices, prices);

    for (size_t i = 2; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], 0.0);
    }
}

TEST(GarmanKlassTest, EmptyInput) {
    GarmanKlass gk;
    auto result = gk.compute({}, {}, {}, {});
    EXPECT_TRUE(result.empty());
}

TEST(GarmanKlassTest, MismatchedLengths) {
    std::vector<double> a = {100, 101};
    std::vector<double> b = {100};
    GarmanKlass gk;
    EXPECT_THROW(auto r = gk.compute(a, a, a, b), std::invalid_argument);
}

TEST(GarmanKlassTest, InvalidWindow) {
    EXPECT_THROW(auto gk = GarmanKlass(0), std::invalid_argument);
}

TEST(GarmanKlassTest, NameFormat) {
    GarmanKlass gk(22);
    EXPECT_EQ(gk.name(), "GarmanKlass(22)");
}

TEST(GarmanKlassTest, WarmupPeriod) {
    GarmanKlass gk(22);
    EXPECT_EQ(gk.warmup_period(), 21);
}

TEST(GarmanKlassTest, RejectsZeroPrices) {
    std::vector<double> o = {100, 0};
    std::vector<double> h = {110, 110};
    std::vector<double> l = {90, 90};
    std::vector<double> c = {105, 105};
    GarmanKlass gk(1);
    EXPECT_THROW(auto r = gk.compute(o, h, l, c), std::invalid_argument);
}

TEST(GarmanKlassTest, RejectsNegativePrices) {
    std::vector<double> o = {100, 101};
    std::vector<double> h = {110, 110};
    std::vector<double> l = {90, -1};
    std::vector<double> c = {105, 105};
    GarmanKlass gk(1);
    EXPECT_THROW(auto r = gk.compute(o, h, l, c), std::invalid_argument);
}

TEST(GarmanKlassTest, RejectsHighLessThanLow) {
    std::vector<double> o = {100};
    std::vector<double> h = {90};   // invalid: high < low
    std::vector<double> l = {95};
    std::vector<double> c = {92};
    GarmanKlass gk(1);
    EXPECT_THROW(auto r = gk.compute(o, h, l, c), std::invalid_argument);
}

TEST(GarmanKlassTest, OutputIsNonNegative) {
    std::vector<double> o = {100, 101, 99, 102, 98, 103, 97, 104, 96, 105};
    std::vector<double> h = {105, 106, 104, 107, 103, 108, 102, 109, 101, 110};
    std::vector<double> l = {95, 96, 94, 97, 93, 98, 92, 99, 91, 100};
    std::vector<double> c = {101, 100, 102, 99, 103, 98, 104, 97, 105, 96};

    GarmanKlass gk(3);
    auto result = gk.compute(o, h, l, c);
    for (size_t i = 2; i < result.size(); ++i) {
        EXPECT_GE(result[i], 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════
// Parkinson Tests
// ═══════════════════════════════════════════════════════════════

TEST(ParkinsonTest, WarmupNaN) {
    std::vector<double> h = {102, 103, 104, 105, 106, 107, 108, 109, 110, 111};
    std::vector<double> l = {99, 100, 101, 102, 103, 104, 105, 106, 107, 108};
    // open/close not used by Parkinson but interface requires them
    std::vector<double> o = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    std::vector<double> c = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110};

    Parkinson pk(5);
    auto result = pk.compute(o, h, l, c);
    EXPECT_EQ(result.size(), 10u);

    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(result[i])) << "index " << i;
    }
    for (size_t i = 4; i < result.size(); ++i) {
        EXPECT_FALSE(std::isnan(result[i])) << "index " << i;
    }
}

TEST(ParkinsonTest, KnownReference) {
    // Single bar: H=110, L=90
    // PK_daily = (1/(4*ln(2))) * ln(110/90)^2
    std::vector<double> h = {110};
    std::vector<double> l = {90};
    std::vector<double> dummy = {100};

    Parkinson pk(1);
    auto result = pk.compute(dummy, h, l, dummy);

    double log_hl = std::log(110.0 / 90.0);
    double pk_daily = (1.0 / (4.0 * std::numbers::ln2)) * log_hl * log_hl;
    double expected = std::sqrt(pk_daily) * std::sqrt(252.0);
    EXPECT_NEAR(result[0], expected, 1e-10);
}

TEST(ParkinsonTest, EqualHighLowGivesZero) {
    std::vector<double> prices(10, 100.0);
    Parkinson pk(3);
    auto result = pk.compute(prices, prices, prices, prices);

    for (size_t i = 2; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], 0.0);
    }
}

TEST(ParkinsonTest, EmptyInput) {
    Parkinson pk;
    auto result = pk.compute({}, {}, {}, {});
    EXPECT_TRUE(result.empty());
}

TEST(ParkinsonTest, RejectsZeroPrices) {
    std::vector<double> h = {110, 0};
    std::vector<double> l = {90, 90};
    std::vector<double> dummy = {100, 100};
    Parkinson pk(1);
    EXPECT_THROW(auto r = pk.compute(dummy, h, l, dummy), std::invalid_argument);
}

TEST(ParkinsonTest, MismatchedLengths) {
    std::vector<double> a = {100, 101};
    std::vector<double> b = {100};
    Parkinson pk;
    EXPECT_THROW(auto r = pk.compute(a, a, a, b), std::invalid_argument);
}

TEST(ParkinsonTest, RejectsHighLessThanLow) {
    std::vector<double> o = {100};
    std::vector<double> h = {90};   // invalid: high < low
    std::vector<double> l = {95};
    std::vector<double> c = {92};
    Parkinson pk(1);
    EXPECT_THROW(auto r = pk.compute(o, h, l, c), std::invalid_argument);
}

TEST(ParkinsonTest, InvalidWindow) {
    EXPECT_THROW(auto pk = Parkinson(0), std::invalid_argument);
}

TEST(ParkinsonTest, OutputIsNonNegative) {
    std::vector<double> h = {105, 106, 104, 107, 103, 108, 102, 109, 101, 110};
    std::vector<double> l = {95, 96, 94, 97, 93, 98, 92, 99, 91, 100};
    std::vector<double> dummy = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};

    Parkinson pk(3);
    auto result = pk.compute(dummy, h, l, dummy);
    for (size_t i = 2; i < result.size(); ++i) {
        EXPECT_GE(result[i], 0.0);
    }
}

TEST(ParkinsonTest, NameFormat) {
    Parkinson pk(22);
    EXPECT_EQ(pk.name(), "Parkinson(22)");
}

TEST(ParkinsonTest, WarmupPeriod) {
    Parkinson pk(22);
    EXPECT_EQ(pk.warmup_period(), 21);
}

// ═══════════════════════════════════════════════════════════════
// Rolling Helpers Tests
// ═══════════════════════════════════════════════════════════════

TEST(RollingTest, MeanWindow1) {
    std::vector<double> data = {10, 20, 30, 40, 50};
    auto result = detail::rolling_mean(data, 1);
    EXPECT_EQ(result.size(), 5u);
    EXPECT_DOUBLE_EQ(result[0], 10.0);
    EXPECT_DOUBLE_EQ(result[4], 50.0);
}

TEST(RollingTest, MeanWindowEqualsN) {
    std::vector<double> data = {10, 20, 30};
    auto result = detail::rolling_mean(data, 3);
    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isnan(result[1]));
    EXPECT_NEAR(result[2], 20.0, 1e-10);
}

TEST(RollingTest, MeanWindowGreaterThanN) {
    std::vector<double> data = {10, 20};
    auto result = detail::rolling_mean(data, 5);
    for (const auto& v : result) {
        EXPECT_TRUE(std::isnan(v));
    }
}

TEST(RollingTest, MeanEmptyInput) {
    auto result = detail::rolling_mean({}, 3);
    EXPECT_TRUE(result.empty());
}

TEST(RollingTest, StdConstantValues) {
    // Std of constant values should be 0
    std::vector<double> data(10, 42.0);
    auto result = detail::rolling_std(data, 5);
    for (size_t i = 4; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], 0.0);
    }
}

TEST(RollingTest, StdKnownReference) {
    // std of [1, 2, 3, 4, 5] with ddof=1
    // mean=3, deviations=[-2,-1,0,1,2], sum_sq_dev=10, var=10/4=2.5, std=sqrt(2.5)
    std::vector<double> data = {1, 2, 3, 4, 5};
    auto result = detail::rolling_std(data, 5, 1);
    EXPECT_NEAR(result[4], std::sqrt(2.5), 1e-10);
}

TEST(RollingTest, StdPopulation) {
    // Same data, ddof=0: var=10/5=2.0, std=sqrt(2.0)
    std::vector<double> data = {1, 2, 3, 4, 5};
    auto result = detail::rolling_std(data, 5, 0);
    EXPECT_NEAR(result[4], std::sqrt(2.0), 1e-10);
}

TEST(RollingTest, StdWindow1) {
    // Window=1 with ddof=0: std should be 0 (single element)
    std::vector<double> data = {10, 20, 30};
    auto result = detail::rolling_std(data, 1, 0);
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
}

TEST(RollingTest, StdWindowGreaterThanN) {
    std::vector<double> data = {10, 20};
    auto result = detail::rolling_std(data, 5);
    for (const auto& v : result) {
        EXPECT_TRUE(std::isnan(v));
    }
}

TEST(RollingTest, StdSlidingMultipleOutputs) {
    // rolling_std([1,2,3,4,5,6], window=3, ddof=1)
    // Window [1,2,3]: mean=2, var=1, std=1.0
    // Window [2,3,4]: mean=3, var=1, std=1.0
    // Window [3,4,5]: mean=4, var=1, std=1.0
    // Window [4,5,6]: mean=5, var=1, std=1.0
    std::vector<double> data = {1, 2, 3, 4, 5, 6};
    auto result = detail::rolling_std(data, 3, 1);

    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isnan(result[1]));
    EXPECT_NEAR(result[2], 1.0, 1e-10);
    EXPECT_NEAR(result[3], 1.0, 1e-10);
    EXPECT_NEAR(result[4], 1.0, 1e-10);
    EXPECT_NEAR(result[5], 1.0, 1e-10);
}

TEST(RollingTest, StdSlidingVaryingValues) {
    // rolling_std([10, 20, 10, 20], window=2, ddof=1)
    // [10,20]: mean=15, var=50, std=sqrt(50)≈7.071
    // [20,10]: same
    // [10,20]: same
    std::vector<double> data = {10, 20, 10, 20};
    auto result = detail::rolling_std(data, 2, 1);

    EXPECT_TRUE(std::isnan(result[0]));
    double expected = std::sqrt(50.0);
    EXPECT_NEAR(result[1], expected, 1e-10);
    EXPECT_NEAR(result[2], expected, 1e-10);
    EXPECT_NEAR(result[3], expected, 1e-10);
}

TEST(RollingTest, StdNumericalStabilityLargeValues) {
    // Welford's should handle large values without cancellation
    // Data: [1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4, 1e9 + 5]
    // Same std as [1,2,3,4,5] = sqrt(2.5)
    std::vector<double> data = {1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4, 1e9 + 5};
    auto result = detail::rolling_std(data, 5, 1);
    EXPECT_NEAR(result[4], std::sqrt(2.5), 1e-6);
}

TEST(RollingTest, StdWindow1Ddof1AllNaN) {
    // Sample std with window=1 and ddof=1 is undefined (0 degrees of freedom)
    std::vector<double> data = {42.0, 99.0, 7.0};
    auto result = detail::rolling_std(data, 1, 1);
    EXPECT_EQ(result.size(), 3u);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(std::isnan(result[i])) << "index " << i;
    }
}

}  // namespace
}  // namespace quant
