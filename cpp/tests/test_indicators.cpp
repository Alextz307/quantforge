#include <cmath>
#include <numbers>
#include <vector>

#include <gtest/gtest.h>

#include "quant/core/types.hpp"
#include "quant/indicators/bollinger_bands.hpp"
#include "quant/indicators/detail/rolling.hpp"
#include "quant/indicators/garman_klass.hpp"
#include "quant/indicators/macd.hpp"
#include "quant/indicators/parkinson.hpp"
#include "quant/indicators/rsi.hpp"

namespace quant {
namespace {

constexpr int kRSIDefaultPeriod = 14;
constexpr int kRSILongerSeriesLen = 100;
constexpr int kRSIShortSeriesLen = 20;
constexpr int kRSIDeepIndex = 50;
constexpr int kRSIPeriod3 = 3;
constexpr int kRSIPeriod5 = 5;
constexpr int kRSIPeriod1 = 1;
constexpr double kRSIBaseRamp = 100.0;
constexpr double kRSIRampStep = 0.5;
constexpr double kRSIDescBase = 200.0;
constexpr double kRSIConstantPrice = 100.0;
constexpr double kRSINeutral = 50.0;
constexpr double kRSIFloor = 0.0;
constexpr double kRSICeiling = 100.0;
constexpr double kRSIReferenceTolerance = 0.001;
constexpr double kRSIExpectedAtIdx3 = 66.6667;
constexpr double kRSIExpectedAtIdx4 = 83.3333;

constexpr int kMACDFastPeriod = 12;
constexpr int kMACDSlowPeriod = 26;
constexpr int kMACDSignalPeriod = 9;
constexpr int kMACDWarmupBars = 25;
constexpr int kMACDSignalWarmupBars = 33;
constexpr int kMACDSeriesLen = 50;
constexpr int kMACDLongerSeriesLen = 60;
constexpr double kMACDRampBase = 100.0;
constexpr double kMACDRampStep = 0.5;
constexpr double kMACDSineAmplitude = 5.0;
constexpr double kMACDSineFreq = 0.3;
constexpr double kMACDExactTolerance = 1e-10;

constexpr int kBBDefaultWindow = 20;
constexpr int kBBSmallWindow = 5;
constexpr int kBBDefaultWarmup = 19;
constexpr double kBBDefaultK = 2.0;
constexpr double kBBZeroK = 0.0;
constexpr double kBBConstantPrice = 50.0;
constexpr double kBBExactTolerance = 1e-10;

constexpr int kGKDefaultWindow = 22;
constexpr int kGKSmallWindow = 5;
constexpr int kGKDefaultWarmup = 21;
constexpr int kGKSingleBarWindow = 1;
constexpr int kGKConstWindow = 3;
constexpr double kGKSingleBarHigh = 110.0;
constexpr double kGKSingleBarLow = 90.0;
constexpr double kGKSingleBarOpen = 100.0;
constexpr double kGKSingleBarClose = 105.0;
constexpr double kGKExactTolerance = 1e-10;

constexpr int kPKDefaultWindow = 22;
constexpr int kPKSmallWindow = 5;
constexpr int kPKDefaultWarmup = 21;
constexpr int kPKSingleBarWindow = 1;
constexpr int kPKConstWindow = 3;
constexpr double kPKExactTolerance = 1e-10;

constexpr int kRollingWindow1 = 1;
constexpr int kRollingWindow2 = 2;
constexpr int kRollingWindow3 = 3;
constexpr int kRollingWindow5 = 5;
constexpr int kRollingDdofSample = 1;
constexpr int kRollingDdofPopulation = 0;
constexpr double kRollingExactTolerance = 1e-10;
constexpr double kRollingLargeValuesTolerance = 1e-6;
constexpr double kRollingConstantValue = 42.0;
constexpr double kLargeValueOffset = 1e9;

TEST(RSITest, WarmupIsNaN) {
    std::vector<double> prices(kRSIShortSeriesLen, kRSIBaseRamp);
    for (int i = 0; i < kRSIShortSeriesLen; ++i) prices[i] = kRSIBaseRamp + i * kRSIRampStep;

    RSI rsi(kRSIDefaultPeriod);
    auto result = rsi.compute(prices);

    EXPECT_EQ(result.size(), prices.size());
    for (int i = 0; i < kRSIDefaultPeriod; ++i) {
        EXPECT_TRUE(std::isnan(result[i])) << "Index " << i << " should be NaN";
    }
    for (size_t i = kRSIDefaultPeriod; i < result.size(); ++i) {
        EXPECT_FALSE(std::isnan(result[i])) << "Index " << i << " should not be NaN";
    }
}

TEST(RSITest, MonotonicallyIncreasingApproaches100) {
    std::vector<double> prices(kRSILongerSeriesLen);
    for (int i = 0; i < kRSILongerSeriesLen; ++i) prices[i] = kRSIBaseRamp + i;

    RSI rsi(kRSIDefaultPeriod);
    auto result = rsi.compute(prices);

    EXPECT_DOUBLE_EQ(result[kRSIDefaultPeriod], kRSICeiling);
    EXPECT_DOUBLE_EQ(result[kRSIDeepIndex], kRSICeiling);
}

TEST(RSITest, MonotonicallyDecreasingApproaches0) {
    std::vector<double> prices(kRSILongerSeriesLen);
    for (int i = 0; i < kRSILongerSeriesLen; ++i) prices[i] = kRSIDescBase - i;

    RSI rsi(kRSIDefaultPeriod);
    auto result = rsi.compute(prices);

    EXPECT_DOUBLE_EQ(result[kRSIDefaultPeriod], kRSIFloor);
    EXPECT_DOUBLE_EQ(result[kRSIDeepIndex], kRSIFloor);
}

TEST(RSITest, ConstantPricesGive50) {
    std::vector<double> prices(30, kRSIConstantPrice);

    RSI rsi(kRSIDefaultPeriod);
    auto result = rsi.compute(prices);

    EXPECT_DOUBLE_EQ(result[kRSIDefaultPeriod], kRSINeutral);
    EXPECT_DOUBLE_EQ(result[20], kRSINeutral);
}

TEST(RSITest, KnownReferenceValue) {
    // Hand-computed RSI(3) for prices [10, 11, 12, 11, 13, 14, 12, 15]
    // Deltas: +1, +1, -1, +2, +1, -2, +3
    // Seed: avg_gain = 0.6667, avg_loss = 0.3333 → RSI[3] = 66.667
    // RSI[4]: avg_gain=1.1111, avg_loss=0.2222 → RS=5.0, RSI=83.333
    std::vector<double> prices = {10, 11, 12, 11, 13, 14, 12, 15};
    RSI rsi(kRSIPeriod3);
    auto result = rsi.compute(prices);

    EXPECT_NEAR(result[3], kRSIExpectedAtIdx3, kRSIReferenceTolerance);
    EXPECT_NEAR(result[4], kRSIExpectedAtIdx4, kRSIReferenceTolerance);
}

TEST(RSITest, EmptyInput) {
    RSI rsi(kRSIDefaultPeriod);
    auto result = rsi.compute(std::span<const double>{});
    EXPECT_TRUE(result.empty());
}

TEST(RSITest, TooFewPrices) {
    std::vector<double> prices = {100.0, 101.0, 102.0};
    RSI rsi(kRSIDefaultPeriod);
    auto result = rsi.compute(prices);
    EXPECT_EQ(result.size(), 3u);
    for (const auto& v : result) {
        EXPECT_TRUE(std::isnan(v));
    }
}

TEST(RSITest, NameFormat) {
    RSI rsi(kRSIDefaultPeriod);
    EXPECT_EQ(rsi.name(), "RSI(14)");
}

TEST(RSITest, WarmupPeriod) {
    RSI rsi(kRSIDefaultPeriod);
    EXPECT_EQ(rsi.warmup_period(), kRSIDefaultPeriod);
}

TEST(RSITest, InvalidPeriod) {
    EXPECT_THROW(auto r = RSI(0), std::invalid_argument);
    EXPECT_THROW(auto r = RSI(-1), std::invalid_argument);
}

TEST(RSITest, Period1) {
    std::vector<double> prices = {100, 102, 101, 103};
    RSI rsi(kRSIPeriod1);
    auto result = rsi.compute(prices);

    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_DOUBLE_EQ(result[1], kRSICeiling);
    EXPECT_DOUBLE_EQ(result[2], kRSIFloor);
    EXPECT_DOUBLE_EQ(result[3], kRSICeiling);
}

TEST(RSITest, RangeIsBounded) {
    std::vector<double> prices = {100, 105, 95, 110, 85, 120, 80, 115, 90, 105,
                                  95, 110, 85, 120, 80, 115, 90, 105, 95, 110};
    RSI rsi(kRSIPeriod5);
    auto result = rsi.compute(prices);
    for (size_t i = kRSIPeriod5; i < result.size(); ++i) {
        EXPECT_GE(result[i], kRSIFloor);
        EXPECT_LE(result[i], kRSICeiling);
    }
}

TEST(MACDTest, OutputSameLength) {
    std::vector<double> prices(kMACDSeriesLen);
    for (int i = 0; i < kMACDSeriesLen; ++i) prices[i] = kMACDRampBase + i;

    MACD macd;
    auto result = macd.compute(prices);
    EXPECT_EQ(result.size(), prices.size());

    auto full = macd.compute_all(prices);
    EXPECT_EQ(full.macd_line.size(), prices.size());
    EXPECT_EQ(full.signal_line.size(), prices.size());
    EXPECT_EQ(full.histogram.size(), prices.size());
}

TEST(MACDTest, WarmupNaN) {
    std::vector<double> prices(kMACDSeriesLen);
    for (int i = 0; i < kMACDSeriesLen; ++i) prices[i] = kMACDRampBase + i * kMACDRampStep;

    MACD macd(kMACDFastPeriod, kMACDSlowPeriod, kMACDSignalPeriod);
    auto result = macd.compute(prices);

    for (int i = 0; i < kMACDWarmupBars; ++i) {
        EXPECT_TRUE(std::isnan(result[i])) << "MACD line index " << i;
    }
    EXPECT_FALSE(std::isnan(result[kMACDWarmupBars]));
}

TEST(MACDTest, HistogramIsLineMinusSignal) {
    std::vector<double> prices(kMACDLongerSeriesLen);
    for (int i = 0; i < kMACDLongerSeriesLen; ++i) {
        prices[i] = kMACDRampBase + std::sin(i * kMACDSineFreq) * kMACDSineAmplitude;
    }

    MACD macd;
    auto full = macd.compute_all(prices);

    for (size_t i = 0; i < full.histogram.size(); ++i) {
        if (!std::isnan(full.histogram[i])) {
            EXPECT_NEAR(full.histogram[i],
                        full.macd_line[i] - full.signal_line[i], kMACDExactTolerance)
                << "at index " << i;
        }
    }
}

TEST(MACDTest, ConstantPricesMACDIsZero) {
    std::vector<double> prices(kMACDSeriesLen, kMACDRampBase);

    MACD macd;
    auto full = macd.compute_all(prices);

    for (size_t i = kMACDWarmupBars; i < full.macd_line.size(); ++i) {
        EXPECT_NEAR(full.macd_line[i], 0.0, kMACDExactTolerance);
    }
}

TEST(MACDTest, EmptyInput) {
    MACD macd;
    auto result = macd.compute(std::span<const double>{});
    EXPECT_TRUE(result.empty());
}

TEST(MACDTest, NameFormat) {
    MACD macd(kMACDFastPeriod, kMACDSlowPeriod, kMACDSignalPeriod);
    EXPECT_EQ(macd.name(), "MACD(12,26,9)");
}

TEST(MACDTest, WarmupPeriod) {
    MACD macd(kMACDFastPeriod, kMACDSlowPeriod, kMACDSignalPeriod);
    EXPECT_EQ(macd.warmup_period(), kMACDWarmupBars);
}

TEST(MACDTest, SignalLineWarmup) {
    std::vector<double> prices(kMACDSeriesLen);
    for (int i = 0; i < kMACDSeriesLen; ++i) prices[i] = kMACDRampBase + i * kMACDRampStep;

    MACD macd(kMACDFastPeriod, kMACDSlowPeriod, kMACDSignalPeriod);
    auto full = macd.compute_all(prices);

    for (int i = 0; i < kMACDSignalWarmupBars; ++i) {
        EXPECT_TRUE(std::isnan(full.signal_line[i])) << "signal at " << i;
        EXPECT_TRUE(std::isnan(full.histogram[i])) << "histogram at " << i;
    }
    EXPECT_FALSE(std::isnan(full.signal_line[kMACDSignalWarmupBars]));
    EXPECT_FALSE(std::isnan(full.histogram[kMACDSignalWarmupBars]));
}

TEST(MACDTest, InvalidPeriods) {
    EXPECT_THROW(auto m = MACD(0, kMACDSlowPeriod, kMACDSignalPeriod), std::invalid_argument);
    EXPECT_THROW(auto m = MACD(kMACDFastPeriod, kMACDFastPeriod, kMACDSignalPeriod),
                 std::invalid_argument);
    EXPECT_THROW(auto m = MACD(30, kMACDSlowPeriod, kMACDSignalPeriod), std::invalid_argument);
}

TEST(BollingerTest, UpperGeqMidGeqLower) {
    std::vector<double> prices = {100, 102, 98, 105, 97, 103, 99, 106, 94, 101,
                                  100, 102, 98, 105, 97, 103, 99, 106, 94, 101,
                                  100, 102, 98, 105, 97};

    BollingerBands bb(kBBSmallWindow, kBBDefaultK);
    auto result = bb.compute_all(prices);

    for (size_t i = kBBSmallWindow - 1; i < result.mid.size(); ++i) {
        EXPECT_GE(result.upper[i], result.mid[i]) << "upper >= mid at index " << i;
        EXPECT_GE(result.mid[i], result.lower[i]) << "mid >= lower at index " << i;
    }
}

TEST(BollingerTest, WarmupNaN) {
    std::vector<double> prices(30, 100.0);

    BollingerBands bb(kBBDefaultWindow, kBBDefaultK);
    auto result = bb.compute_all(prices);

    for (int i = 0; i < kBBDefaultWarmup; ++i) {
        EXPECT_TRUE(std::isnan(result.mid[i])) << "mid at " << i;
        EXPECT_TRUE(std::isnan(result.upper[i])) << "upper at " << i;
        EXPECT_TRUE(std::isnan(result.lower[i])) << "lower at " << i;
    }
    EXPECT_FALSE(std::isnan(result.mid[kBBDefaultWarmup]));
}

TEST(BollingerTest, ConstantPricesBandsEqual) {
    std::vector<double> prices(30, kBBConstantPrice);

    BollingerBands bb(kBBSmallWindow, kBBDefaultK);
    auto result = bb.compute_all(prices);

    for (size_t i = kBBSmallWindow - 1; i < result.mid.size(); ++i) {
        EXPECT_DOUBLE_EQ(result.upper[i], kBBConstantPrice);
        EXPECT_DOUBLE_EQ(result.mid[i], kBBConstantPrice);
        EXPECT_DOUBLE_EQ(result.lower[i], kBBConstantPrice);
    }
}

TEST(BollingerTest, ComputeReturnsMidBand) {
    std::vector<double> prices = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    BollingerBands bb(kBBSmallWindow, kBBDefaultK);
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
    std::vector<double> prices = {10, 11, 12, 13, 14, 15};

    BollingerBands bb(kBBSmallWindow, kBBDefaultK);
    auto result = bb.compute_all(prices);

    EXPECT_NEAR(result.mid[4], 12.0, kBBExactTolerance);
    EXPECT_NEAR(result.mid[5], 13.0, kBBExactTolerance);
}

TEST(BollingerTest, EmptyInput) {
    BollingerBands bb;
    auto result = bb.compute(std::span<const double>{});
    EXPECT_TRUE(result.empty());
}

TEST(BollingerTest, NameFormat) {
    BollingerBands bb(kBBDefaultWindow, kBBDefaultK);
    EXPECT_EQ(bb.name().substr(0, 3), "BB(");
}

TEST(BollingerTest, WarmupPeriod) {
    BollingerBands bb(kBBDefaultWindow, kBBDefaultK);
    EXPECT_EQ(bb.warmup_period(), kBBDefaultWarmup);
}

TEST(BollingerTest, ZeroStdBandsEqualMid) {
    std::vector<double> prices = {100, 102, 98, 105, 97, 103, 99, 106, 94, 101};
    BollingerBands bb(kBBSmallWindow, kBBZeroK);
    auto result = bb.compute_all(prices);

    for (size_t i = kBBSmallWindow - 1; i < result.mid.size(); ++i) {
        EXPECT_DOUBLE_EQ(result.upper[i], result.mid[i]);
        EXPECT_DOUBLE_EQ(result.lower[i], result.mid[i]);
    }
}

TEST(BollingerTest, InvalidPeriod) {
    EXPECT_THROW(auto b = BollingerBands(0, kBBDefaultK), std::invalid_argument);
    EXPECT_THROW(auto b = BollingerBands(kBBDefaultWindow, -1.0), std::invalid_argument);
}

TEST(GarmanKlassTest, WarmupNaN) {
    std::vector<double> o = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    std::vector<double> h = {102, 103, 104, 105, 106, 107, 108, 109, 110, 111};
    std::vector<double> l = {99, 100, 101, 102, 103, 104, 105, 106, 107, 108};
    std::vector<double> c = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110};

    GarmanKlass gk(kGKSmallWindow);
    auto result = gk.compute(o, h, l, c);
    EXPECT_EQ(result.size(), 10u);

    for (int i = 0; i < kGKSmallWindow - 1; ++i) {
        EXPECT_TRUE(std::isnan(result[i])) << "index " << i;
    }
    for (size_t i = kGKSmallWindow - 1; i < result.size(); ++i) {
        EXPECT_FALSE(std::isnan(result[i])) << "index " << i;
    }
}

TEST(GarmanKlassTest, KnownReference) {
    std::vector<double> o = {kGKSingleBarOpen};
    std::vector<double> h = {kGKSingleBarHigh};
    std::vector<double> l = {kGKSingleBarLow};
    std::vector<double> c = {kGKSingleBarClose};

    GarmanKlass gk(kGKSingleBarWindow);
    auto result = gk.compute(o, h, l, c);
    EXPECT_EQ(result.size(), 1u);

    double log_hl = std::log(kGKSingleBarHigh / kGKSingleBarLow);
    double log_co = std::log(kGKSingleBarClose / kGKSingleBarOpen);
    double gk_daily = 0.5 * log_hl * log_hl
                    - (2.0 * std::numbers::ln2 - 1.0) * log_co * log_co;
    double expected = std::sqrt(gk_daily) * std::sqrt(static_cast<double>(kTradingDaysPerYear));
    EXPECT_NEAR(result[0], expected, kGKExactTolerance);
}

TEST(GarmanKlassTest, ConstantOHLCGivesZero) {
    std::vector<double> prices(10, 100.0);
    GarmanKlass gk(kGKConstWindow);
    auto result = gk.compute(prices, prices, prices, prices);

    for (size_t i = kGKConstWindow - 1; i < result.size(); ++i) {
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
    GarmanKlass gk(kGKDefaultWindow);
    EXPECT_EQ(gk.name(), "GarmanKlass(22)");
}

TEST(GarmanKlassTest, WarmupPeriod) {
    GarmanKlass gk(kGKDefaultWindow);
    EXPECT_EQ(gk.warmup_period(), kGKDefaultWarmup);
}

TEST(GarmanKlassTest, RejectsZeroPrices) {
    std::vector<double> o = {100, 0};
    std::vector<double> h = {110, 110};
    std::vector<double> l = {90, 90};
    std::vector<double> c = {105, 105};
    GarmanKlass gk(kGKSingleBarWindow);
    EXPECT_THROW(auto r = gk.compute(o, h, l, c), std::invalid_argument);
}

TEST(GarmanKlassTest, RejectsNegativePrices) {
    std::vector<double> o = {100, 101};
    std::vector<double> h = {110, 110};
    std::vector<double> l = {90, -1};
    std::vector<double> c = {105, 105};
    GarmanKlass gk(kGKSingleBarWindow);
    EXPECT_THROW(auto r = gk.compute(o, h, l, c), std::invalid_argument);
}

TEST(GarmanKlassTest, RejectsHighLessThanLow) {
    std::vector<double> o = {100};
    std::vector<double> h = {90};
    std::vector<double> l = {95};
    std::vector<double> c = {92};
    GarmanKlass gk(kGKSingleBarWindow);
    EXPECT_THROW(auto r = gk.compute(o, h, l, c), std::invalid_argument);
}

TEST(GarmanKlassTest, OutputIsNonNegative) {
    std::vector<double> o = {100, 101, 99, 102, 98, 103, 97, 104, 96, 105};
    std::vector<double> h = {105, 106, 104, 107, 103, 108, 102, 109, 101, 110};
    std::vector<double> l = {95, 96, 94, 97, 93, 98, 92, 99, 91, 100};
    std::vector<double> c = {101, 100, 102, 99, 103, 98, 104, 97, 105, 96};

    GarmanKlass gk(kGKConstWindow);
    auto result = gk.compute(o, h, l, c);
    for (size_t i = kGKConstWindow - 1; i < result.size(); ++i) {
        EXPECT_GE(result[i], 0.0);
    }
}

TEST(ParkinsonTest, WarmupNaN) {
    std::vector<double> h = {102, 103, 104, 105, 106, 107, 108, 109, 110, 111};
    std::vector<double> l = {99, 100, 101, 102, 103, 104, 105, 106, 107, 108};
    // open/close not used by Parkinson but interface requires them
    std::vector<double> o = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    std::vector<double> c = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110};

    Parkinson pk(kPKSmallWindow);
    auto result = pk.compute(o, h, l, c);
    EXPECT_EQ(result.size(), 10u);

    for (int i = 0; i < kPKSmallWindow - 1; ++i) {
        EXPECT_TRUE(std::isnan(result[i])) << "index " << i;
    }
    for (size_t i = kPKSmallWindow - 1; i < result.size(); ++i) {
        EXPECT_FALSE(std::isnan(result[i])) << "index " << i;
    }
}

TEST(ParkinsonTest, KnownReference) {
    std::vector<double> h = {kGKSingleBarHigh};
    std::vector<double> l = {kGKSingleBarLow};
    std::vector<double> dummy = {kGKSingleBarOpen};

    Parkinson pk(kPKSingleBarWindow);
    auto result = pk.compute(dummy, h, l, dummy);

    double log_hl = std::log(kGKSingleBarHigh / kGKSingleBarLow);
    double pk_daily = (1.0 / (4.0 * std::numbers::ln2)) * log_hl * log_hl;
    double expected = std::sqrt(pk_daily) * std::sqrt(static_cast<double>(kTradingDaysPerYear));
    EXPECT_NEAR(result[0], expected, kPKExactTolerance);
}

TEST(ParkinsonTest, EqualHighLowGivesZero) {
    std::vector<double> prices(10, 100.0);
    Parkinson pk(kPKConstWindow);
    auto result = pk.compute(prices, prices, prices, prices);

    for (size_t i = kPKConstWindow - 1; i < result.size(); ++i) {
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
    Parkinson pk(kPKSingleBarWindow);
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
    std::vector<double> h = {90};
    std::vector<double> l = {95};
    std::vector<double> c = {92};
    Parkinson pk(kPKSingleBarWindow);
    EXPECT_THROW(auto r = pk.compute(o, h, l, c), std::invalid_argument);
}

TEST(ParkinsonTest, InvalidWindow) {
    EXPECT_THROW(auto pk = Parkinson(0), std::invalid_argument);
}

TEST(ParkinsonTest, OutputIsNonNegative) {
    std::vector<double> h = {105, 106, 104, 107, 103, 108, 102, 109, 101, 110};
    std::vector<double> l = {95, 96, 94, 97, 93, 98, 92, 99, 91, 100};
    std::vector<double> dummy = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};

    Parkinson pk(kPKConstWindow);
    auto result = pk.compute(dummy, h, l, dummy);
    for (size_t i = kPKConstWindow - 1; i < result.size(); ++i) {
        EXPECT_GE(result[i], 0.0);
    }
}

TEST(ParkinsonTest, NameFormat) {
    Parkinson pk(kPKDefaultWindow);
    EXPECT_EQ(pk.name(), "Parkinson(22)");
}

TEST(ParkinsonTest, WarmupPeriod) {
    Parkinson pk(kPKDefaultWindow);
    EXPECT_EQ(pk.warmup_period(), kPKDefaultWarmup);
}

TEST(RollingTest, MeanWindow1) {
    std::vector<double> data = {10, 20, 30, 40, 50};
    auto result = detail::rolling_mean(data, kRollingWindow1);
    EXPECT_EQ(result.size(), 5u);
    EXPECT_DOUBLE_EQ(result[0], 10.0);
    EXPECT_DOUBLE_EQ(result[4], 50.0);
}

TEST(RollingTest, MeanWindowEqualsN) {
    std::vector<double> data = {10, 20, 30};
    auto result = detail::rolling_mean(data, kRollingWindow3);
    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isnan(result[1]));
    EXPECT_NEAR(result[2], 20.0, kRollingExactTolerance);
}

TEST(RollingTest, MeanWindowGreaterThanN) {
    std::vector<double> data = {10, 20};
    auto result = detail::rolling_mean(data, kRollingWindow5);
    for (const auto& v : result) {
        EXPECT_TRUE(std::isnan(v));
    }
}

TEST(RollingTest, MeanEmptyInput) {
    auto result = detail::rolling_mean({}, kRollingWindow3);
    EXPECT_TRUE(result.empty());
}

TEST(RollingTest, StdConstantValues) {
    std::vector<double> data(10, kRollingConstantValue);
    auto result = detail::rolling_std(data, kRollingWindow5);
    for (size_t i = kRollingWindow5 - 1; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], 0.0);
    }
}

TEST(RollingTest, StdKnownReference) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    auto result = detail::rolling_std(data, kRollingWindow5, kRollingDdofSample);
    EXPECT_NEAR(result[4], std::sqrt(2.5), kRollingExactTolerance);
}

TEST(RollingTest, StdPopulation) {
    std::vector<double> data = {1, 2, 3, 4, 5};
    auto result = detail::rolling_std(data, kRollingWindow5, kRollingDdofPopulation);
    EXPECT_NEAR(result[4], std::sqrt(2.0), kRollingExactTolerance);
}

TEST(RollingTest, StdWindow1) {
    std::vector<double> data = {10, 20, 30};
    auto result = detail::rolling_std(data, kRollingWindow1, kRollingDdofPopulation);
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
}

TEST(RollingTest, StdWindowGreaterThanN) {
    std::vector<double> data = {10, 20};
    auto result = detail::rolling_std(data, kRollingWindow5);
    for (const auto& v : result) {
        EXPECT_TRUE(std::isnan(v));
    }
}

TEST(RollingTest, StdSlidingMultipleOutputs) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6};
    auto result = detail::rolling_std(data, kRollingWindow3, kRollingDdofSample);

    EXPECT_TRUE(std::isnan(result[0]));
    EXPECT_TRUE(std::isnan(result[1]));
    EXPECT_NEAR(result[2], 1.0, kRollingExactTolerance);
    EXPECT_NEAR(result[3], 1.0, kRollingExactTolerance);
    EXPECT_NEAR(result[4], 1.0, kRollingExactTolerance);
    EXPECT_NEAR(result[5], 1.0, kRollingExactTolerance);
}

TEST(RollingTest, StdSlidingVaryingValues) {
    std::vector<double> data = {10, 20, 10, 20};
    auto result = detail::rolling_std(data, kRollingWindow2, kRollingDdofSample);

    EXPECT_TRUE(std::isnan(result[0]));
    double expected = std::sqrt(50.0);
    EXPECT_NEAR(result[1], expected, kRollingExactTolerance);
    EXPECT_NEAR(result[2], expected, kRollingExactTolerance);
    EXPECT_NEAR(result[3], expected, kRollingExactTolerance);
}

TEST(RollingTest, StdNumericalStabilityLargeValues) {
    // Welford's algorithm must stay stable on large-magnitude inputs.
    // Data: [1e9 + 1, ..., 1e9 + 5] has same std as [1..5] = sqrt(2.5).
    std::vector<double> data = {kLargeValueOffset + 1, kLargeValueOffset + 2,
                                 kLargeValueOffset + 3, kLargeValueOffset + 4,
                                 kLargeValueOffset + 5};
    auto result = detail::rolling_std(data, kRollingWindow5, kRollingDdofSample);
    EXPECT_NEAR(result[4], std::sqrt(2.5), kRollingLargeValuesTolerance);
}

TEST(RollingTest, StdWindow1Ddof1AllNaN) {
    std::vector<double> data = {kRollingConstantValue, 99.0, 7.0};
    auto result = detail::rolling_std(data, kRollingWindow1, kRollingDdofSample);
    EXPECT_EQ(result.size(), 3u);
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_TRUE(std::isnan(result[i])) << "index " << i;
    }
}

}  // namespace
}  // namespace quant
