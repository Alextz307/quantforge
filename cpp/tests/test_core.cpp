#include <limits>

#include <gtest/gtest.h>

#include "quant/core/time_series.hpp"
#include "quant/core/types.hpp"

namespace quant {
namespace {

constexpr int64_t kSampleTimestampS = 1000;
constexpr double kValidOpen = 100.0;
constexpr double kValidHigh = 105.0;
constexpr double kValidLow = 95.0;
constexpr double kValidClose = 102.0;
constexpr double kSampleVolume = 1000.0;

constexpr double kInvalidHighBelowLow = 90.0;
constexpr double kInvalidHighBelowClose = 101.0;
constexpr double kInvalidLowAboveOpen = 101.0;
constexpr double kInvalidLowBelowClose = 92.0;
constexpr double kInvalidLowZero = 0.0;
constexpr double kInvalidNegativePrice = -1.0;
constexpr double kInvalidNegativeVolume = -10.0;

// Hardcoded here (rather than imported) because these tests verify
// annualization_factor() returns those exact numbers.
constexpr int kExpectedDailyFactor = 252;
constexpr int kExpectedMinutesPerTradingDay = 390;
constexpr int kExpectedSecondsPerTradingDay = 23400;
constexpr int kExpectedFiveMinPerTradingDay = 78;
constexpr int kExpectedFifteenMinPerTradingDay = 26;
constexpr int kExpectedHoursPerTradingDay = 7;
constexpr int kExpectedWeeklyFactor = 52;
constexpr int kExpectedSecondsPerYear = 5896800;
constexpr int kExpectedMinutesPerYear = 98280;
constexpr int kExpectedFiveMinPerYear = 19656;
constexpr int kExpectedFifteenMinPerYear = 6552;
constexpr int kExpectedHoursPerYear = 1764;

constexpr size_t kReserveCapacity = 1000;

constexpr int kSecondsPerDay = 86400;
constexpr int kSmallSeriesLen = 5;
constexpr int kTinySeriesLen = 3;
constexpr int kMediumSeriesLen = 10;
constexpr int kSliceStartIdx = 2;
constexpr int kSliceEndIdx = 5;
constexpr size_t kSliceExpectedLen = 4;
constexpr int64_t kFarFutureTimestamp = 999999;

TEST(BarTest, ValidBar) {
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = kValidOpen, .high = kValidHigh,
            .low = kValidLow, .close = kValidClose, .volume = kSampleVolume};
    EXPECT_TRUE(bar.is_valid());
}

TEST(BarTest, InvalidHighLessThanLow) {
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = kValidOpen,
            .high = kInvalidHighBelowLow, .low = kValidLow,
            .close = kInvalidLowBelowClose, .volume = kSampleVolume};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, InvalidHighLessThanClose) {
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = kValidOpen,
            .high = kInvalidHighBelowClose, .low = kValidLow,
            .close = kValidClose, .volume = kSampleVolume};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, InvalidLowGreaterThanOpen) {
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = kValidOpen, .high = kValidHigh,
            .low = kInvalidLowAboveOpen, .close = 103.0, .volume = kSampleVolume};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, InvalidNegativeOpen) {
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = kInvalidNegativePrice,
            .high = kValidHigh, .low = kInvalidNegativePrice,
            .close = kValidClose, .volume = kSampleVolume};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, InvalidNegativeVolume) {
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = kValidOpen, .high = kValidHigh,
            .low = kValidLow, .close = kValidClose, .volume = kInvalidNegativeVolume};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, DegenerateBarAllEqual) {
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = kValidOpen, .high = kValidOpen,
            .low = kValidOpen, .close = kValidOpen, .volume = 0.0};
    EXPECT_TRUE(bar.is_valid());
}

TEST(BarTest, ZeroLowIsInvalid) {
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = kValidOpen, .high = kValidHigh,
            .low = kInvalidLowZero, .close = kValidClose, .volume = kSampleVolume};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, ZeroOpenIsInvalid) {
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = 0.0, .high = kValidHigh,
            .low = kInvalidLowZero, .close = kValidClose, .volume = kSampleVolume};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, InfinityIsInvalid) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = inf, .high = inf,
            .low = kValidLow, .close = kValidClose, .volume = kSampleVolume};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, NegativeInfinityIsInvalid) {
    constexpr double neg_inf = -std::numeric_limits<double>::infinity();
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = kValidOpen, .high = kValidHigh,
            .low = neg_inf, .close = kValidClose, .volume = kSampleVolume};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, NaNIsInvalid) {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    Bar bar{.timestamp_epoch_s = kSampleTimestampS, .open = kValidOpen, .high = kValidHigh,
            .low = kValidLow, .close = nan, .volume = kSampleVolume};
    EXPECT_FALSE(bar.is_valid());
}

TEST(IntervalTest, DailyFactor) {
    EXPECT_EQ(annualization_factor(Interval::Daily), kExpectedDailyFactor);
}

TEST(IntervalTest, MinuteFactor) {
    EXPECT_EQ(annualization_factor(Interval::Minute),
              kExpectedDailyFactor * kExpectedMinutesPerTradingDay);
}

TEST(IntervalTest, SecondFactor) {
    EXPECT_EQ(annualization_factor(Interval::Second),
              kExpectedDailyFactor * kExpectedSecondsPerTradingDay);
}

TEST(IntervalTest, FiveMinuteFactor) {
    EXPECT_EQ(annualization_factor(Interval::FiveMinute),
              kExpectedDailyFactor * kExpectedFiveMinPerTradingDay);
}

TEST(IntervalTest, FifteenMinuteFactor) {
    EXPECT_EQ(annualization_factor(Interval::FifteenMinute),
              kExpectedDailyFactor * kExpectedFifteenMinPerTradingDay);
}

TEST(IntervalTest, HourFactor) {
    EXPECT_EQ(annualization_factor(Interval::Hour),
              kExpectedDailyFactor * kExpectedHoursPerTradingDay);
}

TEST(IntervalTest, WeeklyFactor) {
    EXPECT_EQ(annualization_factor(Interval::Weekly), kExpectedWeeklyFactor);
}

TEST(IntervalTest, FactorsMatchPythonConstants) {
    EXPECT_EQ(annualization_factor(Interval::Second), kExpectedSecondsPerYear);
    EXPECT_EQ(annualization_factor(Interval::Minute), kExpectedMinutesPerYear);
    EXPECT_EQ(annualization_factor(Interval::FiveMinute), kExpectedFiveMinPerYear);
    EXPECT_EQ(annualization_factor(Interval::FifteenMinute), kExpectedFifteenMinPerYear);
    EXPECT_EQ(annualization_factor(Interval::Hour), kExpectedHoursPerYear);
    EXPECT_EQ(annualization_factor(Interval::Daily), kExpectedDailyFactor);
    EXPECT_EQ(annualization_factor(Interval::Weekly), kExpectedWeeklyFactor);
}

TEST(BarSoATest, SizeAndEmpty) {
    BarSoA soa;
    EXPECT_EQ(soa.size(), 0u);
    EXPECT_TRUE(soa.empty());

    soa.timestamps.push_back(kSampleTimestampS);
    soa.open.push_back(kValidOpen);
    soa.high.push_back(kValidHigh);
    soa.low.push_back(kValidLow);
    soa.close.push_back(kValidClose);
    soa.volume.push_back(kSampleVolume);

    EXPECT_EQ(soa.size(), 1u);
    EXPECT_FALSE(soa.empty());
}

TEST(BarSoATest, Reserve) {
    BarSoA soa;
    soa.reserve(kReserveCapacity);
    EXPECT_GE(soa.timestamps.capacity(), kReserveCapacity);
    EXPECT_GE(soa.close.capacity(), kReserveCapacity);
}

static std::vector<Bar> make_sorted_bars(int n, int64_t start_ts = kSampleTimestampS) {
    std::vector<Bar> bars;
    bars.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        bars.push_back(Bar{
            .timestamp_epoch_s = start_ts + i * kSecondsPerDay,
            .open = kValidOpen + i,
            .high = kValidHigh + i,
            .low = kValidLow + i,
            .close = kValidClose + i,
            .volume = kSampleVolume
        });
    }
    return bars;
}

TEST(TimeSeriesTest, ValidConstruction) {
    auto bars = make_sorted_bars(kSmallSeriesLen);
    TimeSeries<Bar> ts(bars);
    EXPECT_EQ(ts.size(), static_cast<size_t>(kSmallSeriesLen));
    EXPECT_FALSE(ts.empty());
}

TEST(TimeSeriesTest, ThrowsOnEmpty) {
    std::vector<Bar> empty_bars;
    EXPECT_THROW(auto ts = TimeSeries<Bar>(empty_bars), std::invalid_argument);
}

TEST(TimeSeriesTest, ThrowsOnUnsorted) {
    auto bars = make_sorted_bars(kSmallSeriesLen);
    std::swap(bars[1], bars[3]);
    EXPECT_THROW(auto ts = TimeSeries<Bar>(bars), std::invalid_argument);
}

TEST(TimeSeriesTest, ThrowsOnDuplicateTimestamps) {
    auto bars = make_sorted_bars(kSmallSeriesLen);
    bars[2].timestamp_epoch_s = bars[1].timestamp_epoch_s;
    EXPECT_THROW(auto ts = TimeSeries<Bar>(bars), std::invalid_argument);
}

TEST(TimeSeriesTest, ViewReturnsCorrectSpan) {
    auto bars = make_sorted_bars(kSmallSeriesLen);
    TimeSeries<Bar> ts(bars);
    auto view = ts.view();
    EXPECT_EQ(view.size(), static_cast<size_t>(kSmallSeriesLen));
    EXPECT_DOUBLE_EQ(view[0].open, kValidOpen);
    EXPECT_DOUBLE_EQ(view[kSmallSeriesLen - 1].open, kValidOpen + (kSmallSeriesLen - 1));
}

TEST(TimeSeriesTest, IndexOperator) {
    auto bars = make_sorted_bars(kTinySeriesLen);
    TimeSeries<Bar> ts(bars);
    EXPECT_DOUBLE_EQ(ts[0].close, kValidClose);
    EXPECT_DOUBLE_EQ(ts[kTinySeriesLen - 1].close, kValidClose + (kTinySeriesLen - 1));
}

TEST(TimeSeriesTest, SliceSubset) {
    auto bars = make_sorted_bars(kMediumSeriesLen);
    TimeSeries<Bar> ts(bars);

    int64_t start = kSampleTimestampS + kSliceStartIdx * kSecondsPerDay;
    int64_t end = kSampleTimestampS + kSliceEndIdx * kSecondsPerDay;
    auto sliced = ts.slice(start, end);
    EXPECT_EQ(sliced.size(), kSliceExpectedLen);
    EXPECT_EQ(sliced[0].timestamp_epoch_s, start);
}

TEST(TimeSeriesTest, SliceEmptyThrows) {
    auto bars = make_sorted_bars(kSmallSeriesLen);
    TimeSeries<Bar> ts(bars);
    EXPECT_THROW(auto sliced = ts.slice(kFarFutureTimestamp, kFarFutureTimestamp),
                 std::invalid_argument);
}

TEST(TaggedSeriesTest, TrainTagCompiles) {
    auto bars = make_sorted_bars(kSmallSeriesLen);
    auto ts = TimeSeries<Bar>(bars);
    TaggedSeries<TrainTag> train{std::move(ts)};
    EXPECT_EQ(train.size(), static_cast<size_t>(kSmallSeriesLen));
    EXPECT_DOUBLE_EQ(train[0].open, kValidOpen);
}

TEST(TaggedSeriesTest, TestTagCompiles) {
    auto bars = make_sorted_bars(kTinySeriesLen);
    auto ts = TimeSeries<Bar>(bars);
    TaggedSeries<TestTag> test_series{std::move(ts)};
    EXPECT_EQ(test_series.size(), static_cast<size_t>(kTinySeriesLen));
}

TEST(TaggedSeriesTest, ViewWorks) {
    auto bars = make_sorted_bars(kSmallSeriesLen);
    auto ts = TimeSeries<Bar>(bars);
    TaggedSeries<TrainTag> train{std::move(ts)};
    auto view = train.view();
    EXPECT_EQ(view.size(), static_cast<size_t>(kSmallSeriesLen));
}

}  // namespace
}  // namespace quant
