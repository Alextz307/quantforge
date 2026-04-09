#include <limits>

#include <gtest/gtest.h>

#include "quant/core/time_series.hpp"
#include "quant/core/types.hpp"

namespace quant {
namespace {

// ═══════════════════════════════════════════════════════════════
// Bar::is_valid()
// ═══════════════════════════════════════════════════════════════

TEST(BarTest, ValidBar) {
    Bar bar{.timestamp_epoch_s = 1000, .open = 100.0, .high = 105.0,
            .low = 95.0, .close = 102.0, .volume = 1000.0};
    EXPECT_TRUE(bar.is_valid());
}

TEST(BarTest, InvalidHighLessThanLow) {
    Bar bar{.timestamp_epoch_s = 1000, .open = 100.0, .high = 90.0,
            .low = 95.0, .close = 92.0, .volume = 1000.0};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, InvalidHighLessThanClose) {
    Bar bar{.timestamp_epoch_s = 1000, .open = 100.0, .high = 101.0,
            .low = 95.0, .close = 102.0, .volume = 1000.0};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, InvalidLowGreaterThanOpen) {
    Bar bar{.timestamp_epoch_s = 1000, .open = 100.0, .high = 105.0,
            .low = 101.0, .close = 103.0, .volume = 1000.0};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, InvalidNegativeOpen) {
    Bar bar{.timestamp_epoch_s = 1000, .open = -1.0, .high = 105.0,
            .low = -1.0, .close = 102.0, .volume = 1000.0};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, InvalidNegativeVolume) {
    Bar bar{.timestamp_epoch_s = 1000, .open = 100.0, .high = 105.0,
            .low = 95.0, .close = 102.0, .volume = -10.0};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, DegenerateBarAllEqual) {
    Bar bar{.timestamp_epoch_s = 1000, .open = 100.0, .high = 100.0,
            .low = 100.0, .close = 100.0, .volume = 0.0};
    EXPECT_TRUE(bar.is_valid());
}

TEST(BarTest, ZeroLowIsInvalid) {
    Bar bar{.timestamp_epoch_s = 1000, .open = 100.0, .high = 105.0,
            .low = 0.0, .close = 102.0, .volume = 1000.0};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, ZeroOpenIsInvalid) {
    Bar bar{.timestamp_epoch_s = 1000, .open = 0.0, .high = 105.0,
            .low = 0.0, .close = 102.0, .volume = 1000.0};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, InfinityIsInvalid) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    Bar bar{.timestamp_epoch_s = 1000, .open = inf, .high = inf,
            .low = 95.0, .close = 102.0, .volume = 1000.0};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, NegativeInfinityIsInvalid) {
    constexpr double neg_inf = -std::numeric_limits<double>::infinity();
    Bar bar{.timestamp_epoch_s = 1000, .open = 100.0, .high = 105.0,
            .low = neg_inf, .close = 102.0, .volume = 1000.0};
    EXPECT_FALSE(bar.is_valid());
}

TEST(BarTest, NaNIsInvalid) {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    Bar bar{.timestamp_epoch_s = 1000, .open = 100.0, .high = 105.0,
            .low = 95.0, .close = nan, .volume = 1000.0};
    EXPECT_FALSE(bar.is_valid());
}

// ═══════════════════════════════════════════════════════════════
// annualization_factor()
// ═══════════════════════════════════════════════════════════════

TEST(IntervalTest, DailyFactor) {
    EXPECT_EQ(annualization_factor(Interval::Daily), 252);
}

TEST(IntervalTest, MinuteFactor) {
    EXPECT_EQ(annualization_factor(Interval::Minute), 252 * 390);
}

TEST(IntervalTest, SecondFactor) {
    EXPECT_EQ(annualization_factor(Interval::Second), 252 * 23400);
}

TEST(IntervalTest, FiveMinuteFactor) {
    EXPECT_EQ(annualization_factor(Interval::FiveMinute), 252 * 78);
}

TEST(IntervalTest, FifteenMinuteFactor) {
    EXPECT_EQ(annualization_factor(Interval::FifteenMinute), 252 * 26);
}

TEST(IntervalTest, HourFactor) {
    EXPECT_EQ(annualization_factor(Interval::Hour), 252 * 7);
}

TEST(IntervalTest, WeeklyFactor) {
    EXPECT_EQ(annualization_factor(Interval::Weekly), 52);
}

TEST(IntervalTest, FactorsMatchPythonConstants) {
    // Cross-validate against Python _ANNUALIZATION_FACTORS
    EXPECT_EQ(annualization_factor(Interval::Second), 5896800);
    EXPECT_EQ(annualization_factor(Interval::Minute), 98280);
    EXPECT_EQ(annualization_factor(Interval::FiveMinute), 19656);
    EXPECT_EQ(annualization_factor(Interval::FifteenMinute), 6552);
    EXPECT_EQ(annualization_factor(Interval::Hour), 1764);
    EXPECT_EQ(annualization_factor(Interval::Daily), 252);
    EXPECT_EQ(annualization_factor(Interval::Weekly), 52);
}

// ═══════════════════════════════════════════════════════════════
// BarSoA
// ═══════════════════════════════════════════════════════════════

TEST(BarSoATest, SizeAndEmpty) {
    BarSoA soa;
    EXPECT_EQ(soa.size(), 0u);
    EXPECT_TRUE(soa.empty());

    soa.timestamps.push_back(1000);
    soa.open.push_back(100.0);
    soa.high.push_back(105.0);
    soa.low.push_back(95.0);
    soa.close.push_back(102.0);
    soa.volume.push_back(1000.0);

    EXPECT_EQ(soa.size(), 1u);
    EXPECT_FALSE(soa.empty());
}

TEST(BarSoATest, Reserve) {
    BarSoA soa;
    soa.reserve(1000);
    EXPECT_GE(soa.timestamps.capacity(), 1000u);
    EXPECT_GE(soa.close.capacity(), 1000u);
}

// ═══════════════════════════════════════════════════════════════
// TimeSeries<Bar>
// ═══════════════════════════════════════════════════════════════

static std::vector<Bar> make_sorted_bars(int n, int64_t start_ts = 1000) {
    std::vector<Bar> bars;
    bars.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        bars.push_back(Bar{
            .timestamp_epoch_s = start_ts + i * 86400,
            .open = 100.0 + i,
            .high = 105.0 + i,
            .low = 95.0 + i,
            .close = 102.0 + i,
            .volume = 1000.0
        });
    }
    return bars;
}

TEST(TimeSeriesTest, ValidConstruction) {
    auto bars = make_sorted_bars(5);
    TimeSeries<Bar> ts(bars);
    EXPECT_EQ(ts.size(), 5u);
    EXPECT_FALSE(ts.empty());
}

TEST(TimeSeriesTest, ThrowsOnEmpty) {
    std::vector<Bar> empty_bars;
    EXPECT_THROW(auto ts = TimeSeries<Bar>(empty_bars), std::invalid_argument);
}

TEST(TimeSeriesTest, ThrowsOnUnsorted) {
    auto bars = make_sorted_bars(5);
    std::swap(bars[1], bars[3]);  // break ordering
    EXPECT_THROW(auto ts = TimeSeries<Bar>(bars), std::invalid_argument);
}

TEST(TimeSeriesTest, ThrowsOnDuplicateTimestamps) {
    auto bars = make_sorted_bars(5);
    bars[2].timestamp_epoch_s = bars[1].timestamp_epoch_s;  // duplicate
    EXPECT_THROW(auto ts = TimeSeries<Bar>(bars), std::invalid_argument);
}

TEST(TimeSeriesTest, ViewReturnsCorrectSpan) {
    auto bars = make_sorted_bars(5);
    TimeSeries<Bar> ts(bars);
    auto view = ts.view();
    EXPECT_EQ(view.size(), 5u);
    EXPECT_DOUBLE_EQ(view[0].open, 100.0);
    EXPECT_DOUBLE_EQ(view[4].open, 104.0);
}

TEST(TimeSeriesTest, IndexOperator) {
    auto bars = make_sorted_bars(3);
    TimeSeries<Bar> ts(bars);
    EXPECT_DOUBLE_EQ(ts[0].close, 102.0);
    EXPECT_DOUBLE_EQ(ts[2].close, 104.0);
}

TEST(TimeSeriesTest, SliceSubset) {
    auto bars = make_sorted_bars(10);
    TimeSeries<Bar> ts(bars);

    // Slice bars 2-5 (timestamps 1000+2*86400 to 1000+5*86400)
    int64_t start = 1000 + 2 * 86400;
    int64_t end = 1000 + 5 * 86400;
    auto sliced = ts.slice(start, end);
    EXPECT_EQ(sliced.size(), 4u);  // indices 2, 3, 4, 5
    EXPECT_EQ(sliced[0].timestamp_epoch_s, start);
}

TEST(TimeSeriesTest, SliceEmptyThrows) {
    auto bars = make_sorted_bars(5);
    TimeSeries<Bar> ts(bars);
    // Range that doesn't include any bars
    EXPECT_THROW(auto sliced = ts.slice(999999, 999999), std::invalid_argument);
}

// ═══════════════════════════════════════════════════════════════
// TaggedSeries
// ═══════════════════════════════════════════════════════════════

TEST(TaggedSeriesTest, TrainTagCompiles) {
    auto bars = make_sorted_bars(5);
    auto ts = TimeSeries<Bar>(bars);
    TaggedSeries<TrainTag> train{std::move(ts)};
    EXPECT_EQ(train.size(), 5u);
    EXPECT_DOUBLE_EQ(train[0].open, 100.0);
}

TEST(TaggedSeriesTest, TestTagCompiles) {
    auto bars = make_sorted_bars(3);
    auto ts = TimeSeries<Bar>(bars);
    TaggedSeries<TestTag> test_series{std::move(ts)};
    EXPECT_EQ(test_series.size(), 3u);
}

TEST(TaggedSeriesTest, ViewWorks) {
    auto bars = make_sorted_bars(5);
    auto ts = TimeSeries<Bar>(bars);
    TaggedSeries<TrainTag> train{std::move(ts)};
    auto view = train.view();
    EXPECT_EQ(view.size(), 5u);
}

}  // namespace
}  // namespace quant
