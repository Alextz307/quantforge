// slice_view returns a non-owning span into the parent's storage. Pointer
// identity against the expected offset is the observable contract.

#include <gtest/gtest.h>

#include "quant/core/time_series.hpp"
#include "quant/core/types.hpp"

#include "detail/parity_helpers.hpp"

namespace {

quant::TimeSeries<quant::Bar> make_ts(std::size_t n) {
    return quant::TimeSeries<quant::Bar>{
        quant::tests::detail::make_synthetic_bars(n, 0xBAAAAAADu, 100.0, 0.001)};
}

}  // namespace

TEST(TimeSeriesSliceView, PointsAtUnderlyingStorage) {
    const auto ts = make_ts(100);
    const auto start_ts = ts[20].timestamp_epoch_s;
    const auto end_ts = ts[60].timestamp_epoch_s;
    const auto view = ts.slice_view(start_ts, end_ts);
    ASSERT_EQ(view.size(), 41u);  // 20..60 inclusive
    // Non-owning view: the first element of the span aliases ts[20].
    EXPECT_EQ(&view[0], &ts[20]);
    EXPECT_EQ(&view[view.size() - 1], &ts[60]);
}

TEST(TimeSeriesSliceView, EmptyRangeThrows) {
    const auto ts = make_ts(10);
    const auto after_end = ts[9].timestamp_epoch_s + 100;
    EXPECT_THROW(
        (void)ts.slice_view(after_end, after_end + 1000), std::invalid_argument);
}

TEST(TimeSeriesSliceView, MatchesSliceElementValues) {
    const auto ts = make_ts(50);
    const auto s_ts = ts[10].timestamp_epoch_s;
    const auto e_ts = ts[40].timestamp_epoch_s;
    const auto view = ts.slice_view(s_ts, e_ts);
    const auto owned = ts.slice(s_ts, e_ts);
    ASSERT_EQ(view.size(), owned.size());
    for (std::size_t i = 0; i < view.size(); ++i) {
        EXPECT_EQ(view[i].timestamp_epoch_s, owned[i].timestamp_epoch_s);
        EXPECT_EQ(view[i].close, owned[i].close);
    }
}

TEST(TimeSeriesSliceView, OffBoundaryQueryWidensToSampledIndices) {
    // Query timestamps that fall *between* samples — lower_bound (>=) and
    // upper_bound (>) must still land on indices 20..60 (same as the
    // exact-boundary case) to preserve closed-interval semantics.
    const auto ts = make_ts(100);
    const auto before_20 = ts[20].timestamp_epoch_s - 1;
    const auto after_60 = ts[60].timestamp_epoch_s + 1;
    const auto view = ts.slice_view(before_20, after_60);
    ASSERT_EQ(view.size(), 41u);
    EXPECT_EQ(&view[0], &ts[20]);
    EXPECT_EQ(&view[view.size() - 1], &ts[60]);
}
