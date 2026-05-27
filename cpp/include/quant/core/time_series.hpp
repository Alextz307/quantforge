#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "quant/core/types.hpp"

namespace quant {

template<typename T>
concept HasTimestamp = requires(const T& t) {
    { t.timestamp_epoch_s } -> std::convertible_to<int64_t>;
};

template<HasTimestamp T>
class TimeSeries {
public:
    /// Construct from a vector of records. Validates strict temporal ordering.
    /// @throws std::invalid_argument if data is empty or not sorted by timestamp.
    explicit TimeSeries(std::vector<T> data)
        : data_(std::move(data))
    {
        if (data_.empty()) {
            throw std::invalid_argument("TimeSeries: data must not be empty");
        }
        for (size_t i = 1; i < data_.size(); ++i) {
            if (data_[i].timestamp_epoch_s <= data_[i - 1].timestamp_epoch_s) {
                throw std::invalid_argument(
                    "TimeSeries: timestamps must be strictly increasing at index "
                    + std::to_string(i)
                );
            }
        }
    }

    /// Non-owning view of the underlying data.
    [[nodiscard]] std::span<const T> view() const noexcept {
        return std::span<const T>(data_);
    }

    /// Create a new TimeSeries containing records in [start_ts, end_ts].
    /// Uses binary search (O(log n)) since data is sorted by timestamp.
    /// @throws std::invalid_argument if the resulting slice is empty.
    [[nodiscard]] TimeSeries<T> slice(int64_t start_ts, int64_t end_ts) const {
        const auto view_span = slice_view(start_ts, end_ts);
        std::vector<T> result(view_span.begin(), view_span.end());
        // Skip re-validation: data came from an already-validated, sorted parent.
        return TimeSeries<T>(std::move(result), SkipValidation{});
    }

    /// Non-owning span of records in [start_ts, end_ts]. Zero-copy alternative
    /// to ``slice()`` for walk-forward evaluation: the returned span is valid
    /// only as long as this ``TimeSeries`` instance outlives it.
    /// @throws std::invalid_argument if the resulting slice is empty.
    [[nodiscard]] std::span<const T> slice_view(
        int64_t start_ts, int64_t end_ts) const
    {
        auto begin_it = std::lower_bound(
            data_.begin(), data_.end(), start_ts,
            [](const T& item, int64_t ts) { return item.timestamp_epoch_s < ts; });
        auto end_it = std::upper_bound(
            begin_it, data_.end(), end_ts,
            [](int64_t ts, const T& item) { return ts < item.timestamp_epoch_s; });
        if (begin_it == end_it) {
            throw std::invalid_argument("TimeSeries: slice produced empty result");
        }
        const auto offset = static_cast<size_t>(begin_it - data_.begin());
        const auto count = static_cast<size_t>(end_it - begin_it);
        return std::span<const T>(data_.data() + offset, count);
    }

    [[nodiscard]] size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] const T& operator[](size_t index) const { return data_[index]; }

private:
    struct SkipValidation {};

    /// ``slice()`` has already confirmed non-empty + sorted via ``slice_view``.
    TimeSeries(std::vector<T> data, SkipValidation)
        : data_(std::move(data)) {}

    std::vector<T> data_;
};

struct TrainTag {};
struct TestTag {};

/// Type-safe wrapper that tags a TimeSeries<Bar> as train or test data.
/// Prevents accidental mixing at compile time.
template<typename Tag>
class TaggedSeries {
public:
    explicit TaggedSeries(TimeSeries<Bar> series)
        : series_(std::move(series)) {}

    [[nodiscard]] std::span<const Bar> view() const noexcept {
        return series_.view();
    }

    [[nodiscard]] size_t size() const noexcept { return series_.size(); }
    [[nodiscard]] bool empty() const noexcept { return series_.empty(); }

    [[nodiscard]] const Bar& operator[](size_t index) const {
        return series_[index];
    }

private:
    TimeSeries<Bar> series_;
};

}  // namespace quant
