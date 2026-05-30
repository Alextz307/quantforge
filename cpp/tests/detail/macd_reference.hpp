#pragma once

// Independent reference implementation of MACD - separate EMA vectors
// composed sequentially. Exists only to give the parity tests a fixed
// point to compare the production kernel against.

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "quant/indicators/macd.hpp"

namespace quant::tests::detail {

inline std::vector<double> ema_reference(std::span<const double> data, int span) {
    const auto n = static_cast<int>(data.size());
    std::vector<double> result(data.size());
    if (n == 0) return result;

    const double alpha = 2.0 / (span + 1.0);
    const double one_minus_alpha = 1.0 - alpha;

    result[0] = data[0];
    for (int i = 1; i < n; ++i) {
        result[i] = alpha * data[i] + one_minus_alpha * result[i - 1];
    }
    return result;
}

inline MACDResult macd_compute_all_reference(
    std::span<const double> prices,
    int fast_period,
    int slow_period,
    int signal_period)
{
    const auto n = static_cast<int>(prices.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();

    MACDResult result;
    result.macd_line.resize(prices.size(), nan);
    result.signal_line.resize(prices.size(), nan);
    result.histogram.resize(prices.size(), nan);
    if (n < slow_period) return result;

    auto ema_fast = ema_reference(prices, fast_period);
    auto ema_slow = ema_reference(prices, slow_period);

    for (int i = slow_period - 1; i < n; ++i) {
        result.macd_line[i] = ema_fast[i] - ema_slow[i];
    }

    const int valid_start = slow_period - 1;
    const int valid_count = n - valid_start;
    if (valid_count >= signal_period) {
        auto valid_span = std::span(result.macd_line).subspan(valid_start);
        auto signal_ema = ema_reference(valid_span, signal_period);

        const int signal_start = valid_start + signal_period - 1;
        for (int i = signal_start; i < n; ++i) {
            int offset = i - valid_start;
            result.signal_line[i] = signal_ema[offset];
            result.histogram[i] = result.macd_line[i] - result.signal_line[i];
        }
    }
    return result;
}

inline std::vector<double> macd_compute_reference(
    std::span<const double> prices,
    int fast_period,
    int slow_period)
{
    const auto n = static_cast<int>(prices.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> result(prices.size(), nan);
    if (n < slow_period) return result;

    auto ema_fast = ema_reference(prices, fast_period);
    auto ema_slow = ema_reference(prices, slow_period);
    for (int i = slow_period - 1; i < n; ++i) {
        result[i] = ema_fast[i] - ema_slow[i];
    }
    return result;
}

}  // namespace quant::tests::detail
