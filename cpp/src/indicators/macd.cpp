#include "quant/indicators/macd.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace quant {

namespace {

/// Compute EMA (adjust=False) seeded with the first value.
/// alpha = 2 / (span + 1).
/// TODO(Phase 6): compute() and compute_all() each allocate 2-3 temp EMA vectors.
/// Fusing the fast/slow EMA into a single pass and writing the difference directly
/// into the result would cut allocations from 3 to 1 and passes from 3 to 1.
std::vector<double> ema(std::span<const double> data, int span) {
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

}  // namespace

MACD::MACD(int fast_period, int slow_period, int signal_period)
    : fast_period_(fast_period)
    , slow_period_(slow_period)
    , signal_period_(signal_period)
{
    if (fast_period < 1 || slow_period < 1 || signal_period < 1) {
        throw std::invalid_argument("MACD periods must be >= 1");
    }
    if (fast_period >= slow_period) {
        throw std::invalid_argument("MACD fast_period must be < slow_period");
    }
}

MACDResult MACD::compute_all(std::span<const double> prices) const {
    const auto n = static_cast<int>(prices.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();

    MACDResult result;
    result.macd_line.resize(prices.size(), nan);
    result.signal_line.resize(prices.size(), nan);
    result.histogram.resize(prices.size(), nan);

    if (n < slow_period_) {
        return result;
    }

    auto ema_fast = ema(prices, fast_period_);
    auto ema_slow = ema(prices, slow_period_);

    // MACD line valid after slow EMA convergence
    for (int i = slow_period_ - 1; i < n; ++i) {
        result.macd_line[i] = ema_fast[i] - ema_slow[i];
    }

    // Signal line = EMA of valid MACD values (zero-copy via span)
    const int valid_start = slow_period_ - 1;
    const int valid_count = n - valid_start;

    if (valid_count >= signal_period_) {
        auto valid_span = std::span(result.macd_line).subspan(valid_start);
        auto signal_ema = ema(valid_span, signal_period_);

        const int signal_start = valid_start + signal_period_ - 1;
        for (int i = signal_start; i < n; ++i) {
            int offset = i - valid_start;
            result.signal_line[i] = signal_ema[offset];
            result.histogram[i] = result.macd_line[i] - result.signal_line[i];
        }
    }

    return result;
}

std::vector<double> MACD::compute(std::span<const double> prices) const {
    const auto n = static_cast<int>(prices.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> result(prices.size(), nan);

    if (n < slow_period_) {
        return result;
    }

    // Dedicated path: only the two EMAs needed for the MACD line,
    // avoiding signal/histogram allocation that compute_all() does.
    auto ema_fast = ema(prices, fast_period_);
    auto ema_slow = ema(prices, slow_period_);

    for (int i = slow_period_ - 1; i < n; ++i) {
        result[i] = ema_fast[i] - ema_slow[i];
    }

    return result;
}

int MACD::warmup_period() const noexcept {
    return slow_period_ - 1;
}

std::string MACD::name() const {
    return "MACD(" + std::to_string(fast_period_) + ","
           + std::to_string(slow_period_) + ","
           + std::to_string(signal_period_) + ")";
}

}  // namespace quant
