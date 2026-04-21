#include "quant/indicators/macd.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "quant/core/validation.hpp"

namespace quant {

// Signal EMA is seeded at the first valid MACD (index slow_period_-1) and
// the (adjust=False) recurrence advances from there; match this seed
// convention if you ever factor out an ewm helper.

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
    MACDResult result;
    result.macd_line.resize(prices.size());
    result.signal_line.resize(prices.size());
    result.histogram.resize(prices.size());
    compute_all(prices, result);
    return result;
}

void MACD::compute_all(
    std::span<const double> prices,
    MACDResult& out) const
{
    const auto n = static_cast<int>(prices.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();

    detail::check_out_size(prices.size(), out.macd_line.size(), "MACD::compute_all");
    detail::check_out_size(prices.size(), out.signal_line.size(), "MACD::compute_all");
    detail::check_out_size(prices.size(), out.histogram.size(), "MACD::compute_all");

    std::fill(out.macd_line.begin(), out.macd_line.end(), nan);
    std::fill(out.signal_line.begin(), out.signal_line.end(), nan);
    std::fill(out.histogram.begin(), out.histogram.end(), nan);

    if (n < slow_period_) {
        return;
    }

    const double alpha_fast = 2.0 / (fast_period_ + 1.0);
    const double alpha_slow = 2.0 / (slow_period_ + 1.0);
    const double alpha_sig = 2.0 / (signal_period_ + 1.0);
    const double one_minus_fast = 1.0 - alpha_fast;
    const double one_minus_slow = 1.0 - alpha_slow;
    const double one_minus_sig = 1.0 - alpha_sig;

    double fast = prices[0];
    double slow = prices[0];
    double sig = 0.0;
    int signal_valid = 0;

    for (int i = 1; i < n; ++i) {
        const double p = prices[i];
        fast = alpha_fast * p + one_minus_fast * fast;
        slow = alpha_slow * p + one_minus_slow * slow;

        if (i >= slow_period_ - 1) {
            const double macd = fast - slow;
            out.macd_line[i] = macd;

            if (signal_valid == 0) {
                sig = macd;
                signal_valid = 1;
            } else {
                sig = alpha_sig * macd + one_minus_sig * sig;
                ++signal_valid;
            }
            if (signal_valid >= signal_period_) {
                out.signal_line[i] = sig;
                out.histogram[i] = macd - sig;
            }
        }
    }
}

void MACD::compute(
    std::span<const double> prices,
    std::span<double> out) const
{
    detail::check_out_size(prices.size(), out.size(), "MACD::compute");

    const auto n = static_cast<int>(prices.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill(out.begin(), out.end(), nan);

    if (n < slow_period_) {
        return;
    }

    const double alpha_fast = 2.0 / (fast_period_ + 1.0);
    const double alpha_slow = 2.0 / (slow_period_ + 1.0);
    const double one_minus_fast = 1.0 - alpha_fast;
    const double one_minus_slow = 1.0 - alpha_slow;

    double fast = prices[0];
    double slow = prices[0];

    for (int i = 1; i < n; ++i) {
        const double p = prices[i];
        fast = alpha_fast * p + one_minus_fast * fast;
        slow = alpha_slow * p + one_minus_slow * slow;
        if (i >= slow_period_ - 1) {
            out[i] = fast - slow;
        }
    }
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
