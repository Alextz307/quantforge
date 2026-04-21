#include "quant/indicators/macd.hpp"

#include <limits>
#include <stdexcept>

namespace quant {

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

// Signal EMA is seeded at the first valid MACD (index slow_period_-1) and
// the (adjust=False) recurrence advances from there; match this seed
// convention if you ever factor out an ewm helper.
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
            result.macd_line[i] = macd;

            if (signal_valid == 0) {
                sig = macd;
                signal_valid = 1;
            } else {
                sig = alpha_sig * macd + one_minus_sig * sig;
                ++signal_valid;
            }
            if (signal_valid >= signal_period_) {
                result.signal_line[i] = sig;
                result.histogram[i] = macd - sig;
            }
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
            result[i] = fast - slow;
        }
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
