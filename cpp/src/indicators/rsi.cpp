#include "quant/indicators/rsi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace quant {

RSI::RSI(int period) : period_(period) {
    if (period < 1) {
        throw std::invalid_argument("RSI period must be >= 1, got "
                                    + std::to_string(period));
    }
}

std::vector<double> RSI::compute(std::span<const double> prices) const {
    const auto n = static_cast<int>(prices.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> result(prices.size(), nan);

    // Need at least period + 1 prices to compute first RSI value
    if (n <= period_) {
        return result;
    }

    // Seed with SMA of first period deltas
    double avg_gain = 0.0;
    double avg_loss = 0.0;

    for (int i = 1; i <= period_; ++i) {
        double delta = prices[i] - prices[i - 1];
        avg_gain += std::max(0.0, delta);
        avg_loss += std::max(0.0, -delta);
    }
    avg_gain /= period_;
    avg_loss /= period_;

    // First RSI value at index = period
    if (avg_loss == 0.0) {
        result[period_] = (avg_gain == 0.0) ? 50.0 : 100.0;
    } else {
        double rs = avg_gain / avg_loss;
        result[period_] = 100.0 - 100.0 / (1.0 + rs);
    }

    // Wilder's smoothing for subsequent values
    for (int i = period_ + 1; i < n; ++i) {
        double delta = prices[i] - prices[i - 1];
        double gain = std::max(0.0, delta);
        double loss = std::max(0.0, -delta);

        avg_gain = (avg_gain * (period_ - 1) + gain) / period_;
        avg_loss = (avg_loss * (period_ - 1) + loss) / period_;

        if (avg_loss == 0.0) {
            result[i] = (avg_gain == 0.0) ? 50.0 : 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            result[i] = 100.0 - 100.0 / (1.0 + rs);
        }
    }

    return result;
}

int RSI::warmup_period() const noexcept {
    return period_;
}

std::string RSI::name() const {
    return "RSI(" + std::to_string(period_) + ")";
}

}  // namespace quant
