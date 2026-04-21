#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace quant {

// ── Trading constants ──

inline constexpr int kTradingDaysPerYear = 252;
inline constexpr int kTradingWeeksPerYear = 52;
inline constexpr int kUSMinutesPerDay = 390;   // 6.5 hours
inline constexpr int kUSSecondsPerDay = 23400;  // 390 * 60

inline constexpr double kMaxLeverage = 3.0;
inline constexpr double kMinPosition = -1.0;
inline constexpr double kMaxPosition = 3.0;

// Default label a fresh BacktestResult carries when the caller runs a single
// scenario without naming it. The out-param overload resets to this value so
// reusing a BacktestResult across scenarios cannot leak a prior label.
inline constexpr const char* kDefaultScenarioLabel = "normal";

// ── Interval ──

enum class Interval : uint8_t {
    Second,
    Minute,
    FiveMinute,
    FifteenMinute,
    Hour,
    Daily,
    Weekly
};

/// Returns the number of bars per year for a given interval.
[[nodiscard]] constexpr int annualization_factor(Interval iv) noexcept {
    switch (iv) {
        case Interval::Second:         return kTradingDaysPerYear * kUSSecondsPerDay;
        case Interval::Minute:         return kTradingDaysPerYear * kUSMinutesPerDay;
        case Interval::FiveMinute:     return kTradingDaysPerYear * (kUSMinutesPerDay / 5);
        case Interval::FifteenMinute:  return kTradingDaysPerYear * (kUSMinutesPerDay / 15);
        // 6.5 trading hours = 7 discrete hourly bars (the last bar is partial).
        // Matches Python's _ANNUALIZATION_FACTORS["1h"] = 252 * 7.
        case Interval::Hour:           return kTradingDaysPerYear * 7;
        case Interval::Daily:          return kTradingDaysPerYear;
        case Interval::Weekly:         return kTradingWeeksPerYear;
    }
    return kTradingDaysPerYear;  // unreachable, satisfies -Wreturn-type
}

// ── Bar (AoS — for API/pybind11 interfaces) ──

struct Bar {
    int64_t timestamp_epoch_s{};
    double open{};
    double high{};
    double low{};
    double close{};
    double volume{};

    [[nodiscard]] bool is_valid() const noexcept {
        return std::isfinite(open) && std::isfinite(high)
            && std::isfinite(low) && std::isfinite(close)
            && std::isfinite(volume)
            && high >= low
            && high >= std::max(open, close)
            && low <= std::min(open, close)
            && open > 0.0
            && close > 0.0
            && low > 0.0
            && volume >= 0.0;
    }
};

// ── BarSoA (Structure of Arrays — for computation hot paths) ──

struct BarSoA {
    // Heap data alignment depends on the allocator (macOS default is 16-byte,
    // sufficient for NEON). Use a custom aligned allocator if cache-line
    // alignment of the data buffer is needed for SIMD (profile first).
    std::vector<int64_t> timestamps;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;

    [[nodiscard]] size_t size() const noexcept {
        return timestamps.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return timestamps.empty();
    }

    void reserve(size_t n) {
        timestamps.reserve(n);
        open.reserve(n);
        high.reserve(n);
        low.reserve(n);
        close.reserve(n);
        volume.reserve(n);
    }
};

// ── Signal ──

struct Signal {
    int64_t timestamp_epoch_s{};
    double position{};  // [kMinPosition, kMaxPosition]
};

// ── PairSignal ──

struct PairSignal {
    int64_t timestamp_epoch_s{};
    double leg_a_position{};
    double leg_b_position{};
    double spread_zscore{};
};

// ── BacktestResult ──

struct BacktestResult {
    double total_return{};
    double annualized_return{};
    double annualized_volatility{};
    double sharpe_ratio{};
    double sortino_ratio{};
    double max_drawdown{};
    double win_rate{};
    int trade_count{};
    std::vector<double> equity_curve;
    std::string scenario_label{kDefaultScenarioLabel};
};

}  // namespace quant
