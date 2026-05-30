#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quant/indicators/detail/rolling.hpp"
#include "quant/strategies/adaptive_bollinger.hpp"
#include "quant/strategies/state_machines.hpp"

namespace quant::strategies {
namespace {

constexpr double kTol = 1e-12;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

TEST(AdaptiveBollingerStrategy, MetadataMatchesConfig) {
    AdaptiveBollingerStrategy::Config cfg{20, 2.0, 100};
    const AdaptiveBollingerStrategy strategy(cfg);
    EXPECT_EQ(strategy.name(), "AdaptiveBollinger");
    EXPECT_EQ(strategy.required_warmup(), 100);
    EXPECT_EQ(strategy.config().band_window, 20);
}

TEST(AdaptiveBollingerStrategy, RejectsBadConfig) {
    EXPECT_THROW(AdaptiveBollingerStrategy(AdaptiveBollingerStrategy::Config{1, 2.0, 10}),
                 std::invalid_argument);
    EXPECT_THROW(AdaptiveBollingerStrategy(AdaptiveBollingerStrategy::Config{10, 0.0, 10}),
                 std::invalid_argument);
    EXPECT_THROW(AdaptiveBollingerStrategy(AdaptiveBollingerStrategy::Config{10, 2.0, 1}),
                 std::invalid_argument);
}

TEST(AdaptiveBollingerStrategy, GenerateSignalsMatchesManualComposition) {
    // Sinusoidal close with constant vol -> bands oscillate around the mid;
    // verify the strategy's end-to-end output equals the explicit
    // rolling_mean + bands + state_machine composition.
    const std::size_t n = 200;
    std::vector<double> close(n);
    std::vector<double> cond_vol(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i);
        close[i] = 100.0 + 5.0 * std::sin(t / 7.0) + 0.05 * t;
        cond_vol[i] = 1.5;
    }

    const AdaptiveBollingerStrategy::Config cfg{10, 2.0, 50};
    const AdaptiveBollingerStrategy strategy(cfg);
    const auto strategy_out = strategy.generate_signals(close, cond_vol);

    const auto mid = detail::rolling_mean(close, cfg.band_window);
    const auto trend_ma = detail::rolling_mean(close, cfg.trend_window);
    std::vector<double> upper(n, kNaN);
    std::vector<double> lower(n, kNaN);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isnan(mid[i]) && !std::isnan(cond_vol[i])) {
            upper[i] = mid[i] + cfg.k * cond_vol[i];
            lower[i] = mid[i] - cfg.k * cond_vol[i];
        }
    }
    const auto manual_out = run_mean_reversion_state_machine(
        close, mid, upper, lower, trend_ma);

    ASSERT_EQ(strategy_out.size(), manual_out.size());
    for (std::size_t i = 0; i < strategy_out.size(); ++i) {
        if (std::isnan(strategy_out[i]) || std::isnan(manual_out[i])) {
            EXPECT_EQ(std::isnan(strategy_out[i]), std::isnan(manual_out[i]))
                << "NaN mismatch at position " << i;
        } else {
            EXPECT_NEAR(strategy_out[i], manual_out[i], kTol)
                << "value mismatch at position " << i;
        }
    }
}

TEST(AdaptiveBollingerStrategy, NaNCondVolProducesFlatOutput) {
    // cond_vol all NaN -> upper/lower NaN -> state machine emits NaN from bar 0
    // (no entries possible). All output bars are NaN.
    std::vector<double> close(30, 100.0);
    std::vector<double> cond_vol(30, kNaN);
    const AdaptiveBollingerStrategy strategy(AdaptiveBollingerStrategy::Config{5, 2.0, 10});
    const auto out = strategy.generate_signals(close, cond_vol);
    ASSERT_EQ(out.size(), close.size());
    for (const double v : out) {
        EXPECT_TRUE(std::isnan(v));
    }
}

TEST(AdaptiveBollingerStrategy, LengthMismatchThrows) {
    const AdaptiveBollingerStrategy strategy(AdaptiveBollingerStrategy::Config{5, 2.0, 10});
    const std::vector<double> close(20, 100.0);
    const std::vector<double> cond_vol(19, 1.0);
    EXPECT_THROW((void)strategy.generate_signals(close, cond_vol), std::invalid_argument);
}

}  // namespace
}  // namespace quant::strategies
