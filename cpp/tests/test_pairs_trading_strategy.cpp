#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quant/statistics/spread.hpp"
#include "quant/strategies/pairs_trading.hpp"
#include "quant/strategies/state_machines.hpp"

namespace quant::strategies {
namespace {

constexpr double kTol = 1e-12;

TEST(PairsTradingStrategy, MetadataMatchesConfig) {
    PairsTradingStrategy::Config cfg{2.0, 0.5, 4.0, 30};
    const PairsTradingStrategy strategy(cfg);
    EXPECT_EQ(strategy.name(), "PairsTrading");
    EXPECT_EQ(strategy.required_warmup(), 30);
    EXPECT_DOUBLE_EQ(strategy.config().entry_zscore, 2.0);
}

TEST(PairsTradingStrategy, RejectsBadConfig) {
    EXPECT_THROW(PairsTradingStrategy(PairsTradingStrategy::Config{-1.0, 0.5, 4.0, 10}),
                 std::invalid_argument);
    EXPECT_THROW(PairsTradingStrategy(PairsTradingStrategy::Config{2.0, 2.5, 4.0, 10}),
                 std::invalid_argument);
    EXPECT_THROW(PairsTradingStrategy(PairsTradingStrategy::Config{2.0, 0.5, 1.5, 10}),
                 std::invalid_argument);
    EXPECT_THROW(PairsTradingStrategy(PairsTradingStrategy::Config{2.0, 0.5, 4.0, 1}),
                 std::invalid_argument);
}

TEST(PairsTradingStrategy, GenerateSignalsMatchesManualComposition) {
    // Build prices that trace a known spread pattern, compare the strategy's
    // end-to-end output against a manual composition of SpreadCalculator +
    // run_pairs_state_machine.
    std::vector<double> prices_a(50);
    std::vector<double> prices_b(50);
    for (std::size_t i = 0; i < prices_a.size(); ++i) {
        const double t = static_cast<double>(i);
        prices_b[i] = 100.0 + t * 0.1;
        // Spread oscillates: sin * large amplitude to trigger entries/exits.
        prices_a[i] = prices_b[i] * 1.5 + 5.0 * std::sin(t / 3.0);
    }
    const statistics::CointegrationParams coint{1.5, 0.0, 3.0};
    const PairsTradingStrategy::Config cfg{1.0, 0.25, 3.0, 10};
    const PairsTradingStrategy strategy(cfg);

    const auto strategy_out = strategy.generate_signals(prices_a, prices_b, coint);

    const auto spread = statistics::SpreadCalculator::compute_spread(
        prices_a, prices_b, coint.hedge_ratio);
    const auto zscore = statistics::SpreadCalculator::compute_zscore(
        spread, cfg.zscore_lookback);
    const auto manual_out = run_pairs_state_machine(
        zscore, cfg.entry_zscore, cfg.exit_zscore, cfg.stop_loss_zscore);

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

TEST(PairsTradingStrategy, LengthMismatchThrows) {
    const PairsTradingStrategy strategy(PairsTradingStrategy::Config{2.0, 0.5, 4.0, 10});
    const std::vector<double> a(20, 100.0);
    const std::vector<double> b(19, 100.0);
    const statistics::CointegrationParams coint{1.0, 0.0, 1.0};
    EXPECT_THROW((void)strategy.generate_signals(a, b, coint), std::invalid_argument);
}

}  // namespace
}  // namespace quant::strategies
