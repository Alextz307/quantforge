// Parity + validation tests for every buffer-reuse overload added across
// indicators, metrics, the backtest engine, spread, and strategies. Each
// family asserts that the out-param result is bit-identical to the
// allocating convenience and that a deliberately wrong-size out buffer
// raises ``std::invalid_argument``.

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quant/core/types.hpp"
#include "quant/engine/backtest_engine.hpp"
#include "quant/indicators/bollinger_bands.hpp"
#include "quant/indicators/garman_klass.hpp"
#include "quant/indicators/macd.hpp"
#include "quant/indicators/parkinson.hpp"
#include "quant/indicators/rsi.hpp"
#include "quant/metrics/performance.hpp"
#include "quant/statistics/spread.hpp"
#include "quant/strategies/adaptive_bollinger.hpp"
#include "quant/strategies/pairs_trading.hpp"

#include "detail/parity_helpers.hpp"

using quant::tests::detail::additive_random_walk;
using quant::tests::detail::expect_array_eq;
using quant::tests::detail::geometric_random_walk;

namespace {

constexpr std::size_t kN = 500;
constexpr unsigned kSeed = 0xC0FFEEu;

std::vector<double> make_prices() {
    return geometric_random_walk(kN, kSeed, 100.0, 0.01);
}

std::vector<double> make_equity_curve() {
    return geometric_random_walk(kN, kSeed + 1u, 10000.0, 0.005);
}

std::vector<quant::Bar> make_bars() {
    return quant::tests::detail::make_synthetic_bars(kN, kSeed + 2u, 100.0, 0.01);
}

std::vector<double> make_alternating_signals(std::size_t n) {
    std::vector<double> sig(n);
    for (std::size_t i = 0; i < n; ++i) {
        sig[i] = (i % 2 == 0) ? 1.0 : -1.0;
    }
    return sig;
}

}  // namespace

TEST(BufferReuseOverloads, RSIOutParamMatchesAllocating) {
    const auto prices = make_prices();
    quant::RSI rsi(14);
    const auto expected = rsi.compute(prices);
    std::vector<double> actual(prices.size());
    rsi.compute(prices, actual);
    expect_array_eq(expected, actual);
}

TEST(BufferReuseOverloads, RSIOutBufferTooSmallThrows) {
    const auto prices = make_prices();
    quant::RSI rsi(14);
    std::vector<double> bad(prices.size() - 1);
    EXPECT_THROW(rsi.compute(prices, bad), std::invalid_argument);
}

TEST(BufferReuseOverloads, MACDOutParamMatchesAllocating) {
    const auto prices = make_prices();
    quant::MACD macd(12, 26, 9);
    const auto expected = macd.compute(prices);
    std::vector<double> actual(prices.size());
    macd.compute(prices, actual);
    expect_array_eq(expected, actual);
}

TEST(BufferReuseOverloads, MACDComputeAllOutParamMatchesAllocating) {
    const auto prices = make_prices();
    quant::MACD macd(12, 26, 9);
    const auto expected = macd.compute_all(prices);
    quant::MACDResult actual;
    actual.macd_line.resize(prices.size());
    actual.signal_line.resize(prices.size());
    actual.histogram.resize(prices.size());
    macd.compute_all(prices, actual);
    expect_array_eq(expected.macd_line, actual.macd_line);
    expect_array_eq(expected.signal_line, actual.signal_line);
    expect_array_eq(expected.histogram, actual.histogram);
}

TEST(BufferReuseOverloads, MACDComputeAllWrongSizedVectorThrows) {
    const auto prices = make_prices();
    quant::MACD macd(12, 26, 9);
    quant::MACDResult bad;
    bad.macd_line.resize(prices.size());
    bad.signal_line.resize(prices.size() - 1);
    bad.histogram.resize(prices.size());
    EXPECT_THROW(macd.compute_all(prices, bad), std::invalid_argument);
}

TEST(BufferReuseOverloads, BollingerOutParamMatchesAllocating) {
    const auto prices = make_prices();
    quant::BollingerBands bb(20, 2.0);
    const auto expected = bb.compute(prices);
    std::vector<double> actual(prices.size());
    bb.compute(prices, actual);
    expect_array_eq(expected, actual);
}

TEST(BufferReuseOverloads, BollingerComputeAllOutParamMatchesAllocating) {
    const auto prices = make_prices();
    quant::BollingerBands bb(20, 2.0);
    const auto expected = bb.compute_all(prices);
    quant::BollingerResult actual;
    actual.upper.resize(prices.size());
    actual.mid.resize(prices.size());
    actual.lower.resize(prices.size());
    bb.compute_all(prices, actual);
    expect_array_eq(expected.upper, actual.upper);
    expect_array_eq(expected.mid, actual.mid);
    expect_array_eq(expected.lower, actual.lower);
}

TEST(BufferReuseOverloads, GarmanKlassOutParamMatchesAllocating) {
    auto closes = geometric_random_walk(kN, kSeed + 3u, 100.0, 0.01);
    std::vector<double> open(kN), high(kN), low(kN), close(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        close[i] = closes[i];
        open[i] = closes[i] * 0.999;
        high[i] = closes[i] * 1.01;
        low[i] = closes[i] * 0.99;
    }
    quant::GarmanKlass gk(22);
    const auto expected = gk.compute(open, high, low, close);
    std::vector<double> actual(kN);
    gk.compute(open, high, low, close, actual);
    expect_array_eq(expected, actual);

    std::vector<double> bad(kN - 1);
    EXPECT_THROW(gk.compute(open, high, low, close, bad), std::invalid_argument);
}

TEST(BufferReuseOverloads, ParkinsonOutParamMatchesAllocating) {
    auto closes = geometric_random_walk(kN, kSeed + 4u, 100.0, 0.01);
    std::vector<double> open(kN), high(kN), low(kN), close(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        close[i] = closes[i];
        open[i] = closes[i] * 0.999;
        high[i] = closes[i] * 1.01;
        low[i] = closes[i] * 0.99;
    }
    quant::Parkinson pk(22);
    const auto expected = pk.compute(open, high, low, close);
    std::vector<double> actual(kN);
    pk.compute(open, high, low, close, actual);
    expect_array_eq(expected, actual);

    std::vector<double> bad(kN - 1);
    EXPECT_THROW(pk.compute(open, high, low, close, bad), std::invalid_argument);
}

TEST(BufferReuseOverloads, MetricsEquityToReturnsBufferMatchesAllocating) {
    const auto eq = make_equity_curve();
    const auto expected = quant::MetricsCalculator::equity_to_returns(eq);
    quant::MetricsBuffer buf;
    const auto view = quant::MetricsCalculator::equity_to_returns(eq, buf);
    ASSERT_EQ(view.size(), expected.size());
    std::vector<double> actual(view.begin(), view.end());
    expect_array_eq(expected, actual);
}

TEST(BufferReuseOverloads, MetricsBufferResizesAcrossCalls) {
    const auto eq_small = make_equity_curve();
    std::vector<double> eq_big(eq_small);
    eq_big.resize(eq_small.size() + 50, eq_small.back());

    quant::MetricsBuffer buf;
    auto view_small = quant::MetricsCalculator::equity_to_returns(eq_small, buf);
    EXPECT_EQ(view_small.size(), eq_small.size() - 1);
    auto view_big = quant::MetricsCalculator::equity_to_returns(eq_big, buf);
    EXPECT_EQ(view_big.size(), eq_big.size() - 1);
    // Second call should have reused the same storage via resize (no span
    // invalidation is guaranteed across calls — callers snapshot view_small
    // before re-entering).
    EXPECT_EQ(buf.returns.size(), eq_big.size() - 1);
}

TEST(BufferReuseOverloads, BacktestEngineRunOutParamMatchesAllocating) {
    const auto bars = make_bars();
    const auto signals = make_alternating_signals(bars.size());
    quant::BacktestEngine::Config cfg;
    const quant::BacktestEngine engine{cfg};
    const auto expected = engine.run(bars, signals);
    quant::BacktestResult actual;
    engine.run(bars, signals, actual);
    EXPECT_EQ(expected.total_return, actual.total_return);
    EXPECT_EQ(expected.trade_count, actual.trade_count);
    expect_array_eq(expected.equity_curve, actual.equity_curve);
}

TEST(BufferReuseOverloads, BacktestResultReusedAcrossScenarios) {
    const auto bars = make_bars();
    const auto signals = make_alternating_signals(bars.size());
    quant::BacktestEngine::Config cfg;
    const quant::BacktestEngine engine{cfg};

    // Prime the reused struct with stale state so the reset path has
    // something to clobber.
    quant::BacktestResult reused;
    reused.scenario_label = "leftover";
    reused.total_return = -9999.0;
    reused.trade_count = 42;
    reused.equity_curve.assign(1000, 1.0);

    const auto fresh = engine.run(bars, signals);
    engine.run(bars, signals, reused);

    EXPECT_EQ(fresh.total_return, reused.total_return);
    EXPECT_EQ(fresh.trade_count, reused.trade_count);
    EXPECT_EQ(reused.scenario_label, std::string{quant::kDefaultScenarioLabel});
    expect_array_eq(fresh.equity_curve, reused.equity_curve);
}

TEST(BufferReuseOverloads, BacktestResultPreservesEquityCurveStorage) {
    // Assert pointer identity across calls of equal size — capacity ≥ N
    // alone would pass even if the impl fully reallocated the vector, which
    // would defeat the amortization contract of the out-param overload.
    const auto bars = make_bars();
    const auto signals = make_alternating_signals(bars.size());
    quant::BacktestEngine::Config cfg;
    const quant::BacktestEngine engine{cfg};

    quant::BacktestResult reused;
    engine.run(bars, signals, reused);
    ASSERT_GE(reused.equity_curve.capacity(), bars.size());
    const double* const first_data = reused.equity_curve.data();
    engine.run(bars, signals, reused);
    EXPECT_EQ(reused.equity_curve.data(), first_data);
}

TEST(BufferReuseOverloads, BacktestResultResetIsDriftSafe) {
    // The out-param overload must reset every scalar field a fresh
    // BacktestResult would default. Seed the struct with non-default
    // sentinels across every field and confirm each is clobbered.
    const auto bars = make_bars();
    quant::BacktestEngine::Config cfg;
    cfg.allow_short = false;
    const quant::BacktestEngine engine{cfg};

    std::vector<double> zero_signals(bars.size(), 0.0);
    quant::BacktestResult reused;
    reused.total_return = 42.0;
    reused.annualized_return = 42.0;
    reused.annualized_volatility = 42.0;
    reused.sharpe_ratio = 42.0;
    reused.sortino_ratio = 42.0;
    reused.max_drawdown = -42.0;
    reused.win_rate = 42.0;
    reused.trade_count = 42;
    reused.scenario_label = "leftover";
    engine.run(bars, zero_signals, reused);

    EXPECT_EQ(reused.total_return, 0.0);
    EXPECT_EQ(reused.annualized_return, 0.0);
    EXPECT_EQ(reused.annualized_volatility, 0.0);
    EXPECT_EQ(reused.sharpe_ratio, 0.0);
    EXPECT_EQ(reused.sortino_ratio, 0.0);
    EXPECT_EQ(reused.max_drawdown, 0.0);
    EXPECT_EQ(reused.win_rate, 0.0);
    EXPECT_EQ(reused.trade_count, 0);
    EXPECT_EQ(reused.scenario_label, std::string{quant::kDefaultScenarioLabel});
}

TEST(BufferReuseOverloads, PairsBufferReusesStorageAcrossCalls) {
    const auto a = geometric_random_walk(kN, kSeed + 11u, 100.0, 0.01);
    const auto b = geometric_random_walk(kN, kSeed + 12u, 120.0, 0.015);
    quant::statistics::CointegrationParams coint{0.8, 0.0, 1.0};
    quant::strategies::PairsTradingStrategy strat(
        quant::strategies::PairsTradingStrategy::Config{});

    std::vector<double> out(kN);
    quant::strategies::PairsTradingStrategy::Buffer scratch;
    strat.generate_signals(a, b, coint, out, scratch);
    ASSERT_GE(scratch.spread.capacity(), kN);
    ASSERT_GE(scratch.zscore.capacity(), kN);
    const double* const spread_data = scratch.spread.data();
    const double* const zscore_data = scratch.zscore.data();
    strat.generate_signals(a, b, coint, out, scratch);
    EXPECT_EQ(scratch.spread.data(), spread_data);
    EXPECT_EQ(scratch.zscore.data(), zscore_data);
}

TEST(BufferReuseOverloads, AdaptiveBollingerBufferReusesStorageAcrossCalls) {
    const auto close = geometric_random_walk(kN, kSeed + 13u, 100.0, 0.01);
    auto cond_vol = additive_random_walk(kN, kSeed + 14u, 1.0, 0.01);
    for (auto& v : cond_vol) {
        v = std::max(0.1, v);
    }
    quant::strategies::AdaptiveBollingerStrategy strat(
        quant::strategies::AdaptiveBollingerStrategy::Config{});

    std::vector<double> out(kN);
    quant::strategies::AdaptiveBollingerStrategy::Buffer scratch;
    strat.generate_signals(close, cond_vol, out, scratch);
    ASSERT_GE(scratch.mid.capacity(), kN);
    const double* const mid_data = scratch.mid.data();
    const double* const trend_data = scratch.trend_ma.data();
    const double* const upper_data = scratch.upper.data();
    const double* const lower_data = scratch.lower.data();
    strat.generate_signals(close, cond_vol, out, scratch);
    EXPECT_EQ(scratch.mid.data(), mid_data);
    EXPECT_EQ(scratch.trend_ma.data(), trend_data);
    EXPECT_EQ(scratch.upper.data(), upper_data);
    EXPECT_EQ(scratch.lower.data(), lower_data);
}

TEST(BufferReuseOverloads, SpreadCalculatorOutParamsMatchAllocating) {
    const auto a = geometric_random_walk(kN, kSeed + 5u, 100.0, 0.01);
    const auto b = geometric_random_walk(kN, kSeed + 6u, 120.0, 0.015);
    const double hedge_ratio = 0.8;
    const int window = 30;
    const auto spread_expected =
        quant::statistics::SpreadCalculator::compute_spread(a, b, hedge_ratio);
    std::vector<double> spread_actual(kN);
    quant::statistics::SpreadCalculator::compute_spread(
        a, b, hedge_ratio, spread_actual);
    expect_array_eq(spread_expected, spread_actual);

    const auto z_expected =
        quant::statistics::SpreadCalculator::compute_zscore(spread_expected, window);
    std::vector<double> z_actual(kN);
    quant::statistics::SpreadCalculator::compute_zscore(
        spread_expected, window, z_actual);
    expect_array_eq(z_expected, z_actual);

    std::vector<double> bad(kN - 1);
    EXPECT_THROW(
        quant::statistics::SpreadCalculator::compute_spread(a, b, hedge_ratio, bad),
        std::invalid_argument);
    EXPECT_THROW(
        quant::statistics::SpreadCalculator::compute_zscore(spread_expected, window, bad),
        std::invalid_argument);
}

TEST(BufferReuseOverloads, PairsTradingOutParamMatchesAllocating) {
    const auto a = geometric_random_walk(kN, kSeed + 7u, 100.0, 0.01);
    const auto b = geometric_random_walk(kN, kSeed + 8u, 120.0, 0.015);
    quant::statistics::CointegrationParams coint{0.8, 0.0, 1.0};
    quant::strategies::PairsTradingStrategy strat(
        quant::strategies::PairsTradingStrategy::Config{});
    const auto expected = strat.generate_signals(a, b, coint);
    std::vector<double> actual(kN);
    strat.generate_signals(a, b, coint, actual);
    expect_array_eq(expected, actual);

    std::vector<double> bad(kN - 1);
    EXPECT_THROW(strat.generate_signals(a, b, coint, bad), std::invalid_argument);
}

TEST(BufferReuseOverloads, AdaptiveBollingerOutParamMatchesAllocating) {
    const auto close = geometric_random_walk(kN, kSeed + 9u, 100.0, 0.01);
    auto cond_vol = additive_random_walk(kN, kSeed + 10u, 1.0, 0.01);
    for (auto& v : cond_vol) {
        v = std::max(0.1, v);
    }
    quant::strategies::AdaptiveBollingerStrategy strat(
        quant::strategies::AdaptiveBollingerStrategy::Config{});
    const auto expected = strat.generate_signals(close, cond_vol);
    std::vector<double> actual(kN);
    strat.generate_signals(close, cond_vol, actual);
    expect_array_eq(expected, actual);

    std::vector<double> bad(kN - 1);
    EXPECT_THROW(strat.generate_signals(close, cond_vol, bad), std::invalid_argument);
}
