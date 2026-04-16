#include <algorithm>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "quant/core/types.hpp"
#include "quant/engine/backtest_engine.hpp"
#include "quant/engine/slippage.hpp"

namespace {

// ───── Bench data constants (mirror cpp/tests/test_backtest_engine.cpp) ─────
constexpr int kSeed = 42;
constexpr int64_t kBaseTimestampS = 1'700'000'000;
constexpr int64_t kSecondsPerBar = 86'400;
constexpr double kStartPrice = 100.0;
constexpr double kPriceFloor = 0.01;
constexpr double kReturnStd = 0.01;
constexpr double kMinSpread = 0.001;
constexpr double kMaxSpread = 0.02;
constexpr double kSampleVolume = 1.0e6;

// ───── Slippage variants under bench ─────
constexpr double kVolumeScaledBaseBps = 1.0;
constexpr double kVolumeImpactCoeff = 100.0;

[[nodiscard]] std::vector<quant::Bar> generate_bars(size_t n) {
    std::mt19937 gen(kSeed);
    std::normal_distribution<double> ret_dist(0.0, kReturnStd);
    std::uniform_real_distribution<double> spread_dist(kMinSpread, kMaxSpread);

    std::vector<quant::Bar> bars(n);
    double price = kStartPrice;
    int64_t ts = kBaseTimestampS;
    for (size_t i = 0; i < n; ++i) {
        const double open = price;
        const double change = price * ret_dist(gen);
        const double close = std::max(kPriceFloor, price + change);
        const double spread = price * spread_dist(gen);
        bars[i] = quant::Bar{
            .timestamp_epoch_s = ts,
            .open = open,
            .high = std::max(open, close) + spread,
            .low = std::max(kPriceFloor, std::min(open, close) - spread),
            .close = close,
            .volume = kSampleVolume,
        };
        price = close;
        ts += kSecondsPerBar;
    }
    return bars;
}

// Alternating ±1 forces a fill on every bar — worst-case for the engine hot
// loop (maximum commission + slippage work per iteration).
[[nodiscard]] std::vector<double> generate_alternating_signals(size_t n) {
    std::vector<double> sig(n);
    for (size_t i = 0; i < n; ++i) {
        sig[i] = (i % 2 == 0) ? 1.0 : -1.0;
    }
    return sig;
}

// What these benches measure: each iteration includes the `equity_curve`
// allocation inside `engine.run()` (one fresh N-element vector per call).
// That mirrors what a single-shot caller pays, not what an HPO sweep pays —
// see the `TODO(Phase 6)` on `BacktestEngine::run()` for the buffer-reuse
// overload that would amortize that cost.

void BM_BacktestEngine_Run_NoSlippage(benchmark::State& state) {
    const auto bars = generate_bars(static_cast<size_t>(state.range(0)));
    const auto signals = generate_alternating_signals(bars.size());
    quant::BacktestEngine::Config cfg;
    cfg.slippage = quant::SlippageConfig{.model = quant::SlippageModel::NoSlippage};
    const quant::BacktestEngine engine{cfg};
    for (auto _ : state) {
        benchmark::DoNotOptimize(engine.run(bars, signals));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_BacktestEngine_Run_NoSlippage)
    ->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

void BM_BacktestEngine_Run(benchmark::State& state) {
    const auto bars = generate_bars(static_cast<size_t>(state.range(0)));
    const auto signals = generate_alternating_signals(bars.size());
    const quant::BacktestEngine engine{quant::BacktestEngine::Config{}};
    for (auto _ : state) {
        benchmark::DoNotOptimize(engine.run(bars, signals));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_BacktestEngine_Run)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

void BM_BacktestEngine_Run_VolumeScaled(benchmark::State& state) {
    const auto bars = generate_bars(static_cast<size_t>(state.range(0)));
    const auto signals = generate_alternating_signals(bars.size());
    quant::BacktestEngine::Config cfg;
    cfg.slippage = quant::SlippageConfig{
        .model = quant::SlippageModel::VolumeScaled,
        .base_bps = kVolumeScaledBaseBps,
        .volume_impact_coeff = kVolumeImpactCoeff,
    };
    const quant::BacktestEngine engine{cfg};
    for (auto _ : state) {
        benchmark::DoNotOptimize(engine.run(bars, signals));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_BacktestEngine_Run_VolumeScaled)
    ->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
