#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "quant/strategies/state_machines.hpp"

namespace {

constexpr std::uint_fast32_t kBenchSeed = 42;
constexpr double kPriceStart = 100.0;
constexpr double kPriceStdDev = 1.0;
constexpr double kBandHalfWidth = 2.0;
constexpr double kTrendScale = 0.99;  // trend_ma slightly below close to exercise entries
constexpr double kZScoreStdDev = 1.5;
constexpr double kEntryZ = 2.0;
constexpr double kExitZ = 0.5;
constexpr double kStopLossZ = 3.0;

std::vector<double> generate_prices(std::size_t n) {
    std::mt19937 gen(kBenchSeed);
    std::normal_distribution<double> dist(0.0, kPriceStdDev);
    std::vector<double> prices(n);
    double p = kPriceStart;
    for (auto& v : prices) {
        p += dist(gen);
        v = p;
    }
    return prices;
}

std::vector<double> generate_zscore(std::size_t n) {
    std::mt19937 gen(kBenchSeed);
    std::normal_distribution<double> dist(0.0, kZScoreStdDev);
    std::vector<double> z(n);
    for (auto& v : z) v = dist(gen);
    return z;
}

void BM_MeanReversionStateMachine(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto close = generate_prices(n);
    std::vector<double> mid(n);
    std::vector<double> upper(n);
    std::vector<double> lower(n);
    std::vector<double> trend_ma(n);
    for (std::size_t i = 0; i < n; ++i) {
        mid[i] = close[i];
        upper[i] = close[i] + kBandHalfWidth;
        lower[i] = close[i] - kBandHalfWidth;
        trend_ma[i] = close[i] * kTrendScale;
    }
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            quant::strategies::run_mean_reversion_state_machine(
                close, mid, upper, lower, trend_ma));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_MeanReversionStateMachine)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

void BM_PairsStateMachine(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto z = generate_zscore(n);
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            quant::strategies::run_pairs_state_machine(
                z, kEntryZ, kExitZ, kStopLossZ));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_PairsStateMachine)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
