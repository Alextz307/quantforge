#include <cstddef>
#include <vector>

#include <benchmark/benchmark.h>

#include "detail/measure.hpp"
#include "detail/random.hpp"
#include "quant/strategies/state_machines.hpp"

namespace {

constexpr double kPriceStart = 100.0;
constexpr double kPriceStdDev = 1.0;
constexpr double kBandHalfWidth = 2.0;
constexpr double kTrendScale = 0.99;  // trend_ma slightly below close to exercise entries
constexpr double kZScoreStdDev = 1.5;
constexpr double kEntryZ = 2.0;
constexpr double kExitZ = 0.5;
constexpr double kStopLossZ = 3.0;

void BM_MeanReversionStateMachine(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto close = quant::benchmark::detail::additive_random_walk(n, kPriceStart, kPriceStdDev);
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
    quant::benchmark::detail::measure(state, [&] {
        benchmark::DoNotOptimize(
            quant::strategies::run_mean_reversion_state_machine(
                close, mid, upper, lower, trend_ma));
    });
}
BENCHMARK(BM_MeanReversionStateMachine)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

void BM_PairsStateMachine(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    auto z = quant::benchmark::detail::filled_normal(n, 0.0, kZScoreStdDev);
    quant::benchmark::detail::measure(state, [&] {
        benchmark::DoNotOptimize(
            quant::strategies::run_pairs_state_machine(
                z, kEntryZ, kExitZ, kStopLossZ));
    });
}
BENCHMARK(BM_PairsStateMachine)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
