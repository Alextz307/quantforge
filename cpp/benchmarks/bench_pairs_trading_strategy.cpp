#include <cstddef>

#include <benchmark/benchmark.h>

#include "detail/random.hpp"
#include "quant/statistics/spread.hpp"
#include "quant/strategies/pairs_trading.hpp"

namespace {

constexpr double kPriceStartA = 100.0;
constexpr double kPriceStartB = 90.0;
constexpr double kPriceStdDev = 1.0;
constexpr double kHedgeRatio = 1.1;
constexpr double kSpreadMean = 0.0;
constexpr double kSpreadStd = 1.0;
constexpr double kEntryZ = 2.0;
constexpr double kExitZ = 0.5;
constexpr double kStopLossZ = 4.0;
constexpr int kZScoreLookback = 60;

void BM_PairsTradingStrategy(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = quant::bench::detail::additive_random_walk(n, kPriceStartA, kPriceStdDev);
    const auto b = quant::bench::detail::additive_random_walk(n, kPriceStartB, kPriceStdDev,
                                                              /*seed=*/321);
    const quant::statistics::CointegrationParams coint{kHedgeRatio, kSpreadMean, kSpreadStd};
    const quant::strategies::PairsTradingStrategy strategy(
        quant::strategies::PairsTradingStrategy::Config{
            kEntryZ, kExitZ, kStopLossZ, kZScoreLookback});
    for (auto _ : state) {
        benchmark::DoNotOptimize(strategy.generate_signals(a, b, coint));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_PairsTradingStrategy)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
