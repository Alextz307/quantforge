#include <cstddef>

#include <benchmark/benchmark.h>

#include "detail/random.hpp"
#include "quant/statistics/spread.hpp"

namespace {

constexpr double kPriceStartA = 100.0;
constexpr double kPriceStartB = 90.0;
constexpr double kPriceStdDev = 1.0;
constexpr double kHedgeRatio = 1.1;
constexpr int kZScoreWindow = 60;

void BM_ComputeSpread(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto a = quant::bench::detail::additive_random_walk(n, kPriceStartA, kPriceStdDev);
    const auto b = quant::bench::detail::additive_random_walk(n, kPriceStartB, kPriceStdDev,
                                                              /*seed=*/123);
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            quant::statistics::SpreadCalculator::compute_spread(a, b, kHedgeRatio));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_ComputeSpread)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

void BM_ComputeZScore(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto spread = quant::bench::detail::additive_random_walk(n, 0.0, kPriceStdDev);
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            quant::statistics::SpreadCalculator::compute_zscore(spread, kZScoreWindow));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_ComputeZScore)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
