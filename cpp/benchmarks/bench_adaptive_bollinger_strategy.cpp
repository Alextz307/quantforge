#include <cstddef>

#include <benchmark/benchmark.h>

#include "detail/measure.hpp"
#include "detail/random.hpp"
#include "quant/strategies/adaptive_bollinger.hpp"

namespace {

constexpr double kPriceStart = 100.0;
constexpr double kPriceStdDev = 1.0;
constexpr double kCondVolMean = 1.5;
constexpr double kCondVolStdDev = 0.2;
constexpr int kBandWindow = 20;
constexpr double kBandK = 2.0;
constexpr int kTrendWindow = 100;

void BM_AdaptiveBollingerStrategy(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto close = quant::benchmark::detail::additive_random_walk(n, kPriceStart, kPriceStdDev);
    const auto cond_vol = quant::benchmark::detail::filled_normal(
        n, kCondVolMean, kCondVolStdDev, /*seed=*/7);
    const quant::strategies::AdaptiveBollingerStrategy strategy(
        quant::strategies::AdaptiveBollingerStrategy::Config{kBandWindow, kBandK, kTrendWindow});
    quant::benchmark::detail::measure(state, [&] {
        benchmark::DoNotOptimize(strategy.generate_signals(close, cond_vol));
    });
}
BENCHMARK(BM_AdaptiveBollingerStrategy)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
