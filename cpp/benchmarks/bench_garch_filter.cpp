#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "quant/filters/garch_filter.hpp"

namespace {

constexpr double kOmega = 0.05;
constexpr double kAlpha1 = 0.10;
constexpr double kBeta1 = 0.85;
constexpr double kBackcast = 1.0;
constexpr double kReturnStdDev = 1.0;
constexpr std::uint_fast32_t kBenchSeed = 42;

std::vector<double> generate_returns(size_t n) {
    std::mt19937 gen(kBenchSeed);
    std::normal_distribution<double> dist(0.0, kReturnStdDev);
    std::vector<double> r(n);
    for (auto& x : r) x = dist(gen);
    return r;
}

void BM_GarchFilter(benchmark::State& state) {
    auto r = generate_returns(static_cast<size_t>(state.range(0)));
    quant::filters::GarchParams params{kOmega, {kAlpha1}, {kBeta1}, 0.0, kBackcast};
    for (auto _ : state) {
        benchmark::DoNotOptimize(quant::filters::garch_filter(r, params));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_GarchFilter)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
