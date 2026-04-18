#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "detail/random.hpp"
#include "quant/indicators/bollinger_bands.hpp"
#include "quant/indicators/garman_klass.hpp"
#include "quant/indicators/macd.hpp"
#include "quant/indicators/parkinson.hpp"
#include "quant/indicators/rsi.hpp"

namespace {

constexpr double kStartPrice = 100.0;
constexpr double kPriceFloor = 0.01;
constexpr double kReturnStdDev = 0.01;
constexpr double kSpreadMin = 0.001;
constexpr double kSpreadMax = 0.02;

std::vector<double> generate_prices(std::size_t n) {
    auto gen = quant::bench::detail::seeded_rng();
    std::normal_distribution<double> dist(0.0, kReturnStdDev);
    std::vector<double> prices(n);
    prices[0] = kStartPrice;
    for (std::size_t i = 1; i < n; ++i) {
        prices[i] = prices[i - 1] * (1.0 + dist(gen));
    }
    return prices;
}

struct OHLCData {
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
};

OHLCData generate_ohlc(std::size_t n) {
    auto gen = quant::bench::detail::seeded_rng();
    std::normal_distribution<double> ret_dist(0.0, kReturnStdDev);
    std::uniform_real_distribution<double> spread_dist(kSpreadMin, kSpreadMax);

    OHLCData data;
    data.open.resize(n);
    data.high.resize(n);
    data.low.resize(n);
    data.close.resize(n);

    double price = kStartPrice;
    for (std::size_t i = 0; i < n; ++i) {
        data.open[i] = price;
        double change = price * ret_dist(gen);
        data.close[i] = std::max(kPriceFloor, price + change);
        double spread = price * spread_dist(gen);
        data.high[i] = std::max(data.open[i], data.close[i]) + spread;
        data.low[i] = std::max(kPriceFloor, std::min(data.open[i], data.close[i]) - spread);
        price = data.close[i];
    }
    return data;
}

// ── RSI ──

void BM_RSI(benchmark::State& state) {
    auto prices = generate_prices(static_cast<std::size_t>(state.range(0)));
    quant::RSI rsi(14);
    for (auto _ : state) {
        benchmark::DoNotOptimize(rsi.compute(prices));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_RSI)->Arg(10000)->Arg(100000)->Arg(1000000);

// ── MACD ──

void BM_MACD(benchmark::State& state) {
    auto prices = generate_prices(static_cast<std::size_t>(state.range(0)));
    quant::MACD macd;
    for (auto _ : state) {
        benchmark::DoNotOptimize(macd.compute(prices));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_MACD)->Arg(10000)->Arg(100000)->Arg(1000000);

void BM_MACD_All(benchmark::State& state) {
    auto prices = generate_prices(static_cast<std::size_t>(state.range(0)));
    quant::MACD macd;
    for (auto _ : state) {
        benchmark::DoNotOptimize(macd.compute_all(prices));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_MACD_All)->Arg(10000)->Arg(100000)->Arg(1000000);

// ── Bollinger Bands ──

void BM_Bollinger(benchmark::State& state) {
    auto prices = generate_prices(static_cast<std::size_t>(state.range(0)));
    quant::BollingerBands bb(20, 2.0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(bb.compute_all(prices));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_Bollinger)->Arg(10000)->Arg(100000)->Arg(1000000);

// ── Garman-Klass ──

void BM_GarmanKlass(benchmark::State& state) {
    auto ohlc = generate_ohlc(static_cast<std::size_t>(state.range(0)));
    quant::GarmanKlass gk(22);
    for (auto _ : state) {
        benchmark::DoNotOptimize(gk.compute(ohlc.open, ohlc.high, ohlc.low, ohlc.close));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_GarmanKlass)->Arg(10000)->Arg(100000)->Arg(1000000);

// ── Parkinson ──

void BM_Parkinson(benchmark::State& state) {
    auto ohlc = generate_ohlc(static_cast<std::size_t>(state.range(0)));
    quant::Parkinson pk(22);
    for (auto _ : state) {
        benchmark::DoNotOptimize(pk.compute(ohlc.open, ohlc.high, ohlc.low, ohlc.close));
    }
    state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_Parkinson)->Arg(10000)->Arg(100000)->Arg(1000000);

}  // namespace
