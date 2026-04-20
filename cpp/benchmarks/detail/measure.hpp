#pragma once

#include <cstddef>
#include <cstdint>

#include <benchmark/benchmark.h>

#include "quant/benchmark/cycle_counter.hpp"

// Accumulates CPU cycles per bench iteration via CycleCounter::Scope and
// emits them as Google Benchmark custom counters. Retired-instruction and
// IPC counters are emitted only when a real PMU backend is wired.

namespace quant::benchmark::detail {

inline void report_cycles(
    ::benchmark::State& state,
    std::uint64_t cycles_total,
    std::uint64_t instructions_total,
    std::size_t items_per_iter) noexcept {
    const auto iters = state.iterations();
    state.SetItemsProcessed(
        iters * static_cast<std::int64_t>(items_per_iter));

    if (iters == 0) {
        return;
    }
    const double iters_d = static_cast<double>(iters);
    state.counters[kCyclesCounter] =
        static_cast<double>(cycles_total) / iters_d;
    if (items_per_iter > 1) {
        const double items_d = iters_d * static_cast<double>(items_per_iter);
        state.counters[kCyclesPerItemCounter] =
            static_cast<double>(cycles_total) / items_d;
    }
    if constexpr (CycleCounter::is_available()) {
        state.counters[kInstructionsCounter] =
            static_cast<double>(instructions_total) / iters_d;
        if (cycles_total > 0) {
            state.counters[kIPCCounter] =
                static_cast<double>(instructions_total)
                / static_cast<double>(cycles_total);
        }
    }
}

template <typename F>
void measure(
    ::benchmark::State& state,
    std::size_t items_per_iter,
    F&& body) {
    std::uint64_t cycles_total = 0;
    std::uint64_t instructions_total = 0;
    for (auto _ : state) {
        CycleCounter::Scope scope(cycles_total, instructions_total);
        body();
    }
    report_cycles(state, cycles_total, instructions_total, items_per_iter);
}

// Overload defaulting items-per-iter to state.range(0), which covers the
// common case where the bench sweep argument is the workload size.
template <typename F>
void measure(::benchmark::State& state, F&& body) {
    measure(
        state,
        static_cast<std::size_t>(state.range(0)),
        static_cast<F&&>(body));
}

}  // namespace quant::benchmark::detail
