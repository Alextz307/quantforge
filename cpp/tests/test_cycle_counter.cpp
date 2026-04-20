#include <chrono>
#include <cstdint>
#include <thread>

#include <gtest/gtest.h>

#include "quant/benchmark/cycle_counter.hpp"

namespace {

using quant::benchmark::CycleCounter;

constexpr auto kSleepDuration = std::chrono::milliseconds(2);
constexpr int kRepeatedScopeIterations = 3;
constexpr auto kPerIterationSleep = std::chrono::microseconds(100);

TEST(CycleCounter, ReadCyclesIsMonotonicOnThread) {
    const auto a = CycleCounter::read_cycles();
    const auto b = CycleCounter::read_cycles();
    EXPECT_LE(a, b);
}

TEST(CycleCounter, InstructionsAreZeroWhenPMUUnavailable) {
    if constexpr (CycleCounter::is_available()) {
        GTEST_SKIP() << "PMU backend is active — instructions counter is real.";
    } else {
        EXPECT_EQ(CycleCounter::read_instructions(), 0u);
    }
}

TEST(CycleCounter, ScopeAccumulatesPositiveCycleDeltaOverSleep) {
    std::uint64_t cycles = 0;
    std::uint64_t instructions = 0;
    {
        CycleCounter::Scope scope(cycles, instructions);
        std::this_thread::sleep_for(kSleepDuration);
    }
    EXPECT_GT(cycles, 0u);
    if constexpr (CycleCounter::is_available()) {
        EXPECT_GT(instructions, 0u);
    } else {
        EXPECT_EQ(instructions, 0u);
    }
}

TEST(CycleCounter, ScopeAccumulatesAcrossIterations) {
    // Guards against an overwriting (non-accumulating) Scope implementation.
    std::uint64_t cycles = 0;
    std::uint64_t instructions = 0;

    {
        CycleCounter::Scope scope(cycles, instructions);
        std::this_thread::sleep_for(kPerIterationSleep);
    }
    const std::uint64_t cycles_after_one = cycles;
    const std::uint64_t instructions_after_one = instructions;

    for (int i = 1; i < kRepeatedScopeIterations; ++i) {
        CycleCounter::Scope scope(cycles, instructions);
        std::this_thread::sleep_for(kPerIterationSleep);
    }

    EXPECT_GT(cycles, cycles_after_one);
    if constexpr (CycleCounter::is_available()) {
        EXPECT_GT(instructions, instructions_after_one);
    }
}

}  // namespace
