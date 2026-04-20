#include "quant/benchmark/cycle_counter.hpp"

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <x86intrin.h>
#elif !defined(__aarch64__)
#  include <chrono>
#endif

namespace quant::benchmark {

std::uint64_t CycleCounter::read_cycles() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    return __rdtsc();
#elif defined(__aarch64__)
    std::uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

std::uint64_t CycleCounter::read_instructions() noexcept {
    return 0;
}

}  // namespace quant::benchmark
