#pragma once

#include <cstdint>

// Cross-platform monotonic cycle-ish counter for benchmark instrumentation.
// Bench/test-only: never linked into quant_core. Cycles unit is backend-
// dependent (x86 TSC, arm64 CNTVCT virtual ticks, or steady_clock ns
// fallback) — comparable within a single machine/run, not across platforms.
// Retired-instruction counters require a PMU backend (macOS kpc, Linux
// perf_event_open) that is not wired yet; is_available() is a compile-time
// constant so the instructions/IPC branches in measure() dead-strip.

namespace quant::benchmark {

// Custom-counter key names. Exposed so Python parsers and LaTeX writers can
// reference the same strings the C++ bench emits without hand-copying them.
inline constexpr const char* kCyclesCounter = "Cycles";
inline constexpr const char* kCyclesPerItemCounter = "CyclesPerItem";
inline constexpr const char* kInstructionsCounter = "Instructions";
inline constexpr const char* kIPCCounter = "IPC";

class CycleCounter {
public:
    // Compile-time availability so callers can `if constexpr (...)` the
    // instructions path away entirely on platforms without a real PMU.
    [[nodiscard]] static constexpr bool is_available() noexcept { return false; }

    [[nodiscard]] static std::uint64_t read_cycles() noexcept;
    [[nodiscard]] static std::uint64_t read_instructions() noexcept;

    // RAII accumulator: ctor snapshots start counts, dtor adds the delta into
    // caller-owned totals. Thread-local — only valid on the constructing
    // thread. Non-copyable, non-movable.
    class Scope {
    public:
        Scope(std::uint64_t& cycles_total, std::uint64_t& instructions_total) noexcept
            : cycles_total_(cycles_total),
              instructions_total_(instructions_total),
              cycles_start_(CycleCounter::read_cycles()),
              instructions_start_(0) {
            if constexpr (CycleCounter::is_available()) {
                instructions_start_ = CycleCounter::read_instructions();
            }
        }

        ~Scope() noexcept {
            cycles_total_ += CycleCounter::read_cycles() - cycles_start_;
            if constexpr (CycleCounter::is_available()) {
                instructions_total_ +=
                    CycleCounter::read_instructions() - instructions_start_;
            }
        }

        Scope(const Scope&) = delete;
        Scope(Scope&&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope& operator=(Scope&&) = delete;

    private:
        std::uint64_t& cycles_total_;
        std::uint64_t& instructions_total_;
        std::uint64_t cycles_start_;
        std::uint64_t instructions_start_;
    };
};

}  // namespace quant::benchmark
