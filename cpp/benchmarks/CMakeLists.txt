# C++ benchmarks (Google Benchmark)

add_executable(quant_bench
    bench_indicators.cpp
    bench_backtest.cpp
    bench_metrics.cpp
    bench_garch_filter.cpp
    bench_state_machines.cpp
)
target_link_libraries(quant_bench PRIVATE quant_core benchmark::benchmark_main)
