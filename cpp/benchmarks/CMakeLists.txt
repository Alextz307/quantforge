# C++ benchmarks (Google Benchmark)

add_executable(quant_bench
    bench_indicators.cpp
    bench_backtest.cpp
    bench_metrics.cpp
    bench_garch_filter.cpp
    bench_state_machines.cpp
    bench_spread.cpp
    bench_pairs_trading_strategy.cpp
    bench_adaptive_bollinger_strategy.cpp
)
target_include_directories(quant_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quant_bench PRIVATE quant_core benchmark::benchmark_main)
