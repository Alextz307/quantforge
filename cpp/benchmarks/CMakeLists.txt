# C++ benchmarks (Google Benchmark)

add_executable(quant_bench
    bench_indicators.cpp
)
target_link_libraries(quant_bench PRIVATE quant_core benchmark::benchmark_main)
