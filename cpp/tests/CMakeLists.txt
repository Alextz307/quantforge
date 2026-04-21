# C++ unit tests (Google Test)

add_executable(quant_tests
    test_core.cpp
    test_indicators.cpp
    test_backtest_engine.cpp
    test_metrics.cpp
    test_garch_filter.cpp
    test_state_machines.cpp
    test_spread.cpp
    test_pairs_trading_strategy.cpp
    test_adaptive_bollinger_strategy.cpp
    test_cycle_counter.cpp
    test_macd_fused_parity.cpp
    test_bollinger_rolling_mean_std_parity.cpp
    test_metrics_fused_parity.cpp
    test_spread_welford_zscore_parity.cpp
    test_buffer_reuse_overloads.cpp
    test_time_series_slice_view.cpp
)
target_include_directories(quant_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quant_tests
    PRIVATE quant_core quant_bench_instrumentation GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(quant_tests)
