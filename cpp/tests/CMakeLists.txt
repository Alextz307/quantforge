# C++ unit tests (Google Test)

add_executable(quant_tests
    test_core.cpp
    test_indicators.cpp
    test_backtest_engine.cpp
    test_metrics.cpp
    test_garch_filter.cpp
    test_state_machines.cpp
)
target_link_libraries(quant_tests PRIVATE quant_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(quant_tests)
