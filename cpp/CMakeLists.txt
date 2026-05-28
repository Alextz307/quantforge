cmake_minimum_required(VERSION 3.20)
project(quant_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Release optimizations ──
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")

# ── Debug with sanitizers ──
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer")

# ── Profile-Guided Optimization (PGO) support ──
if(CMAKE_BUILD_TYPE STREQUAL "PGO_INSTRUMENT")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate")
elseif(CMAKE_BUILD_TYPE STREQUAL "PGO_OPTIMIZE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-use=default.profdata")
endif()

# ── Compiler warnings as errors ──
add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wshadow -Wnon-virtual-dtor)

# ── Include paths ──
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Dependencies via FetchContent ──
include(FetchContent)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)

# pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0
)

FetchContent_MakeAvailable(googletest pybind11)

# ── Core static library ──
add_library(quant_core STATIC
    src/core/types.cpp
    src/indicators/rsi.cpp
    src/indicators/macd.cpp
    src/indicators/bollinger_bands.cpp
    src/indicators/garman_klass.cpp
    src/indicators/parkinson.cpp
    src/filters/garch_filter.cpp
    src/statistics/spread.cpp
    src/strategies/state_machines.cpp
    src/strategies/pairs_trading.cpp
    src/strategies/adaptive_bollinger.cpp
    src/engine/backtest_engine.cpp
    src/metrics/performance.cpp
)
target_include_directories(quant_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Required so the static archive can be linked into the shared `quant_engine`
# pybind11 module on Linux (Apple ld is permissive; GNU ld rejects non-PIC
# objects in shared modules with R_X86_64_PC32 relocation errors).
set_target_properties(quant_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── pybind11 module ──
# Exposes BacktestEngine / MetricsCalculator / SlippageConfig (+ a `hello`
# smoke hook) as the `quant_engine` Python module. Installed into the
# `quant_engine` package so `import quant_engine` picks it up via
# `src/quant_engine/__init__.py`.
pybind11_add_module(quant_engine bindings/python_module.cpp)
target_link_libraries(quant_engine PRIVATE quant_core)
install(TARGETS quant_engine LIBRARY DESTINATION quant_engine)

# ── Tests ──
enable_testing()
add_subdirectory(tests)
