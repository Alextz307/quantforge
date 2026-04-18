"""
Python wrapper around the compiled ``quant_engine`` C++ extension.
"""

from __future__ import annotations

from quant_engine.quant_engine import (
    MACD,
    RSI,
    BacktestEngine,
    BacktestResult,
    BollingerBands,
    BollingerResult,
    GarchParams,
    GarmanKlass,
    MACDResult,
    MetricsCalculator,
    Parkinson,
    PerformanceMetrics,
    SlippageConfig,
    SlippageModel,
    garch_filter,
    hello,
    run_mean_reversion_state_machine,
    run_pairs_state_machine,
)

__all__: list = [
    "MACD",
    "RSI",
    "BacktestEngine",
    "BacktestResult",
    "BollingerBands",
    "BollingerResult",
    "GarchParams",
    "GarmanKlass",
    "MACDResult",
    "MetricsCalculator",
    "Parkinson",
    "PerformanceMetrics",
    "SlippageConfig",
    "SlippageModel",
    "garch_filter",
    "hello",
    "run_mean_reversion_state_machine",
    "run_pairs_state_machine",
]
