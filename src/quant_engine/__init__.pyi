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
]
