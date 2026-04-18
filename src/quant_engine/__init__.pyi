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
    GarmanKlass,
    MACDResult,
    MetricsCalculator,
    Parkinson,
    PerformanceMetrics,
    SlippageConfig,
    SlippageModel,
    hello,
)

__all__: list = [
    "MACD",
    "RSI",
    "BacktestEngine",
    "BacktestResult",
    "BollingerBands",
    "BollingerResult",
    "GarmanKlass",
    "MACDResult",
    "MetricsCalculator",
    "Parkinson",
    "PerformanceMetrics",
    "SlippageConfig",
    "SlippageModel",
    "hello",
]
