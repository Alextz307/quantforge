"""
Python wrapper around the compiled ``quant_engine`` C++ extension.
"""

from __future__ import annotations

from quant_engine.quant_engine import (
    BacktestEngine,
    BacktestResult,
    MetricsCalculator,
    PerformanceMetrics,
    SlippageConfig,
    SlippageModel,
    hello,
)

__all__: list = [
    "BacktestEngine",
    "BacktestResult",
    "MetricsCalculator",
    "PerformanceMetrics",
    "SlippageConfig",
    "SlippageModel",
    "hello",
]
