"""
Backtest engine — Python integration over the C++ ``quant_engine``.
"""

from __future__ import annotations

from src.engine.cpp_engine import CppBacktestEngine
from src.engine.interface import IBacktestEngine
from src.engine.scenarios import SLIPPAGE_SCENARIOS, SlippageScenario
from src.engine.walk_forward import FoldResult, evaluate_walk_forward

__all__ = [
    "SLIPPAGE_SCENARIOS",
    "CppBacktestEngine",
    "FoldResult",
    "IBacktestEngine",
    "SlippageScenario",
    "evaluate_walk_forward",
]
