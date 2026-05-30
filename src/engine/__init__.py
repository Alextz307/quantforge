"""
Backtest engine - Python integration over the C++ ``quant_engine``.
"""

from __future__ import annotations

from src.engine.cpp_engine import CppBacktestEngine
from src.engine.interface import IBacktestEngine
from src.engine.scenarios import (
    COST_SCENARIOS,
    SLIPPAGE_SCENARIOS,
    CostScenario,
    SlippageScenario,
    commission_bps_for,
    commission_fraction_for,
)
from src.engine.walk_forward import FoldResult, evaluate_walk_forward

__all__ = [
    "COST_SCENARIOS",
    "SLIPPAGE_SCENARIOS",
    "CostScenario",
    "CppBacktestEngine",
    "FoldResult",
    "IBacktestEngine",
    "SlippageScenario",
    "commission_bps_for",
    "commission_fraction_for",
    "evaluate_walk_forward",
]
