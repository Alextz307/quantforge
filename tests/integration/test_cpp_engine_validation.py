"""
Boundary validation for ``CppBacktestEngine``.

The C++ binding only checks ``len(bars) == len(signals)``; the Python
adapter is the layer that enforces the *pandas-shaped* contract
(DatetimeIndex, OHLCV columns present, signals index aligned). These
tests lock in those three failure modes so a future refactor can't
silently strip them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine import SLIPPAGE_SCENARIOS, CppBacktestEngine, SlippageScenario
from tests.conftest import make_synthetic_ohlcv_df

VALIDATION_N_ROWS = 50
SHIFTED_INDEX_START = "2030-01-02"
NORMAL_SCENARIO = SLIPPAGE_SCENARIOS[SlippageScenario.NORMAL]


def _zero_signals(index: pd.Index) -> pd.Series:
    return pd.Series(np.zeros(len(index), dtype=np.float64), index=index)


def test_run_rejects_non_datetime_index() -> None:
    bars = make_synthetic_ohlcv_df(n_rows=VALIDATION_N_ROWS).reset_index(drop=True)
    signals = _zero_signals(bars.index)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        CppBacktestEngine().run(bars, signals, NORMAL_SCENARIO)


def test_run_rejects_missing_columns() -> None:
    bars = make_synthetic_ohlcv_df(n_rows=VALIDATION_N_ROWS).drop(columns=["volume"])
    signals = _zero_signals(bars.index)
    with pytest.raises(ValueError, match="missing required columns"):
        CppBacktestEngine().run(bars, signals, NORMAL_SCENARIO)


def test_run_rejects_misaligned_signals_index() -> None:
    bars = make_synthetic_ohlcv_df(n_rows=VALIDATION_N_ROWS)
    shifted_idx = pd.bdate_range(start=SHIFTED_INDEX_START, periods=VALIDATION_N_ROWS)
    signals = _zero_signals(shifted_idx)
    with pytest.raises(ValueError, match="signals index"):
        CppBacktestEngine().run(bars, signals, NORMAL_SCENARIO)


def test_run_scenarios_validates_inputs_too() -> None:
    """
    The same _validate_inputs path covers both run and run_scenarios.
    """

    bars = make_synthetic_ohlcv_df(n_rows=VALIDATION_N_ROWS).drop(columns=["close"])
    signals = _zero_signals(bars.index)
    with pytest.raises(ValueError, match="missing required columns"):
        CppBacktestEngine().run_scenarios(bars, signals, [NORMAL_SCENARIO])
