"""Python adapter for the compiled ``quant_engine`` C++ extension.

Marshals pandas DataFrames + Series to the six contiguous numpy arrays
the binding expects (``timestamps`` int64 epoch seconds; OHLCV float64),
then dispatches to ``quant_engine.BacktestEngine``.

Validation is at the boundary only: shape + dtype + index alignment.
The engine itself enforces ``len(bars) >= 2`` (a single bar leaves no
``t+1`` to fill into) and rejects mismatched signal/bar lengths via
``std::invalid_argument`` on the C++ side.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from quant_engine import BacktestEngine, BacktestResult, SlippageConfig
from src.core.constants import OHLCV_COLUMNS
from src.engine.interface import IBacktestEngine

_NS_PER_SECOND = 1_000_000_000


def _validate_inputs(bars: pd.DataFrame, signals: pd.Series) -> None:
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must have a DatetimeIndex")
    missing = [c for c in OHLCV_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    if not signals.index.equals(bars.index):
        raise ValueError("signals index must equal bars index")


def _bars_to_arrays(
    bars: pd.DataFrame,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    assert isinstance(bars.index, pd.DatetimeIndex)  # mypy narrowing across fn boundary
    # epoch seconds keeps the wire format in a familiar range; the C++
    # `Bar.timestamp` field is opaque to the iteration loop.
    ts_ns: npt.NDArray[np.int64] = bars.index.values.view("int64")
    timestamps = ts_ns // _NS_PER_SECOND
    return (
        timestamps,
        bars["open"].to_numpy(dtype=np.float64, copy=False),
        bars["high"].to_numpy(dtype=np.float64, copy=False),
        bars["low"].to_numpy(dtype=np.float64, copy=False),
        bars["close"].to_numpy(dtype=np.float64, copy=False),
        bars["volume"].to_numpy(dtype=np.float64, copy=False),
    )


class CppBacktestEngine(IBacktestEngine):
    """Thin pandas → numpy → ``quant_engine.BacktestEngine`` adapter.

    The underlying ``BacktestEngine`` is injected so the adapter never
    duplicates the binding's defaults — pass ``BacktestEngine(...)`` to
    customize, or omit for the binding's defaults.
    """

    def __init__(self, engine: BacktestEngine | None = None) -> None:
        self._engine = engine if engine is not None else BacktestEngine()

    def run(
        self,
        bars: pd.DataFrame,
        signals: pd.Series,
        slippage: SlippageConfig,
    ) -> BacktestResult:
        _validate_inputs(bars, signals)
        ts, o, h, lo, c, v = _bars_to_arrays(bars)
        sig = signals.to_numpy(dtype=np.float64, copy=False)
        return self._engine.run(
            timestamps=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            signals=sig,
            slippage=slippage,
        )

    def run_scenarios(
        self,
        bars: pd.DataFrame,
        signals: pd.Series,
        scenarios: Sequence[SlippageConfig],
    ) -> list[BacktestResult]:
        _validate_inputs(bars, signals)
        ts, o, h, lo, c, v = _bars_to_arrays(bars)
        sig = signals.to_numpy(dtype=np.float64, copy=False)
        return self._engine.run_scenarios(
            timestamps=ts,
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            signals=sig,
            scenarios=list(scenarios),
        )
