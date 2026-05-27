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


def _validate_bars_columns(bars: pd.DataFrame) -> None:
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError(
            "bars must have a DatetimeIndex; fix by setting df.index to a "
            "DatetimeIndex (or calling df.set_index('date'))."
        )
    missing = [c for c in OHLCV_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(
            f"bars missing required columns: {missing}; fix by running the "
            f"frame through DataNormalizer before invoking the engine."
        )


def _validate_inputs(bars: pd.DataFrame, signals: pd.Series) -> None:
    _validate_bars_columns(bars)
    if not signals.index.equals(bars.index):
        raise ValueError(
            "signals index must equal bars index; fix by reindexing the "
            "signal series to bars.index (a strategy that emits NaN at warmup "
            "still keeps the same index — drop or fill, never reshape)."
        )


def _bars_to_ohlcv_arrays(
    bars: pd.DataFrame,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    return (
        bars["open"].to_numpy(dtype=np.float64, copy=False),
        bars["high"].to_numpy(dtype=np.float64, copy=False),
        bars["low"].to_numpy(dtype=np.float64, copy=False),
        bars["close"].to_numpy(dtype=np.float64, copy=False),
        bars["volume"].to_numpy(dtype=np.float64, copy=False),
    )


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
    assert isinstance(bars.index, pd.DatetimeIndex)
    ts_ns: npt.NDArray[np.int64] = bars.index.values.view("int64")
    timestamps = ts_ns // _NS_PER_SECOND
    return (timestamps, *_bars_to_ohlcv_arrays(bars))


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

    def run_pairs(
        self,
        bars_a: pd.DataFrame,
        bars_b: pd.DataFrame,
        signals: pd.Series,
        hedge_ratio: float,
        slippage: SlippageConfig,
    ) -> BacktestResult:
        _validate_inputs(bars_a, signals)
        _validate_bars_columns(bars_b)
        if not bars_a.index.equals(bars_b.index):
            raise ValueError(
                "bars_a and bars_b must share the same DatetimeIndex; fix "
                "by inner-joining the two leg fetches before invoking the "
                "pairs engine."
            )

        ts, oa, ha, la, ca, va = _bars_to_arrays(bars_a)
        ob, hb, lb, cb, vb = _bars_to_ohlcv_arrays(bars_b)
        sig = signals.to_numpy(dtype=np.float64, copy=False)

        return self._engine.run_pairs(
            timestamps=ts,
            open_a=oa,
            high_a=ha,
            low_a=la,
            close_a=ca,
            volume_a=va,
            open_b=ob,
            high_b=hb,
            low_b=lb,
            close_b=cb,
            volume_b=vb,
            signals=sig,
            hedge_ratio=hedge_ratio,
            slippage=slippage,
        )
