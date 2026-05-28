"""
Per-universe reference baselines for honest strategy evaluation.

A backtested strategy on its own is uninterpretable — a Sharpe of 0.8
sounds respectable until you learn the underlying universe (e.g. SPY
2010-2020) buy-and-hold scored 1.1 under the same slippage assumptions.
The baseline computed here is the long-only "do nothing" benchmark: hold
1.0 unit notional in the universe's primary asset from the first bar to
the last, paying identical entry / exit slippage to the strategy.

Anti-leakage note
-----------------
The baseline holds a constant signal of 1.0 across every bar — there is
no fit step, no parameter to leak. The function takes a slice of bars
already known to the caller (typically the holdout window) and re-runs
the SAME ``IBacktestEngine`` with the SAME slippage scenario the
strategy used. No new fingerprint, no new state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_engine import MetricsCalculator, SlippageConfig
from src.core import json_io
from src.core.types import Interval
from src.engine.interface import IBacktestEngine


@dataclass(frozen=True)
class BaselineResult:
    """
    Long-only buy-and-hold metric snapshot.

    Mirrors the metric surface of :class:`FoldRecord` / per-leg holdout
    so callers can render the baseline alongside strategy results
    without case-splitting on the shape.
    """

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    annualized_return: float
    annualized_volatility: float
    total_return: float
    win_rate: float
    trade_count: int
    equity_curve: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "total_return": self.total_return,
            "win_rate": self.win_rate,
            "trade_count": self.trade_count,
            "equity_curve": list(self.equity_curve),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> BaselineResult:
        return cls(
            sharpe_ratio=json_io.get_float(d, "sharpe_ratio"),
            sortino_ratio=json_io.get_float(d, "sortino_ratio"),
            calmar_ratio=json_io.get_float(d, "calmar_ratio"),
            max_drawdown=json_io.get_float(d, "max_drawdown"),
            annualized_return=json_io.get_float(d, "annualized_return"),
            annualized_volatility=json_io.get_float(d, "annualized_volatility"),
            total_return=json_io.get_float(d, "total_return"),
            win_rate=json_io.get_float(d, "win_rate"),
            trade_count=json_io.get_int(d, "trade_count"),
            equity_curve=tuple(json_io.get_float_list(d, "equity_curve")),
        )


def compute_buy_and_hold(
    bars: pd.DataFrame,
    *,
    slippage: SlippageConfig,
    interval: Interval,
    engine: IBacktestEngine,
    risk_free_rate: float = 0.0,
) -> BaselineResult:
    """
    Run a long-only baseline on a canonical OHLCV frame.

    The frame must already be sliced to the evaluation window the
    caller wants the baseline computed against (typically the holdout
    region) and in canonical ``open/high/low/close/volume`` form.
    Pairs / multi-feature universes should slice the primary leg
    before calling — see :func:`src.engine.walk_forward.split_pairs_frame`
    and :func:`slice_primary_ohlcv`.

    Args:
        bars: OHLCV frame with a DatetimeIndex covering the evaluation
            window. Must have at least 2 bars.
        slippage: Same slippage scenario the strategy used — the
            baseline must pay the same friction or the comparison is
            unfair.
        interval: Bar interval, used for the annualisation factor.
        engine: The same backtest engine instance the strategy uses;
            passed in to keep the analysis package decoupled from any
            concrete engine implementation.
        risk_free_rate: Per-period risk-free rate for Sharpe / Sortino
            (default 0.0).
    """

    if len(bars) < 2:
        raise ValueError(
            f"compute_buy_and_hold needs at least 2 bars, got {len(bars)}; "
            f"fix by widening the evaluation window."
        )
    signals = pd.Series(np.ones(len(bars), dtype=np.float64), index=bars.index)
    raw = engine.run(bars, signals, slippage)
    metrics = MetricsCalculator.compute(
        raw.equity_curve,
        interval.annualization_factor(),
        risk_free_rate,
    )
    return BaselineResult(
        sharpe_ratio=metrics.sharpe_ratio,
        sortino_ratio=metrics.sortino_ratio,
        calmar_ratio=metrics.calmar_ratio,
        max_drawdown=metrics.max_drawdown,
        annualized_return=metrics.annualized_return,
        annualized_volatility=metrics.annualized_volatility,
        total_return=raw.total_return,
        win_rate=metrics.win_rate,
        trade_count=raw.trade_count,
        equity_curve=tuple(raw.equity_curve.tolist()),
    )


__all__ = ["BaselineResult", "compute_buy_and_hold"]
