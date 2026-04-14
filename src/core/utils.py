"""Shared domain utilities for the quant trading framework."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(close: pd.Series[float]) -> pd.Series[float]:
    """Compute log returns from a close-price series.

    Equivalent to ``log(close[t] / close[t-1])``.  The first value is
    NaN (no prior close).  Callers should ``.dropna()`` when needed.
    """
    result: pd.Series[float] = np.log1p(close.pct_change())  # type: ignore[assignment]
    return result
