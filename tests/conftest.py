"""Shared pytest fixtures for the quant trading framework."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.core.types import BarData, Interval


@pytest.fixture
def deterministic_seed() -> None:
    """Set deterministic random seeds for reproducibility."""
    np.random.seed(42)
    try:
        import torch

        torch.manual_seed(42)
    except ImportError:
        pass


@pytest.fixture
def sample_spy_df() -> pd.DataFrame:
    """Small realistic SPY OHLCV DataFrame for unit testing.

    Returns 100 business days of synthetic OHLCV starting from 2023-01-02,
    with base price ~$400 and 1% daily volatility. OHLC constraints are
    enforced (high >= max(open, close), low <= min(open, close)).
    """
    np.random.seed(42)
    n = 100
    idx = pd.bdate_range(start="2023-01-02", periods=n, freq="B")

    # Generate realistic price series with random walk
    base_price = 400.0
    returns = np.random.normal(0.0005, 0.01, n)
    close = base_price * np.cumprod(1 + returns)

    # Generate OHLCV from close
    daily_range = np.abs(np.random.normal(0, 0.005, n))
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)

    # Open is close shifted by a small random amount
    open_offset = np.random.normal(0, 0.002, n)
    open_price = close * (1 + open_offset)

    # Ensure OHLC constraints: high >= max(open, close), low <= min(open, close)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.randint(50_000_000, 150_000_000, n).astype(float)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


@pytest.fixture
def sample_bar_data() -> list[BarData]:
    """List of BarData instances for testing."""
    return [
        BarData(
            timestamp=datetime(2024, 1, i + 1),
            open=100.0 + i,
            high=105.0 + i,
            low=99.0 + i,
            close=103.0 + i,
            volume=1_000_000.0 + i * 100_000,
            interval=Interval.DAILY,
        )
        for i in range(10)
    ]


def make_synthetic_close_df(
    n_rows: int = 200,
    start: str = "2020-01-02",
    seed: int = 42,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Create a DataFrame with realistic close prices and volume.

    Not a fixture — call directly with parameters. Uses random-walk
    returns with ~0.03% mean and ~1.2% daily volatility.
    """
    np.random.seed(seed)
    idx = pd.bdate_range(start=start, periods=n_rows, freq="B")
    returns = np.random.normal(0.0003, 0.012, n_rows)
    close = base_price * np.cumprod(1 + returns)
    return pd.DataFrame({"close": close, "volume": [1e6] * n_rows}, index=idx)


def make_daily_df(n_rows: int, start: str = "2020-01-01") -> pd.DataFrame:
    """Create a simple DataFrame with DatetimeIndex for testing.

    Not a fixture — call directly with parameters. For minimal test DataFrames
    that only need 'close' and 'volume' columns.
    """
    idx = pd.bdate_range(start=start, periods=n_rows, freq="B")
    return pd.DataFrame(
        {"close": range(n_rows), "volume": [1000] * n_rows},
        index=idx,
    )


@pytest.fixture
def large_daily_df() -> pd.DataFrame:
    """Large DataFrame (~2000 rows) for walk-forward validation testing.

    Returns 2000 business days (~8 years) of synthetic OHLCV starting from
    2016-01-04, with base price ~$200 and 1.2% daily volatility. Suitable
    for WalkForwardValidator tests requiring multiple folds.
    """
    np.random.seed(123)
    n = 2000
    idx = pd.bdate_range(start="2016-01-04", periods=n, freq="B")

    returns = np.random.normal(0.0003, 0.012, n)
    close = 200.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_price = close * (1 + np.random.normal(0, 0.002, n))

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.randint(30_000_000, 100_000_000, n).astype(float)

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
