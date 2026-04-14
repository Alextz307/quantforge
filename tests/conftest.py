"""Shared pytest fixtures for the quant trading framework."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.core.types import BarData, Interval

# Deterministic seeds applied across all tests via the `deterministic_seed` fixture
GLOBAL_NUMPY_SEED = 42
GLOBAL_TORCH_SEED = 42

# sample_spy_df: small SPY-like OHLCV
SPY_ROW_COUNT = 100
SPY_START_DATE = "2023-01-02"
SPY_BASE_PRICE = 400.0
SPY_RETURN_MEAN = 0.0005  # ~12.6% annualized drift
SPY_RETURN_STD = 0.01  # ~16% annualized vol
SPY_DAILY_RANGE_STD = 0.005  # std of |high-close|/close
SPY_OPEN_OFFSET_STD = 0.002  # std of (open-close)/close
SPY_VOLUME_LOW = 50_000_000
SPY_VOLUME_HIGH = 150_000_000
SPY_FIXTURE_SEED = 42

# sample_bar_data: synthetic BarData ladder
BAR_LADDER_COUNT = 10
BAR_LADDER_BASE_PRICE = 100.0
BAR_LADDER_BASE_VOLUME = 1_000_000.0

# make_synthetic_close_df defaults
SYNTH_DEFAULT_ROW_COUNT = 200
SYNTH_DEFAULT_START_DATE = "2020-01-02"
SYNTH_DEFAULT_SEED = 42
SYNTH_DEFAULT_BASE_PRICE = 100.0
SYNTH_RETURN_MEAN = 0.0003  # ~7.6% annualized drift
SYNTH_RETURN_STD = 0.012  # ~19% annualized vol
SYNTH_FIXED_VOLUME = 1e6

# make_daily_df defaults
DAILY_DEFAULT_START_DATE = "2020-01-01"
DAILY_FIXED_VOLUME = 1000

# large_daily_df: 8-year synthetic OHLCV for walk-forward testing
LARGE_ROW_COUNT = 2000
LARGE_START_DATE = "2016-01-04"
LARGE_BASE_PRICE = 200.0
LARGE_VOLUME_LOW = 30_000_000
LARGE_VOLUME_HIGH = 100_000_000
LARGE_FIXTURE_SEED = 123

# Hourly-interval fixture parameters shared by hybrid tests
HOURLY_ROW_COUNT = 250
HOURLY_START = "2020-01-02 09:30"
HOURLY_RETURN_STD = 0.005
HOURLY_BASE_PRICE = 100.0


@pytest.fixture
def deterministic_seed() -> None:
    """Set deterministic random seeds for reproducibility."""
    np.random.seed(GLOBAL_NUMPY_SEED)
    try:
        import torch

        torch.manual_seed(GLOBAL_TORCH_SEED)
    except ImportError:
        pass


@pytest.fixture
def synthetic_feature_columns() -> list[str]:
    """Two synthetic feature column names used by composite-model tests."""
    return ["feat_a", "feat_b"]


@pytest.fixture
def sample_spy_df() -> pd.DataFrame:
    """Small realistic SPY OHLCV DataFrame for unit testing.

    Returns ``SPY_ROW_COUNT`` business days of synthetic OHLCV starting
    from ``SPY_START_DATE``, with ``SPY_BASE_PRICE`` and ``SPY_RETURN_STD``
    daily volatility. OHLC constraints are enforced
    (high >= max(open, close), low <= min(open, close)).
    """
    np.random.seed(SPY_FIXTURE_SEED)
    idx = pd.bdate_range(start=SPY_START_DATE, periods=SPY_ROW_COUNT, freq="B")

    # Generate realistic price series with random walk
    returns = np.random.normal(SPY_RETURN_MEAN, SPY_RETURN_STD, SPY_ROW_COUNT)
    close = SPY_BASE_PRICE * np.cumprod(1 + returns)

    # Generate OHLCV from close
    daily_range = np.abs(np.random.normal(0, SPY_DAILY_RANGE_STD, SPY_ROW_COUNT))
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)

    # Open is close shifted by a small random amount
    open_offset = np.random.normal(0, SPY_OPEN_OFFSET_STD, SPY_ROW_COUNT)
    open_price = close * (1 + open_offset)

    # Ensure OHLC constraints: high >= max(open, close), low <= min(open, close)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.randint(SPY_VOLUME_LOW, SPY_VOLUME_HIGH, SPY_ROW_COUNT).astype(float)

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
            open=BAR_LADDER_BASE_PRICE + i,
            high=BAR_LADDER_BASE_PRICE + 5.0 + i,
            low=BAR_LADDER_BASE_PRICE - 1.0 + i,
            close=BAR_LADDER_BASE_PRICE + 3.0 + i,
            volume=BAR_LADDER_BASE_VOLUME + i * 100_000,
            interval=Interval.DAILY,
        )
        for i in range(BAR_LADDER_COUNT)
    ]


def make_synthetic_close_df(
    n_rows: int = SYNTH_DEFAULT_ROW_COUNT,
    start: str = SYNTH_DEFAULT_START_DATE,
    seed: int = SYNTH_DEFAULT_SEED,
    base_price: float = SYNTH_DEFAULT_BASE_PRICE,
) -> pd.DataFrame:
    """Create a DataFrame with realistic close prices and volume.

    Not a fixture — call directly with parameters. Uses random-walk
    returns with ``SYNTH_RETURN_MEAN`` mean and ``SYNTH_RETURN_STD``
    daily volatility.
    """
    np.random.seed(seed)
    idx = pd.bdate_range(start=start, periods=n_rows, freq="B")
    returns = np.random.normal(SYNTH_RETURN_MEAN, SYNTH_RETURN_STD, n_rows)
    close = base_price * np.cumprod(1 + returns)
    return pd.DataFrame({"close": close, "volume": [SYNTH_FIXED_VOLUME] * n_rows}, index=idx)


def make_daily_df(n_rows: int, start: str = DAILY_DEFAULT_START_DATE) -> pd.DataFrame:
    """Create a simple DataFrame with DatetimeIndex for testing.

    Not a fixture — call directly with parameters. For minimal test DataFrames
    that only need 'close' and 'volume' columns.
    """
    idx = pd.bdate_range(start=start, periods=n_rows, freq="B")
    return pd.DataFrame(
        {"close": range(n_rows), "volume": [DAILY_FIXED_VOLUME] * n_rows},
        index=idx,
    )


@pytest.fixture
def large_daily_df() -> pd.DataFrame:
    """Large DataFrame (~``LARGE_ROW_COUNT`` rows) for walk-forward validation testing.

    Returns synthetic OHLCV starting from ``LARGE_START_DATE``, with
    ``LARGE_BASE_PRICE`` and ``SYNTH_RETURN_STD`` daily volatility.
    Suitable for WalkForwardValidator tests requiring multiple folds.
    """
    np.random.seed(LARGE_FIXTURE_SEED)
    idx = pd.bdate_range(start=LARGE_START_DATE, periods=LARGE_ROW_COUNT, freq="B")

    returns = np.random.normal(SYNTH_RETURN_MEAN, SYNTH_RETURN_STD, LARGE_ROW_COUNT)
    close = LARGE_BASE_PRICE * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, SPY_DAILY_RANGE_STD, LARGE_ROW_COUNT)))
    low = close * (1 - np.abs(np.random.normal(0, SPY_DAILY_RANGE_STD, LARGE_ROW_COUNT)))
    open_price = close * (1 + np.random.normal(0, SPY_OPEN_OFFSET_STD, LARGE_ROW_COUNT))

    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.randint(LARGE_VOLUME_LOW, LARGE_VOLUME_HIGH, LARGE_ROW_COUNT).astype(float)

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
