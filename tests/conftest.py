"""Shared pytest fixtures for the quant trading framework."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.core.temporal import WalkForwardValidator
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

# make_declining_ohlcv_df defaults
DECLINING_OHLCV_BAND = 0.002  # symmetric OHL half-width around close

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

# Seed for synthetic feature-column noise attached via ``attach_synthetic_features``.
# Fixed so every integration test sees the same feature signal.
FEATURE_NOISE_SEED = 11


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


def make_synthetic_ohlcv_df(
    n_rows: int = SYNTH_DEFAULT_ROW_COUNT,
    start: str = SYNTH_DEFAULT_START_DATE,
    seed: int = SYNTH_DEFAULT_SEED,
    base_price: float = SYNTH_DEFAULT_BASE_PRICE,
) -> pd.DataFrame:
    """Random-walk synthetic OHLCV with valid HLOC ordering.

    Used by engine integration tests that need a full bar shape (the C++
    backtest engine requires open/high/low/close/volume). Open is the
    prior close (so bar t fills at bar t-1's close); high/low bracket
    the OHL with a small noise band.
    """
    np.random.seed(seed)
    idx = pd.bdate_range(start=start, periods=n_rows, freq="B")
    returns = np.random.normal(SYNTH_RETURN_MEAN, SYNTH_RETURN_STD, n_rows)
    close = base_price * np.cumprod(1 + returns)
    open_ = np.empty(n_rows)
    open_[0] = base_price
    open_[1:] = close[:-1]
    daily_range = np.abs(np.random.normal(0, SPY_DAILY_RANGE_STD, n_rows))
    high = np.maximum(close, open_) * (1 + daily_range)
    low = np.minimum(close, open_) * (1 - daily_range)
    # Force HLOC ordering by construction (mirrors `large_daily_df`).
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = np.full(n_rows, SYNTH_FIXED_VOLUME)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def make_walk_forward_validator(n_splits: int, test_size: int) -> WalkForwardValidator:
    """Pass-through factory shared across engine integration tests.

    Forwards only the two parameters the tests actually vary; ``gap`` and
    ``expanding`` flow through from ``WalkForwardValidator``'s own
    defaults so this helper can never drift from them.
    """
    return WalkForwardValidator(n_splits=n_splits, test_size=test_size)


def assert_params_match_constructor(
    dataclass_type: type,
    constructor_owner: type,
) -> None:
    """Assert a dataclass mirrors a class's constructor kwargs.

    Used by composite strategies to detect drift between their internal
    params dataclass and the leaf model's constructor — mypy can't enforce
    this when the dataclass is spread via ``**asdict(...)``.
    """
    import dataclasses
    import inspect

    dc_fields = {f.name for f in dataclasses.fields(dataclass_type)}
    ctor_params = set(inspect.signature(constructor_owner).parameters)
    assert dc_fields == ctor_params, (
        f"{dataclass_type.__name__} drifted from {constructor_owner.__name__}: "
        f"symmetric diff = {dc_fields ^ ctor_params}"
    )


def make_declining_close_df(
    n_rows: int = 120,
    start: str = "2022-01-03",
    start_price: float = 200.0,
    end_price: float = 100.0,
) -> pd.DataFrame:
    """Monotone-declining close series for bearish-regime tests.

    Every bar sits below its own SMA(50) / SMA(100), guaranteeing
    ``close > trend_ma`` is False throughout — useful for asserting
    that trend filters suppress long entries in strict bearish windows.
    """
    idx = pd.bdate_range(start=start, periods=n_rows, freq="B")
    close = np.linspace(start_price, end_price, n_rows)
    return pd.DataFrame({"close": close, "volume": [1e6] * n_rows}, index=idx)


def make_declining_ohlcv_df(
    n_rows: int = 120,
    start: str = "2022-01-03",
    start_price: float = 200.0,
    end_price: float = 100.0,
    band: float = DECLINING_OHLCV_BAND,
) -> pd.DataFrame:
    """Monotone-declining OHLCV series for bearish-regime tests.

    Open = prior close (first bar's open = ``start_price``); high/low bracket
    the OC range by a small symmetric ``band`` so HLOC ordering holds and the
    Garman-Klass estimator receives non-degenerate OHLC input.
    """
    idx = pd.bdate_range(start=start, periods=n_rows, freq="B")
    close = np.linspace(start_price, end_price, n_rows)
    open_ = np.empty(n_rows)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1.0 + band)
    low = np.minimum(open_, close) * (1.0 - band)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": [1e6] * n_rows},
        index=idx,
    )


def attach_synthetic_features(
    base: pd.DataFrame,
    features: list[str],
    seed: int = FEATURE_NOISE_SEED,
) -> pd.DataFrame:
    """Return a copy of ``base`` with each ``features`` column filled with
    seeded ``N(0, 1)`` noise. Used by composite + strategy tests that need a
    close (or OHLCV) frame plus a deterministic feature matrix.
    """
    rng = np.random.default_rng(seed)
    df = base.copy()
    for col in features:
        df[col] = rng.normal(0, 1, len(df))
    return df


def make_pair_close_df(
    n_rows: int = SYNTH_DEFAULT_ROW_COUNT,
    start: str = SYNTH_DEFAULT_START_DATE,
    seed: int = SYNTH_DEFAULT_SEED,
    base_price: float = SYNTH_DEFAULT_BASE_PRICE,
    scale: float = 1.2,
    noise_std: float = 0.5,
) -> pd.DataFrame:
    """Create a DataFrame with two cointegrated close-price series (close_a, close_b).

    ``close_b`` is constructed as ``close_a * scale + stationary_noise``. The
    resulting spread ``close_a - (1/scale) * close_b`` is stationary — a
    cointegrated pair that passes the Engle-Granger ADF test.
    """
    np.random.seed(seed)
    idx = pd.bdate_range(start=start, periods=n_rows, freq="B")
    returns = np.random.normal(SYNTH_RETURN_MEAN, SYNTH_RETURN_STD, n_rows)
    close_a = base_price * np.cumprod(1 + returns)
    noise = np.random.normal(0.0, noise_std, n_rows)
    close_b = close_a * scale + noise
    return pd.DataFrame({"close_a": close_a, "close_b": close_b}, index=idx)


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
