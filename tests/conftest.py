"""Shared pytest fixtures for the quant trading framework."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis.metrics_aggregator import AggregateStats
from src.benchmarking.types import BenchmarkResult, BenchmarkRun, HardwareInfo
from src.core.temporal import WalkForwardValidator
from src.core.types import BarData, Interval
from src.engine.scenarios import SlippageScenario
from src.orchestration.manifest import Manifest
from src.orchestration.types import ExperimentResult, FoldRecord
from tests import _strategy_stubs as _strategy_stubs  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="session")
def _isolate_webapp_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Point ``WEBAPP_DB_PATH`` at a session-local tmp file.

    The CLI subcommands (``experiment run/tune/compare/holdout-eval``,
    ``study run``) attribute their artifacts via ``attribute_via_username``,
    which opens the webapp DB at ``WEBAPP_DB_PATH``. Without this fixture,
    framework tests invoking those subcommands via ``CliRunner`` would
    insert synthetic jobs rows into the developer's actual
    ``webapp/data/webapp.sqlite`` whenever they ran.
    """
    db_path = tmp_path_factory.mktemp("webapp_db") / "test_webapp.sqlite"
    os.environ["WEBAPP_DB_PATH"] = str(db_path)
    return db_path


def load_script_module(path: Path, name: str) -> ModuleType:
    """Import a top-level script (``scripts/foo.py``) by path for testing."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GLOBAL_NUMPY_SEED = 42
GLOBAL_TORCH_SEED = 42

SPY_ROW_COUNT = 100
SPY_START_DATE = "2023-01-02"
SPY_BASE_PRICE = 400.0
SPY_RETURN_MEAN = 0.0005
SPY_RETURN_STD = 0.01
SPY_DAILY_RANGE_STD = 0.005
SPY_OPEN_OFFSET_STD = 0.002
SPY_VOLUME_LOW = 50_000_000
SPY_VOLUME_HIGH = 150_000_000
SPY_FIXTURE_SEED = 42

BAR_LADDER_COUNT = 10
BAR_LADDER_BASE_PRICE = 100.0
BAR_LADDER_BASE_VOLUME = 1_000_000.0

DECLINING_OHLCV_BAND = 0.002

SYNTH_DEFAULT_ROW_COUNT = 200
SYNTH_DEFAULT_START_DATE = "2020-01-02"
SYNTH_DEFAULT_SEED = 42
SYNTH_DEFAULT_BASE_PRICE = 100.0
SYNTH_RETURN_MEAN = 0.0003
SYNTH_RETURN_STD = 0.012
SYNTH_FIXED_VOLUME = 1e6

DAILY_DEFAULT_START_DATE = "2020-01-01"
DAILY_FIXED_VOLUME = 1000

LARGE_ROW_COUNT = 2000
LARGE_START_DATE = "2016-01-04"
LARGE_BASE_PRICE = 200.0
LARGE_VOLUME_LOW = 30_000_000
LARGE_VOLUME_HIGH = 100_000_000
LARGE_FIXTURE_SEED = 123

HOURLY_ROW_COUNT = 250
HOURLY_START = "2020-01-02 09:30"
HOURLY_RETURN_STD = 0.005
HOURLY_BASE_PRICE = 100.0

FEATURE_NOISE_SEED = 11


def seed_globally() -> None:
    """Plain callable wrapper around the deterministic seeding logic.

    Use this inside tests that need to re-seed multiple times per test
    body (e.g., to compare two runs with identical starting state). A
    pytest fixture would only fire once per test.
    """
    np.random.seed(GLOBAL_NUMPY_SEED)
    try:
        import torch

        torch.manual_seed(GLOBAL_TORCH_SEED)
    except ImportError:
        pass


@pytest.fixture
def deterministic_seed() -> None:
    """Set deterministic random seeds for reproducibility."""
    seed_globally()


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

    returns = np.random.normal(SPY_RETURN_MEAN, SPY_RETURN_STD, SPY_ROW_COUNT)
    close = SPY_BASE_PRICE * np.cumprod(1 + returns)

    daily_range = np.abs(np.random.normal(0, SPY_DAILY_RANGE_STD, SPY_ROW_COUNT))
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)

    open_offset = np.random.normal(0, SPY_OPEN_OFFSET_STD, SPY_ROW_COUNT)
    open_price = close * (1 + open_offset)

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
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = np.full(n_rows, SYNTH_FIXED_VOLUME)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


_MINI_SMOKE_TICKER = "MINI"
_MINI_SMOKE_DEFAULT_NAME = "mini_smoke"
_MINI_SMOKE_DEFAULT_ROWS = 300
_MINI_SMOKE_DEFAULT_HOLDOUT_PCT = 0.15
_MINI_SMOKE_DEFAULT_N_SPLITS = 2
_MINI_SMOKE_DEFAULT_TEST_SIZE = 60
_MINI_SMOKE_DEFAULT_GAP = 1
_MINI_SMOKE_DEFAULT_SEED = 42
_MINI_SMOKE_START_DATE = "2020-01-02"


def make_mini_experiment_fixture(
    tmp_path: Path,
    *,
    name: str = _MINI_SMOKE_DEFAULT_NAME,
    holdout_pct: float = _MINI_SMOKE_DEFAULT_HOLDOUT_PCT,
    n_rows: int = _MINI_SMOKE_DEFAULT_ROWS,
    n_splits: int = _MINI_SMOKE_DEFAULT_N_SPLITS,
    test_size: int = _MINI_SMOKE_DEFAULT_TEST_SIZE,
    gap: int = _MINI_SMOKE_DEFAULT_GAP,
) -> Path:
    """Write the canonical synthetic OHLCV CSV + AdaptiveBollinger YAML for CLI smoke tests.

    Returns the YAML path. The CSV lives at ``<tmp_path>/csv_data/MINI.csv``
    and is regenerated per call (no caching — pytest's ``tmp_path`` is
    already per-test). Every CLI smoke test (``run``, ``holdout-eval``,
    future ``compare`` / ``study``) needs the same skeleton — only
    ``holdout_pct`` and ``name`` typically vary, so the helper takes both
    as kwargs and otherwise pins to a known-fast AdaptiveBollinger config
    (``garch_p_max=garch_q_max=1`` so the AIC grid finishes in seconds).
    """
    csv_dir = tmp_path / "csv_data"
    csv_dir.mkdir()
    df = make_synthetic_ohlcv_df(n_rows=n_rows, start=_MINI_SMOKE_START_DATE)
    df.index.name = "date"
    df.to_csv(csv_dir / f"{_MINI_SMOKE_TICKER}.csv")

    cfg_payload: dict[str, object] = {
        "name": name,
        "seed": _MINI_SMOKE_DEFAULT_SEED,
        "data": {
            "source": {"name": "csv", "params": {"data_dir": str(csv_dir)}},
            "tickers": [_MINI_SMOKE_TICKER],
            "start": datetime(2020, 1, 2).isoformat(),
            "end": datetime(2022, 1, 1).isoformat(),
            "interval": "daily",
        },
        "strategy": {
            "name": "AdaptiveBollinger",
            "params": {
                "window": 20,
                "trend_window": 50,
                "garch_p_max": 1,
                "garch_q_max": 1,
            },
        },
        "validation": {
            "n_splits": n_splits,
            "test_size": test_size,
            "gap": gap,
            "holdout_pct": holdout_pct,
        },
        "slippage": {"scenario": "normal"},
    }
    yaml_path = tmp_path / f"{name}.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(cfg_payload, f)
    return yaml_path


_PAIR_MINI_TICKER_A = "PAIR_A"
_PAIR_MINI_TICKER_B = "PAIR_B"
_PAIR_MINI_DEFAULT_NAME = "pair_mini_smoke"
_PAIR_MINI_DEFAULT_ROWS = 300
_PAIR_MINI_DEFAULT_N_SPLITS = 2
_PAIR_MINI_DEFAULT_TEST_SIZE = 60
_PAIR_MINI_DEFAULT_GAP = 1
_PAIR_MINI_DEFAULT_SEED = 42
_PAIR_MINI_START_DATE = "2020-01-02"
# Cointegration parameters chosen so Engle-Granger ADF rejects the unit
# root on residuals at p < 0.05 for any lookback >= ~50 bars.
_PAIR_MINI_BETA = 0.5
_PAIR_MINI_ALPHA = 50.0
_PAIR_MINI_NOISE_STD = 1.0
_PAIR_MINI_OHLC_BAND_STD = 0.005


def make_pair_mini_experiment_fixture(
    tmp_path: Path,
    *,
    name: str = _PAIR_MINI_DEFAULT_NAME,
    n_rows: int = _PAIR_MINI_DEFAULT_ROWS,
    n_splits: int = _PAIR_MINI_DEFAULT_N_SPLITS,
    test_size: int = _PAIR_MINI_DEFAULT_TEST_SIZE,
    gap: int = _PAIR_MINI_DEFAULT_GAP,
) -> Path:
    """Write two cointegrated synthetic OHLCV CSVs + a PairsTrading YAML.

    Sibling to :func:`make_mini_experiment_fixture` for the pairs CLI smoke.
    Construction:

    * Leg A is :func:`make_synthetic_ohlcv_df` (its own fixed seed → reproducible
      random-walk close).
    * Leg B has ``close_b = alpha + beta * close_a + N(0, sigma)`` with iid
      Gaussian noise. The pair is cointegrated by construction — Engle-Granger
      passes for any reasonable training window — so the strategy's
      ``train()`` does not raise ``"pair not cointegrated"`` on small folds.
    * Both legs share the same business-day index (the orchestrator's pair
      fetch path inner-joins on timestamps; identical indices avoid silent
      bar drops that would invalidate the row-count assertion).

    No ``holdout_pct`` knob: pairs smoke does not exercise the holdout
    boundary, and including it here would require carrying a tighter
    ``zscore_lookback`` to keep the dev region long enough.
    """
    csv_dir = tmp_path / "csv_data"
    csv_dir.mkdir()

    df_a = make_synthetic_ohlcv_df(n_rows=n_rows, start=_PAIR_MINI_START_DATE)
    df_a.index.name = "date"
    df_a.to_csv(csv_dir / f"{_PAIR_MINI_TICKER_A}.csv")

    rng = np.random.default_rng(_PAIR_MINI_DEFAULT_SEED)
    close_a = np.asarray(df_a["close"], dtype=np.float64)
    spread_noise = rng.normal(0.0, _PAIR_MINI_NOISE_STD, n_rows)
    close_b = _PAIR_MINI_ALPHA + _PAIR_MINI_BETA * close_a + spread_noise

    open_b = np.empty(n_rows)
    open_b[0] = close_b[0]
    open_b[1:] = close_b[:-1]
    band = np.abs(rng.normal(0.0, _PAIR_MINI_OHLC_BAND_STD, n_rows))
    high_b = np.maximum(close_b, open_b) * (1 + band)
    low_b = np.minimum(close_b, open_b) * (1 - band)
    high_b = np.maximum(high_b, np.maximum(open_b, close_b))
    low_b = np.minimum(low_b, np.minimum(open_b, close_b))
    volume_b = np.full(n_rows, SYNTH_FIXED_VOLUME)
    df_b = pd.DataFrame(
        {"open": open_b, "high": high_b, "low": low_b, "close": close_b, "volume": volume_b},
        index=df_a.index,
    )
    df_b.index.name = "date"
    df_b.to_csv(csv_dir / f"{_PAIR_MINI_TICKER_B}.csv")

    cfg_payload: dict[str, object] = {
        "name": name,
        "seed": _PAIR_MINI_DEFAULT_SEED,
        "data": {
            "source": {"name": "csv", "params": {"data_dir": str(csv_dir)}},
            "tickers": [_PAIR_MINI_TICKER_A, _PAIR_MINI_TICKER_B],
            "start": datetime(2020, 1, 2).isoformat(),
            "end": datetime(2022, 1, 1).isoformat(),
            "interval": "daily",
        },
        "strategy": {
            "name": "PairsTrading",
            "params": {
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "stop_loss_zscore": 4.0,
                "zscore_lookback": 30,
                "p_value_threshold": 0.05,
            },
        },
        "validation": {
            "n_splits": n_splits,
            "test_size": test_size,
            "gap": gap,
        },
        "slippage": {"scenario": "normal"},
    }
    yaml_path = tmp_path / f"{name}.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(cfg_payload, f)
    return yaml_path


_MULTI_FEATURE_MINI_DEFAULT_NAME = "multi_feature_mini_smoke"
_MULTI_FEATURE_MINI_DEFAULT_ROWS = 250
_MULTI_FEATURE_MINI_DEFAULT_N_SPLITS = 2
_MULTI_FEATURE_MINI_DEFAULT_TEST_SIZE = 50
_MULTI_FEATURE_MINI_DEFAULT_GAP = 1
_MULTI_FEATURE_MINI_DEFAULT_SEED = 17
_MULTI_FEATURE_MINI_START_DATE = "2020-01-02"


def make_multi_feature_mini_experiment_fixture(
    tmp_path: Path,
    *,
    strategy_name: str,
    primary_ticker: str,
    feature_tickers: tuple[str, ...],
    name: str = _MULTI_FEATURE_MINI_DEFAULT_NAME,
    n_rows: int = _MULTI_FEATURE_MINI_DEFAULT_ROWS,
    n_splits: int = _MULTI_FEATURE_MINI_DEFAULT_N_SPLITS,
    test_size: int = _MULTI_FEATURE_MINI_DEFAULT_TEST_SIZE,
    gap: int = _MULTI_FEATURE_MINI_DEFAULT_GAP,
    extra_strategy_params: dict[str, object] | None = None,
) -> Path:
    """Drop one synthetic OHLCV CSV per ticker + a multi-feature YAML.

    Sibling to :func:`make_pair_mini_experiment_fixture` for multi-feature
    smoke tests. Each ticker gets its own deterministic random walk (different
    seed offset per ticker so the inner-join produces distinct columns); all
    legs share the same business-day index so the orchestrator's multi-feature
    inner-join doesn't drop any bars on synthetic data.

    ``strategy_name`` MUST already be registered on ``strategy_registry``
    when ``build_experiment`` runs — caller's responsibility (typically by
    importing the module that contains ``@strategy_registry.register(...)``).
    The helper writes the YAML; it does not import strategies.
    """
    csv_dir = tmp_path / "csv_data"
    csv_dir.mkdir()

    tickers = (primary_ticker, *feature_tickers)
    for offset, ticker in enumerate(tickers):
        df = make_synthetic_ohlcv_df(
            n_rows=n_rows,
            start=_MULTI_FEATURE_MINI_START_DATE,
            seed=_MULTI_FEATURE_MINI_DEFAULT_SEED + offset,
        )
        df.index.name = "date"
        df.to_csv(csv_dir / f"{ticker}.csv")

    params: dict[str, object] = {"primary_ticker": primary_ticker}
    if feature_tickers:
        params["feature_tickers"] = list(feature_tickers)
    if extra_strategy_params is not None:
        params.update(extra_strategy_params)

    cfg_payload: dict[str, object] = {
        "name": name,
        "seed": _MULTI_FEATURE_MINI_DEFAULT_SEED,
        "data": {
            "source": {"name": "csv", "params": {"data_dir": str(csv_dir)}},
            "tickers": list(tickers),
            "start": datetime(2020, 1, 2).isoformat(),
            "end": datetime(2021, 12, 31).isoformat(),
            "interval": "daily",
        },
        "strategy": {"name": strategy_name, "params": params},
        "validation": {"n_splits": n_splits, "test_size": test_size, "gap": gap},
        "slippage": {"scenario": "normal"},
    }
    yaml_path = tmp_path / f"{name}.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(cfg_payload, f)
    return yaml_path


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
    *,
    ignore: Iterable[str] = (),
) -> None:
    """Assert a dataclass mirrors a class's constructor kwargs.

    Used to detect drift between an internal params dataclass and the
    constructor it shadows — mypy can't enforce this when the dataclass
    is spread via ``**asdict(...)``.
    """
    import dataclasses
    import inspect

    dc_fields = {f.name for f in dataclasses.fields(dataclass_type)}
    ctor_params = set(inspect.signature(constructor_owner).parameters) - set(ignore)
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


BENCH_TEST_RUN_ID = "rid"
BENCH_TEST_TIMESTAMP = "2026-04-20T12:00:00Z"
BENCH_TEST_HW_CPU = "test-cpu"
BENCH_TEST_HW_RAM_GB = 1.0
BENCH_TEST_RESULT_N = 10_000
BENCH_TEST_RESULT_NS = 50_000.0


def make_benchmark_hardware(
    *,
    cpu_brand: str = BENCH_TEST_HW_CPU,
    cpu_count: int = 1,
    ram_gb: float = BENCH_TEST_HW_RAM_GB,
    os_name: str = "test-os",
    os_version: str = "test-version",
    python_version: str = "3.12.0",
    git_sha: str = "",
    git_dirty: bool = False,
) -> HardwareInfo:
    return HardwareInfo(
        cpu_brand=cpu_brand,
        cpu_count=cpu_count,
        ram_gb=ram_gb,
        os_name=os_name,
        os_version=os_version,
        python_version=python_version,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )


def make_benchmark_result(
    name: str,
    *,
    family: str | None = None,
    n: int = BENCH_TEST_RESULT_N,
    ns: float = BENCH_TEST_RESULT_NS,
    items_per_second: float = 0.0,
    custom_counters: dict[str, float] | None = None,
    tags: tuple[str, ...] = (),
) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        family=family if family is not None else name.split("/", 1)[0],
        iterations=1,
        real_time_ns=ns,
        cpu_time_ns=ns,
        items_per_second=items_per_second,
        custom_counters=dict(custom_counters) if custom_counters else {},
        params={"n": n},
        tags=tags,
    )


def make_benchmark_run(
    results: tuple[BenchmarkResult, ...],
    *,
    run_id: str = BENCH_TEST_RUN_ID,
    timestamp: str = BENCH_TEST_TIMESTAMP,
    tags: tuple[str, ...] = (),
    hardware: HardwareInfo | None = None,
) -> BenchmarkRun:
    return BenchmarkRun(
        run_id=run_id,
        timestamp=timestamp,
        tags=tags,
        results=results,
        hardware=hardware if hardware is not None else make_benchmark_hardware(),
    )


_STUB_FOLD_START = pd.Timestamp("2020-01-01")
_STUB_FOLD_END = pd.Timestamp("2020-12-31")
_STUB_DATA_HASH = "a" * 64


def make_stub_fold_record(
    fold_index: int,
    *,
    sharpe: float,
    equity_curve: tuple[float, ...],
    max_drawdown: float = -0.08,
    total_return: float = 0.05,
    train_start: pd.Timestamp | None = None,
    train_end: pd.Timestamp | None = None,
    test_start: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
) -> FoldRecord:
    """Build a :class:`FoldRecord` with the minimal fields every caller cares about.

    The ``*_start`` / ``*_end`` kwargs default to a fixed 2020 calendar window
    so callers that only care about metric aggregation can stay terse;
    callers that need specific fold windows pass them in explicitly.
    """
    return FoldRecord(
        fold_index=fold_index,
        train_start=train_start if train_start is not None else _STUB_FOLD_START,
        train_end=train_end if train_end is not None else _STUB_FOLD_END,
        test_start=test_start if test_start is not None else _STUB_FOLD_START,
        test_end=test_end if test_end is not None else _STUB_FOLD_END,
        total_return=total_return,
        annualized_return=total_return * 2,
        annualized_volatility=0.15,
        sharpe_ratio=sharpe,
        sortino_ratio=sharpe * 1.05,
        calmar_ratio=sharpe * 0.9,
        max_drawdown=max_drawdown,
        win_rate=0.55,
        trade_count=30,
        equity_curve=equity_curve,
    )


def make_stub_experiment_result(
    name: str,
    *,
    folds: tuple[FoldRecord, ...],
    seed: int = GLOBAL_NUMPY_SEED,
    data_hash: str = _STUB_DATA_HASH,
) -> ExperimentResult:
    """Build an :class:`ExperimentResult` with a minimal valid Manifest.

    Callers control folds fully; the manifest is plausible scaffolding
    (synthetic data hash, 'unknown' git sha). Used by cross-strategy
    comparison tests that need aligned folds across strategies.
    """
    manifest = Manifest(
        experiment_id=f"stub_{name}",
        name=name,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        git_sha="stubsha1",
        seed=seed,
        data_hash=data_hash,
        slippage_scenario=SlippageScenario.NORMAL,
    )
    return ExperimentResult(
        experiment_id=f"stub_{name}",
        folds=folds,
        manifest=manifest,
    )


def make_log_return_equity_curve(
    sharpe: float,
    *,
    n: int,
    seed: int,
    sigma: float = 0.01,
) -> tuple[float, ...]:
    """Equity curve whose per-bar log-returns have approximately the target Sharpe.

    Used by cross-strategy comparison tests: each fold needs a stable,
    seed-deterministic curve whose downstream Sharpe is recognisable to
    ``aggregate_folds`` so ranking + pairwise bootstrap behave predictably.
    """
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(sharpe * sigma, sigma, size=n - 1)
    curve = np.exp(np.concatenate([[0.0], np.cumsum(log_rets)]))
    return tuple(curve.tolist())


def comparison_curve_seed(name: str, fold_index: int) -> int:
    """Stable, process-independent seed for a (strategy, fold) pair.

    Avoids ``hash((name, i))`` whose value is randomised per Python process
    when ``PYTHONHASHSEED`` is not pinned — that randomness would let two
    test invocations see different equity curves and re-flake otherwise.

    Layout: a 16-bit name anchor (positional sum of ``ord(char)``) packed
    into the high half, fold index into the low half. Reserves a 65k-fold
    window per name without collisions.
    """
    name_anchor = sum((i + 1) * ord(c) for i, c in enumerate(name)) & 0xFFFF
    return (name_anchor << 16) | (fold_index & 0xFFFF)


def make_stub_aggregate_stats(
    *,
    sharpe: float,
    n_folds: int = 1,
    max_drawdown_worst: float = -0.1,
    total_return_mean: float = 0.05,
) -> AggregateStats:
    """Build a stub :class:`AggregateStats` from a scalar Sharpe.

    Used by tuner / CLI tests that monkeypatch ``aggregate_folds`` and only
    care about the objective-driving sharpe/sortino/calmar fields — every
    other numeric field mirrors ``sharpe`` so the dict emitted by
    ``to_dict()`` is a well-formed superset regardless of which objective
    the test happens to select. ``n_folds`` defaults to 1 for the tuner
    callers; the consolidator tests pass higher values to exercise
    fold-weighted pooling.
    """
    return AggregateStats(
        n_folds=n_folds,
        sharpe_mean=sharpe,
        sharpe_std=0.0,
        sharpe_ci95_low=sharpe,
        sharpe_ci95_high=sharpe,
        sortino_mean=sharpe,
        sortino_std=0.0,
        sortino_ci95_low=sharpe,
        sortino_ci95_high=sharpe,
        calmar_mean=sharpe,
        calmar_std=0.0,
        calmar_ci95_low=sharpe,
        calmar_ci95_high=sharpe,
        max_drawdown_worst=max_drawdown_worst,
        max_drawdown_mean=max_drawdown_worst,
        total_return_mean=total_return_mean,
        total_return_std=0.0,
        win_rate_mean=0.5,
        trade_count_total=1,
    )
