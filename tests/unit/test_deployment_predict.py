"""
Anti-leakage + idempotency + warmup behaviour for :func:`deployment.predict`.

A stub :class:`~src.data.live_fetcher.LiveBarFetcher` is monkeypatched
in via :func:`src.orchestration.deployment.resolve_fetcher` so the
predict path runs without yfinance / network. The stub returns a
deterministic slice of a fixed synthetic OHLCV frame — long enough to
warm GARCH + the Bollinger windows, short enough to keep tests fast.

These tests are the load-bearing anti-leakage proof: they pin the
"last fetched bar must be strictly after train_end" invariant, the
NaN-warmup guard, the strategy-state-frozen contract, and the
idempotent-append contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.core.exceptions import LeakageError, WarmupInsufficientError
from src.core.persistence import (
    DEPLOYMENT_SIGNALS_JSONL,
    DEPLOYMENTS_SUBDIR,
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_STRATEGY_SUBDIR,
    RUNS_SUBDIR,
)
from src.core.types import Interval
from src.orchestration.deployment import (
    create_deployment,
    predict,
    read_signals,
)
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from tests.conftest import make_synthetic_ohlcv_df

_TOTAL_BARS = 600
_TRAIN_END_INDEX = 400
_BOLLINGER_WINDOW = 20
_BOLLINGER_TREND_WINDOW = 50
_GARCH_P_MAX = 1
_GARCH_Q_MAX = 1
_WARMUP_BARS = 200
_INSUFFICIENT_WARMUP_BARS = 5
_DEPLOYMENT_ID = "predict-test-deployment"
_RUN_ID = "predict-test-run"
_TICKER = "SPY"


class _StubFetcher:
    """
    Return a slice of a pre-built OHLCV frame, ignoring the date range.

    The predict path passes ``start = as_of - warmup_window``; we don't
    care about the exact window math here, only that the fetcher's
    output ends at the right bar. The test pins ``as_of`` to a specific
    index in the master frame and the stub returns bars up to and
    including that index.
    """

    def __init__(self, bars: pd.DataFrame, last_index: int) -> None:
        self._bars = bars
        self._last_index = last_index

    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval,
    ) -> pd.DataFrame:
        del ticker, start, end, interval
        return self._bars.iloc[: self._last_index + 1]


@pytest.fixture
def bars() -> pd.DataFrame:
    """
    A fixed 600-row OHLCV frame seeded by the synthetic generator.
    """

    return make_synthetic_ohlcv_df(n_rows=_TOTAL_BARS)


@pytest.fixture
def trained_run(tmp_path: Path, bars: pd.DataFrame) -> Path:
    """
    Train AdaptiveBollinger on the first ``_TRAIN_END_INDEX`` rows and persist.

    The remaining rows form a synthetic "live" tape the stub fetcher
    serves from.
    """

    from src.core.config import load_experiment_config, write_frozen_yaml

    store = tmp_path / "store"
    run_dir = store / RUNS_SUBDIR / _RUN_ID
    run_dir.mkdir(parents=True)
    cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
    write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, cfg)

    train_df = bars.iloc[:_TRAIN_END_INDEX]
    strategy = AdaptiveBollingerStrategy(
        window=_BOLLINGER_WINDOW,
        trend_window=_BOLLINGER_TREND_WINDOW,
        garch_p_max=_GARCH_P_MAX,
        garch_q_max=_GARCH_Q_MAX,
    )
    strategy.train(train_df)
    strategy.save(run_dir / EXPERIMENT_STRATEGY_SUBDIR)
    return store


@pytest.fixture
def stub_fetcher(
    bars: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, int]]:
    """
    Patch ``resolve_fetcher`` with a stub whose last bar is controllable.

    Tests mutate ``cursor["last"]`` before calling ``predict`` so the
    fetched window ends on a specific row in the master frame.
    """

    cursor = {"last": _TRAIN_END_INDEX + 10}

    def _resolve(_: Interval) -> _StubFetcher:
        return _StubFetcher(bars, cursor["last"])

    monkeypatch.setattr("src.orchestration.deployment.resolve_fetcher", _resolve)
    yield cursor


def _create(store: Path, *, warmup_bars: int = _WARMUP_BARS) -> None:
    create_deployment(
        source_kind="run",
        source_id=_RUN_ID,
        store_root=store,
        deployment_id=_DEPLOYMENT_ID,
        warmup_bars=warmup_bars,
    )


def test_predict_succeeds_strictly_after_train_end(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    The canonical happy path: ``as_of`` lands on a bar strictly after train_end.
    """

    _create(trained_run)
    stub_fetcher["last"] = _TRAIN_END_INDEX + 50
    as_of = pd.Timestamp(bars.index[stub_fetcher["last"]])

    row = predict(deployment_id=_DEPLOYMENT_ID, store_root=trained_run, as_of=as_of)

    assert row.signal in {-1.0, 0.0, 1.0}
    assert row.bar_ts == pd.Timestamp(bars.index[stub_fetcher["last"]])
    assert row.source_run_id == _RUN_ID
    assert row.warmup_bars_used == _WARMUP_BARS
    assert row.warmup_fingerprint  # non-empty hash


def test_predict_at_train_end_raises_leakage(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    A bar AT train_end is not strictly after — must raise.
    """

    _create(trained_run)
    stub_fetcher["last"] = _TRAIN_END_INDEX - 1
    as_of = pd.Timestamp(bars.index[stub_fetcher["last"]])

    with pytest.raises(LeakageError, match="not strictly after train_end"):
        predict(deployment_id=_DEPLOYMENT_ID, store_root=trained_run, as_of=as_of)


def test_predict_before_train_end_raises_leakage(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    A bar strictly before train_end also fails — defence in depth.
    """

    _create(trained_run)
    stub_fetcher["last"] = _TRAIN_END_INDEX - 50
    as_of = pd.Timestamp(bars.index[stub_fetcher["last"]])

    with pytest.raises(LeakageError, match="not strictly after train_end"):
        predict(deployment_id=_DEPLOYMENT_ID, store_root=trained_run, as_of=as_of)


def test_predict_idempotent_on_same_bar(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    Second predict on the same target bar returns the existing row.
    """

    _create(trained_run)
    stub_fetcher["last"] = _TRAIN_END_INDEX + 50
    as_of = pd.Timestamp(bars.index[stub_fetcher["last"]])

    first = predict(deployment_id=_DEPLOYMENT_ID, store_root=trained_run, as_of=as_of)
    second = predict(deployment_id=_DEPLOYMENT_ID, store_root=trained_run, as_of=as_of)

    assert first.to_dict() == second.to_dict()

    log_path = (
        trained_run / DEPLOYMENTS_SUBDIR / _DEPLOYMENT_ID / DEPLOYMENT_SIGNALS_JSONL
    )
    lines = [
        line for line in log_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(lines) == 1


def test_predict_strategy_state_frozen(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    GARCH params + scaler must be byte-identical pre/post-predict.

    Hashes the trained strategy state directory before and after a
    predict call — any in-place mutation of weights / config / metadata
    would change the bytes.
    """

    import hashlib

    _create(trained_run)
    stub_fetcher["last"] = _TRAIN_END_INDEX + 50
    as_of = pd.Timestamp(bars.index[stub_fetcher["last"]])

    state_dir = trained_run / RUNS_SUBDIR / _RUN_ID / EXPERIMENT_STRATEGY_SUBDIR

    def _hash_state() -> str:
        h = hashlib.sha256()
        for p in sorted(state_dir.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(state_dir)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    before = _hash_state()
    predict(deployment_id=_DEPLOYMENT_ID, store_root=trained_run, as_of=as_of)
    after = _hash_state()
    assert before == after


def test_predict_warmup_insufficient_raises(
    trained_run: Path, bars: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A warmup window shorter than the strategy's longest lookback ends in NaN.

    The plan requires this surface loudly rather than silently writing
    a NaN row to the cache.
    """

    _create(trained_run, warmup_bars=_INSUFFICIENT_WARMUP_BARS)
    last = _TRAIN_END_INDEX + 5

    class _ShortFetcher:
        def fetch(
            self,
            ticker: str,
            start: datetime,
            end: datetime,
            interval: Interval,
        ) -> pd.DataFrame:
            del ticker, start, end, interval
            return bars.iloc[last - _INSUFFICIENT_WARMUP_BARS : last + 1]

    monkeypatch.setattr(
        "src.orchestration.deployment.resolve_fetcher", lambda _: _ShortFetcher()
    )

    as_of = pd.Timestamp(bars.index[last])
    with pytest.raises(WarmupInsufficientError, match="NaN"):
        predict(deployment_id=_DEPLOYMENT_ID, store_root=trained_run, as_of=as_of)


def test_predict_appends_one_row_per_unique_bar(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    Distinct ``as_of`` values land distinct rows; same value dedupes.
    """

    _create(trained_run)

    stub_fetcher["last"] = _TRAIN_END_INDEX + 30
    predict(
        deployment_id=_DEPLOYMENT_ID,
        store_root=trained_run,
        as_of=pd.Timestamp(bars.index[stub_fetcher["last"]]),
    )
    stub_fetcher["last"] = _TRAIN_END_INDEX + 60
    predict(
        deployment_id=_DEPLOYMENT_ID,
        store_root=trained_run,
        as_of=pd.Timestamp(bars.index[stub_fetcher["last"]]),
    )
    stub_fetcher["last"] = _TRAIN_END_INDEX + 30
    predict(
        deployment_id=_DEPLOYMENT_ID,
        store_root=trained_run,
        as_of=pd.Timestamp(bars.index[stub_fetcher["last"]]),
    )

    rows = read_signals(trained_run, _DEPLOYMENT_ID)
    bar_ts_set = {row.bar_ts for row in rows}
    assert len(bar_ts_set) == 2
    assert len(rows) == 2


def test_featurize_rebuilds_external_pipeline_columns(bars: pd.DataFrame) -> None:
    """
    A strategy with an external feature pipeline gets its feature columns
    rebuilt — fit on the model's training window, applied to the live bars —
    before ``generate_signals`` sees them.
    """

    from src.core.config import ComponentConfig
    from src.core.temporal import TrainingMetadata
    from src.orchestration.deployment import _featurize_for_signals

    train_slice = bars.iloc[:_TRAIN_END_INDEX]
    live_slice = bars.iloc[_TRAIN_END_INDEX - 100 :]

    class _TrainWindowFetcher:
        def fetch(
            self, ticker: str, start: datetime, end: datetime, interval: Interval
        ) -> pd.DataFrame:
            del ticker, start, end, interval
            return train_slice

    metadata = TrainingMetadata.from_fit(
        train_slice, Interval.DAILY, ("return_1d", "vol_20", "rsi_14", "macd")
    )
    out = _featurize_for_signals(
        live_slice,
        features_cfg=ComponentConfig(name="standard", params={"keep_ohlc": True}),
        fetcher=_TrainWindowFetcher(),
        ticker=_TICKER,
        metadata=metadata,
    )

    assert "return_1d" in out.columns
    assert "macd" in out.columns
    assert "close" in out.columns  # keep_ohlc preserved
    assert len(out.columns) > len(live_slice.columns)
    assert len(out) == len(live_slice)


def test_featurize_passthrough_when_no_pipeline(bars: pd.DataFrame) -> None:
    """
    Strategies that self-compute from OHLCV (``features_cfg is None``) are
    handed the raw bars unchanged — no spurious training-window fetch.
    """

    from src.core.temporal import TrainingMetadata
    from src.orchestration.deployment import _featurize_for_signals

    metadata = TrainingMetadata.from_fit(bars.iloc[:_TRAIN_END_INDEX], Interval.DAILY, ())
    out = _featurize_for_signals(
        bars,
        features_cfg=None,
        fetcher=_StubFetcher(bars, _TRAIN_END_INDEX),
        ticker=_TICKER,
        metadata=metadata,
    )

    assert out is bars


def test_predict_empty_fetch_raises(
    trained_run: Path, bars: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A fetcher that returns zero bars must not silently succeed.
    """

    _create(trained_run)

    class _EmptyFetcher:
        def fetch(
            self,
            ticker: str,
            start: datetime,
            end: datetime,
            interval: Interval,
        ) -> pd.DataFrame:
            del ticker, start, end, interval
            return bars.iloc[0:0]

    monkeypatch.setattr(
        "src.orchestration.deployment.resolve_fetcher", lambda _: _EmptyFetcher()
    )

    with pytest.raises(WarmupInsufficientError, match="no bars"):
        predict(
            deployment_id=_DEPLOYMENT_ID,
            store_root=trained_run,
            as_of=pd.Timestamp(bars.index[-1]),
        )
