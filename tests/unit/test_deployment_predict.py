"""
Anti-leakage + idempotency + warmup behaviour for :func:`deployment.predict`.

A stub :class:`~src.data.live_fetcher.LiveBarFetcher` is monkeypatched
in via :func:`src.orchestration.deployment.resolve_fetcher` so the
predict path runs without yfinance / network. The stub returns a
deterministic slice of a fixed synthetic OHLCV frame - long enough to
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
    has_session_gaps,
    next_signal_date,
    predict,
    predict_backfill,
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
def stub_fetcher(bars: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
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
    A bar AT train_end is not strictly after - must raise.
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
    A bar strictly before train_end also fails - defence in depth.
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

    log_path = trained_run / DEPLOYMENTS_SUBDIR / _DEPLOYMENT_ID / DEPLOYMENT_SIGNALS_JSONL
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1


def test_predict_strategy_state_frozen(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    GARCH params + scaler must be byte-identical pre/post-predict.

    Hashes the trained strategy state directory before and after a
    predict call - any in-place mutation of weights / config / metadata
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

    monkeypatch.setattr("src.orchestration.deployment.resolve_fetcher", lambda _: _ShortFetcher())

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
    rebuilt - fit on the model's training window, applied to the live bars -
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
    handed the raw bars unchanged - no spurious training-window fetch.
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

    monkeypatch.setattr("src.orchestration.deployment.resolve_fetcher", lambda _: _EmptyFetcher())

    with pytest.raises(WarmupInsufficientError, match="no bars"):
        predict(
            deployment_id=_DEPLOYMENT_ID,
            store_root=trained_run,
            as_of=pd.Timestamp(bars.index[-1]),
        )


_GAP_EARLY_OFFSET = 10
_GAP_LATE_OFFSET = 40
_SOLO_OFFSET = 50
_TRUNCATED_BAR_COUNT = 10


def test_predict_vendor_truncation_raises_clear_error(
    trained_run: Path, bars: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A vendor frame shorter than ``warmup_bars`` is named as a truncation.

    Distinct from the NaN-warmup case: the message must point at the data
    vendor under-delivering (a transient yfinance hiccup), not tell the user
    to swap to a longer-history ticker.
    """

    _create(trained_run)  # warmup_bars = _WARMUP_BARS (200), >> the truncated count
    last = _TRAIN_END_INDEX + _SOLO_OFFSET

    class _TruncatedFetcher:
        def fetch(
            self, ticker: str, start: datetime, end: datetime, interval: Interval
        ) -> pd.DataFrame:
            del ticker, start, end, interval
            return bars.iloc[last - _TRUNCATED_BAR_COUNT + 1 : last + 1]

    monkeypatch.setattr(
        "src.orchestration.deployment.resolve_fetcher", lambda _: _TruncatedFetcher()
    )

    with pytest.raises(WarmupInsufficientError, match="data vendor returned only"):
        predict(
            deployment_id=_DEPLOYMENT_ID,
            store_root=trained_run,
            as_of=pd.Timestamp(bars.index[last]),
        )


def _predict_at(store: Path, bars: pd.DataFrame, cursor: dict[str, int], offset: int) -> None:
    cursor["last"] = _TRAIN_END_INDEX + offset
    predict(
        deployment_id=_DEPLOYMENT_ID,
        store_root=store,
        as_of=pd.Timestamp(bars.index[cursor["last"]]),
    )


def test_predict_backfill_fills_interior_gap(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    Two sporadic observations leave a hole; backfill fills every session.
    """

    _create(trained_run)
    _predict_at(trained_run, bars, stub_fetcher, _GAP_EARLY_OFFSET)
    _predict_at(trained_run, bars, stub_fetcher, _GAP_LATE_OFFSET)
    assert len(read_signals(trained_run, _DEPLOYMENT_ID)) == 2  # gap present

    stub_fetcher["last"] = _TRAIN_END_INDEX + _GAP_LATE_OFFSET
    span = predict_backfill(
        deployment_id=_DEPLOYMENT_ID,
        store_root=trained_run,
        as_of=pd.Timestamp(bars.index[stub_fetcher["last"]]),
    )

    expected = [
        pd.Timestamp(t)
        for t in bars.index[
            _TRAIN_END_INDEX + _GAP_EARLY_OFFSET : _TRAIN_END_INDEX + _GAP_LATE_OFFSET + 1
        ]
    ]
    on_disk = [row.bar_ts for row in read_signals(trained_run, _DEPLOYMENT_ID)]
    assert on_disk == expected  # gap filled, chronological
    assert [row.bar_ts for row in span] == expected
    assert len(on_disk) == _GAP_LATE_OFFSET - _GAP_EARLY_OFFSET + 1


def test_predict_backfill_no_history_emits_only_latest(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    A deployment with no signals has no live history to fill - records one bar.
    """

    _create(trained_run)
    stub_fetcher["last"] = _TRAIN_END_INDEX + _SOLO_OFFSET
    span = predict_backfill(
        deployment_id=_DEPLOYMENT_ID,
        store_root=trained_run,
        as_of=pd.Timestamp(bars.index[stub_fetcher["last"]]),
    )

    assert len(span) == 1
    assert span[-1].bar_ts == pd.Timestamp(bars.index[_TRAIN_END_INDEX + _SOLO_OFFSET])
    assert len(read_signals(trained_run, _DEPLOYMENT_ID)) == 1


def test_predict_backfill_rewrites_log_in_chronological_order(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    Out-of-order observation leaves a disordered log; backfill re-sorts it.
    """

    _create(trained_run)
    _predict_at(trained_run, bars, stub_fetcher, _GAP_LATE_OFFSET)
    _predict_at(trained_run, bars, stub_fetcher, _GAP_EARLY_OFFSET)
    before = [row.bar_ts for row in read_signals(trained_run, _DEPLOYMENT_ID)]
    assert before == [
        pd.Timestamp(bars.index[_TRAIN_END_INDEX + _GAP_LATE_OFFSET]),
        pd.Timestamp(bars.index[_TRAIN_END_INDEX + _GAP_EARLY_OFFSET]),
    ]  # disordered on disk

    stub_fetcher["last"] = _TRAIN_END_INDEX + _GAP_LATE_OFFSET
    predict_backfill(
        deployment_id=_DEPLOYMENT_ID,
        store_root=trained_run,
        as_of=pd.Timestamp(bars.index[stub_fetcher["last"]]),
    )

    after = [row.bar_ts for row in read_signals(trained_run, _DEPLOYMENT_ID)]
    assert after == sorted(after)


def test_predict_backfill_idempotent_second_run_writes_nothing(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    Re-running backfill over an already-complete span changes nothing.
    """

    _create(trained_run)
    _predict_at(trained_run, bars, stub_fetcher, _GAP_EARLY_OFFSET)
    stub_fetcher["last"] = _TRAIN_END_INDEX + _GAP_LATE_OFFSET
    as_of = pd.Timestamp(bars.index[stub_fetcher["last"]])
    first = predict_backfill(deployment_id=_DEPLOYMENT_ID, store_root=trained_run, as_of=as_of)
    second = predict_backfill(deployment_id=_DEPLOYMENT_ID, store_root=trained_run, as_of=as_of)

    assert [row.to_dict() for row in first] == [row.to_dict() for row in second]
    assert len(read_signals(trained_run, _DEPLOYMENT_ID)) == len(first)


def test_predict_backfill_before_train_end_raises_leakage(
    trained_run: Path, stub_fetcher: dict[str, int], bars: pd.DataFrame
) -> None:
    """
    The same anti-leakage boundary as single-bar predict: last bar must be after.
    """

    _create(trained_run)
    stub_fetcher["last"] = _TRAIN_END_INDEX - 5
    with pytest.raises(LeakageError, match="not strictly after train_end"):
        predict_backfill(
            deployment_id=_DEPLOYMENT_ID,
            store_root=trained_run,
            as_of=pd.Timestamp(bars.index[stub_fetcher["last"]]),
        )


def test_has_session_gaps_detects_missing_sessions() -> None:
    """
    True iff NYSE sessions span more days than are recorded.
    """

    monday = pd.Timestamp("2026-06-01")
    tuesday = pd.Timestamp("2026-06-02")
    wednesday = pd.Timestamp("2026-06-03")
    thursday = pd.Timestamp("2026-06-04")

    assert has_session_gaps([monday, tuesday, wednesday], Interval.DAILY) is False
    assert has_session_gaps([monday, thursday], Interval.DAILY) is True  # Tue + Wed missing
    assert has_session_gaps([monday], Interval.DAILY) is False
    assert has_session_gaps([], Interval.DAILY) is False


def test_has_session_gaps_rejects_non_daily() -> None:
    with pytest.raises(NotImplementedError, match="daily"):
        has_session_gaps([pd.Timestamp("2026-06-01")], Interval.HOUR)


def test_next_signal_date_advances_to_next_session() -> None:
    """
    The signal at bar ``t`` is for the next session: a mid-week close rolls
    one day, a Friday close rolls over the weekend to Monday.
    """

    wednesday = pd.Timestamp("2026-05-27")
    thursday = pd.Timestamp("2026-05-28")
    friday = pd.Timestamp("2026-05-29")
    monday = pd.Timestamp("2026-06-01")

    assert next_signal_date(wednesday, Interval.DAILY) == thursday
    assert next_signal_date(friday, Interval.DAILY) == monday


def test_next_signal_date_skips_exchange_holidays() -> None:
    """
    The NYSE calendar skips holidays a bare weekday roll would land on.

    2026-07-03 is the observed Independence Day holiday (July 4 falls on a
    Saturday), so the session after Thursday 2026-07-02 is Monday 2026-07-06
    - not the holiday Friday a naive next-business-day would pick.
    """

    thursday_before_holiday = pd.Timestamp("2026-07-02")
    monday_after_holiday = pd.Timestamp("2026-07-06")

    assert next_signal_date(thursday_before_holiday, Interval.DAILY) == monday_after_holiday


def test_next_signal_date_normalises_tz_aware_intraday_anchor() -> None:
    """
    A tz-aware ``bar_ts`` carrying a wall-clock time still yields a naive
    next-business-day midnight - the signal date never carries a time.
    """

    friday_evening_utc = pd.Timestamp("2026-05-29T20:30:00", tz="UTC")

    assert next_signal_date(friday_evening_utc, Interval.DAILY) == pd.Timestamp("2026-06-01")


def test_next_signal_date_rejects_non_daily() -> None:
    with pytest.raises(NotImplementedError, match="daily"):
        next_signal_date(pd.Timestamp("2026-05-29"), Interval.HOUR)
