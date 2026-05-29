"""
CRUD + predict-if-stale for live-inference deployments.

See :mod:`src.orchestration.deployment` for the on-disk format and the
anti-leakage guard inside ``predict()``. This module owns the SQLite row
that links a user to a deployment plus the in-memory caches that keep
the threadpool-served predict path cheap on a list-page fan-out.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd

from src.core.types import Interval
from src.data.live_fetcher import resolve_fetcher
from src.orchestration.deployment import (
    SignalRow,
    _to_naive,
    next_signal_date,
    read_signals,
    resolve_deployment_dir,
    resolve_strategy_state_path,
)
from src.orchestration.deployment import (
    create_deployment as framework_create_deployment,
)
from src.orchestration.deployment import (
    predict as framework_predict,
)
from src.orchestration.holdout_eval import SourceKind
from src.orchestration.run_loader import (
    load_experiment_config_from_run,
    load_strategy_from_run_dir,
)
from src.strategies.interface import IStrategy
from webapp.backend.app.core.types import Role
from webapp.backend.app.schemas.deployments import (
    DeploymentDetail,
    DeploymentSummary,
    PredictIfStaleResponse,
    SignalRowOut,
)
from webapp.backend.app.schemas.users import UserPublic
from webapp.backend.app.services.user_service import get_user

__all__ = [
    "DeploymentAccessDeniedError",
    "DeploymentNotFoundError",
    "DeploymentSourceInvalidError",
    "create_deployment",
    "delete_deployment",
    "get_deployment",
    "list_deployments",
    "predict_if_stale",
    "read_signal_log",
    "rename_deployment",
]


_OWNERLESS_USERNAME = "system"


class DeploymentNotFoundError(LookupError):
    """No deployment with this id exists in the DB."""


class DeploymentAccessDeniedError(LookupError):
    """
    Caller asked for someone else's deployment.

    Router maps this to 404 (not 403) so the response doesn't disclose
    that the deployment exists.
    """


class DeploymentSourceInvalidError(ValueError):
    """
    Source run / HPO study cannot be deployed.

    Today's predict path supports single-asset daily-cadence strategies
    only. Pairs and non-daily sources are rejected at create time so the
    user sees the limitation before submitting an unusable predict.
    """


_STRATEGY_CACHE_MAX = 16
_strategy_cache: OrderedDict[str, IStrategy] = OrderedDict()
_strategy_cache_lock = threading.Lock()


def _load_strategy_cached(run_dir: Path) -> IStrategy:
    """
    LRU-cached wrapper around :func:`load_strategy_from_run_dir`.

    Strategy state on disk is immutable once saved (framework contract:
    ``save()`` writes to a fresh directory and never overwrites), so
    keying by ``str(run_dir)`` alone is sound. The load itself runs
    outside the lock so a slow disk read doesn't block other keys.
    """

    key = str(run_dir)
    with _strategy_cache_lock:
        cached = _strategy_cache.get(key)
        if cached is not None:
            _strategy_cache.move_to_end(key)
            return cached
    strategy = load_strategy_from_run_dir(run_dir)
    with _strategy_cache_lock:
        _strategy_cache[key] = strategy
        _strategy_cache.move_to_end(key)
        while len(_strategy_cache) > _STRATEGY_CACHE_MAX:
            _strategy_cache.popitem(last=False)
    return strategy


def _clear_strategy_cache() -> None:
    with _strategy_cache_lock:
        _strategy_cache.clear()


# Serializes predict-if-stale per deployment. Two concurrent requests (a
# double-fired mount effect, a double-click, two tabs) would otherwise both
# miss the cache and append a duplicate row for the same bar; the second
# caller blocks here, then re-reads the freshly written signal and recalls it.
_predict_locks: dict[str, threading.Lock] = {}
_predict_locks_guard = threading.Lock()


def _predict_lock(deployment_id: str) -> threading.Lock:
    with _predict_locks_guard:
        lock = _predict_locks.get(deployment_id)
        if lock is None:
            lock = threading.Lock()
            _predict_locks[deployment_id] = lock
        return lock


_BAR_TS_CACHE_TTL_SECONDS = 300.0


@dataclass(frozen=True)
class _BarTsCacheKey:
    ticker: str
    interval: Interval


_bar_ts_cache: dict[_BarTsCacheKey, tuple[pd.Timestamp, float]] = {}
_bar_ts_cache_lock = threading.Lock()


def _probe_latest_bar_ts(ticker: str, interval: Interval) -> pd.Timestamp:
    """
    Ask the live fetcher for the most recent available bar's timestamp.

    Exposed as a module-level function so tests can monkeypatch it
    without faking yfinance end-to-end.
    """

    fetcher = resolve_fetcher(interval)
    now = pd.Timestamp.now(tz="UTC")
    window_days = 14
    start = (now - pd.Timedelta(days=window_days)).to_pydatetime()
    end = now.to_pydatetime()
    bars = fetcher.fetch(ticker, start, end, interval)
    if bars.empty:
        raise DeploymentSourceInvalidError(
            f"vendor returned no bars for {ticker!r} over the last "
            f"{window_days} days; the ticker may be delisted or unknown."
        )
    return pd.Timestamp(bars.index[-1])


def _latest_available_bar_ts_cached(ticker: str, interval: Interval) -> pd.Timestamp:
    """
    Return the latest available bar timestamp for ``(ticker, interval)``.

    5-minute TTL absorbs a list-page refresh that fans out N parallel
    requests for the same ticker without N parallel yfinance round-trips.
    """

    key = _BarTsCacheKey(ticker, interval)
    now = time.time()
    with _bar_ts_cache_lock:
        cached = _bar_ts_cache.get(key)
        if cached is not None and cached[1] > now:
            return cached[0]
    bar_ts = _probe_latest_bar_ts(ticker, interval)
    with _bar_ts_cache_lock:
        _bar_ts_cache[key] = (bar_ts, now + _BAR_TS_CACHE_TTL_SECONDS)
    return bar_ts


def _clear_bar_ts_cache() -> None:
    with _bar_ts_cache_lock:
        _bar_ts_cache.clear()


def _row_to_summary(row: sqlite3.Row, owner_username: str) -> DeploymentSummary:
    return DeploymentSummary(
        id=row["id"],
        name=row["name"],
        source_kind=cast(SourceKind, row["source_kind"]),
        source_id=row["source_id"],
        ticker=row["ticker"],
        strategy_name=row["strategy_name"],
        interval=Interval(row["interval"]),
        train_end=datetime.fromisoformat(row["train_end"]),
        warmup_bars=row["warmup_bars"],
        created_at=datetime.fromisoformat(row["created_at"]),
        owner_username=owner_username,
    )


def _signal_to_out(row: SignalRow, interval: Interval) -> SignalRowOut:
    return SignalRowOut(
        submitted_at=row.submitted_at.to_pydatetime(),
        bar_ts=row.bar_ts.to_pydatetime(),
        signal_date=next_signal_date(row.bar_ts, interval).to_pydatetime(),
        signal=row.signal,
        warmup_fingerprint=row.warmup_fingerprint,
        source_run_id=row.source_run_id,
        warmup_bars_used=row.warmup_bars_used,
    )


def _fetch_row(conn: sqlite3.Connection, deployment_id: str) -> sqlite3.Row | None:
    return conn.execute(  # type: ignore[no-any-return]
        "SELECT * FROM deployments WHERE id = ?", (deployment_id,)
    ).fetchone()


def _resolve_username(conn: sqlite3.Connection, user_id: int) -> str:
    user = get_user(conn, user_id)
    return user.username if user is not None else _OWNERLESS_USERNAME


def _enforce_access(row: sqlite3.Row, user: UserPublic) -> None:
    if user.role is Role.ADMIN:
        return
    if row["user_id"] == user.id:
        return
    raise DeploymentAccessDeniedError(str(row["id"]))


def _last_signal(store_root: Path, deployment_id: str) -> SignalRow | None:
    signals = read_signals(store_root, deployment_id)
    return signals[-1] if signals else None


def _validate_source_for_predict(
    source_kind: SourceKind, source_id: str, store_root: Path
) -> tuple[str, str, Interval, datetime]:
    """
    Resolve the source's manifest + strategy and validate it is predictable.

    Returns ``(ticker, strategy_name, interval, train_end)`` —
    denormalised columns persisted on the deployment row at create time.
    Surfaces the framework's pair / non-daily limitation at create time
    so the user finds out before submitting an unusable predict.
    """

    state_path = resolve_strategy_state_path(source_kind, source_id, store_root)
    run_dir = state_path.parent
    cfg = load_experiment_config_from_run(run_dir)
    if len(cfg.data.tickers) != 1:
        raise DeploymentSourceInvalidError(
            f"source {source_kind}:{source_id} trains a "
            f"{len(cfg.data.tickers)}-ticker strategy; the live-inference path "
            f"only supports single-asset strategies today. Pick a single-ticker "
            f"source or wait for the pairs / multi-feature live-fetch impl."
        )
    if cfg.data.interval is not Interval.DAILY:
        raise DeploymentSourceInvalidError(
            f"source {source_kind}:{source_id} trains on {cfg.data.interval.value} "
            f"bars; the live-inference path only supports the daily cadence "
            f"today. Pick a daily source or wait for the intraday fetcher impl."
        )
    strategy = _load_strategy_cached(run_dir)
    metadata = strategy.training_metadata
    if metadata is None:
        raise DeploymentSourceInvalidError(
            f"source {source_kind}:{source_id} has no training_metadata on "
            f"its persisted strategy state; the source may be corrupt or "
            f"was trained by an older framework version."
        )
    return (
        cfg.data.tickers[0],
        cfg.strategy.name,
        cfg.data.interval,
        metadata.train_end.to_pydatetime(),
    )


def create_deployment(
    conn: sqlite3.Connection,
    *,
    store_root: Path,
    user: UserPublic,
    source_kind: SourceKind,
    source_id: str,
    name: str | None,
    warmup_bars: int | None,
) -> DeploymentDetail:
    """
    Materialise on-disk artifacts + insert the DB row.

    Disk write happens before the DB INSERT so a downstream INSERT
    failure leaves the disk artifacts orphaned (cheap to clean) rather
    than orphaning the DB row at a nonexistent dir.
    """

    ticker, strategy_name, interval, train_end = _validate_source_for_predict(
        source_kind, source_id, store_root
    )
    deployment_id = uuid.uuid4().hex
    deployment = framework_create_deployment(
        source_kind=source_kind,
        source_id=source_id,
        store_root=store_root,
        name=name,
        warmup_bars=warmup_bars,
        deployment_id=deployment_id,
    )
    conn.execute(
        """
        INSERT INTO deployments (
            id, user_id, name, source_kind, source_id,
            ticker, strategy_name, interval, train_end,
            warmup_bars, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            deployment.deployment_id,
            user.id,
            deployment.name,
            deployment.source_kind,
            deployment.source_id,
            ticker,
            strategy_name,
            interval.value,
            train_end.isoformat(),
            deployment.warmup_bars,
            deployment.created_at.isoformat(),
        ),
    )
    conn.commit()
    row = _fetch_row(conn, deployment.deployment_id)
    assert row is not None
    return DeploymentDetail(
        **_row_to_summary(row, _resolve_username(conn, user.id)).model_dump(),
        latest_signal=None,
    )


def list_deployments(
    conn: sqlite3.Connection,
    *,
    user: UserPublic,
    all_users: bool,
) -> list[DeploymentSummary]:
    """
    Return every deployment ``user`` may see, newest first.
    """

    if user.role is Role.ADMIN and all_users:
        rows = conn.execute(
            "SELECT d.*, u.username AS owner_username FROM deployments d "
            "LEFT JOIN users u ON u.id = d.user_id "
            "ORDER BY d.created_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT d.*, u.username AS owner_username FROM deployments d "
            "LEFT JOIN users u ON u.id = d.user_id "
            "WHERE d.user_id = ? "
            "ORDER BY d.created_at DESC",
            (user.id,),
        ).fetchall()
    return [_row_to_summary(row, row["owner_username"] or _OWNERLESS_USERNAME) for row in rows]


def get_deployment(
    conn: sqlite3.Connection,
    *,
    store_root: Path,
    user: UserPublic,
    deployment_id: str,
) -> DeploymentDetail:
    row = _fetch_row(conn, deployment_id)
    if row is None:
        raise DeploymentNotFoundError(deployment_id)
    _enforce_access(row, user)
    last = _last_signal(store_root, deployment_id)
    summary = _row_to_summary(row, _resolve_username(conn, row["user_id"]))
    return DeploymentDetail(
        **summary.model_dump(),
        latest_signal=_signal_to_out(last, summary.interval) if last is not None else None,
    )


def rename_deployment(
    conn: sqlite3.Connection,
    *,
    store_root: Path,
    user: UserPublic,
    deployment_id: str,
    new_name: str,
) -> DeploymentDetail:
    """
    Update the deployment's display name.

    The on-disk manifest's ``name`` is NOT rewritten — the DB row is the
    canonical UI label; the manifest records the auto-generated identity
    at creation time and stays stable for audit reproducibility.
    """

    row = _fetch_row(conn, deployment_id)
    if row is None:
        raise DeploymentNotFoundError(deployment_id)
    _enforce_access(row, user)
    conn.execute("UPDATE deployments SET name = ? WHERE id = ?", (new_name, deployment_id))
    conn.commit()
    return get_deployment(conn, store_root=store_root, user=user, deployment_id=deployment_id)


def delete_deployment(
    conn: sqlite3.Connection,
    *,
    store_root: Path,
    user: UserPublic,
    deployment_id: str,
) -> None:
    row = _fetch_row(conn, deployment_id)
    if row is None:
        raise DeploymentNotFoundError(deployment_id)
    _enforce_access(row, user)
    dep_dir = resolve_deployment_dir(store_root, deployment_id)
    # disk teardown before DB delete: a failed rmtree leaves a retryable orphan row
    if dep_dir.is_dir():
        shutil.rmtree(dep_dir)
    conn.execute("DELETE FROM deployments WHERE id = ?", (deployment_id,))
    conn.commit()


def read_signal_log(
    conn: sqlite3.Connection,
    *,
    store_root: Path,
    user: UserPublic,
    deployment_id: str,
    limit: int | None,
) -> list[SignalRowOut]:
    row = _fetch_row(conn, deployment_id)
    if row is None:
        raise DeploymentNotFoundError(deployment_id)
    _enforce_access(row, user)
    interval = Interval(row["interval"])
    signals = read_signals(store_root, deployment_id)
    tail = signals if limit is None else signals[-limit:]
    return [_signal_to_out(s, interval) for s in tail]


def predict_if_stale(
    conn: sqlite3.Connection,
    *,
    store_root: Path,
    user: UserPublic,
    deployment_id: str,
) -> PredictIfStaleResponse:
    """
    Return today's signal — recall the cached row when fresh, otherwise predict.
    """

    row = _fetch_row(conn, deployment_id)
    if row is None:
        raise DeploymentNotFoundError(deployment_id)
    _enforce_access(row, user)

    ticker = row["ticker"]
    interval = Interval(row["interval"])

    with _predict_lock(deployment_id):
        latest_available = _to_naive(_latest_available_bar_ts_cached(ticker, interval))
        cached = _last_signal(store_root, deployment_id)
        if cached is not None and _to_naive(cached.bar_ts) >= latest_available:
            return PredictIfStaleResponse(stale=False, signal=_signal_to_out(cached, interval))

        new_signal = framework_predict(
            deployment_id=deployment_id,
            store_root=store_root,
            as_of=None,
            strategy_loader=_load_strategy_cached,
        )
        is_stale = cached is None or _to_naive(cached.bar_ts) != _to_naive(new_signal.bar_ts)
        return PredictIfStaleResponse(stale=is_stale, signal=_signal_to_out(new_signal, interval))


def _clear_caches_for_tests() -> None:
    _clear_strategy_cache()
    _clear_bar_ts_cache()
    with _predict_locks_guard:
        _predict_locks.clear()
