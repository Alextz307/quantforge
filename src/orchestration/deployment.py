"""
Live deployment of a previously trained strategy.

A deployment is the smallest persistent primitive that bridges the
framework's backtest world (frozen, validated strategies on disk) and
practical decision support (today's signal for a given ticker). It is
*only* a pointer to a trained run plus an accumulating signal log —
no model state of its own, no refit clock. Refreshing a stale model is
out of scope here: the user trains a fresher run via the existing
experiment flow and points a new deployment at it.

On-disk layout per deployment::

    <store_root>/deployments/<deployment_id>/
        manifest.json            # typed round-trippable Deployment
        signals.jsonl            # append-only signal log (one row per bar)

Anti-leakage contract — read this before changing :func:`predict`
-----------------------------------------------------------------
The single invariant ``predict`` enforces is

    bars.index[-1] > training_metadata.train_end

i.e. the bar whose signal we act on is strictly after the last bar the
model was fit on. The warmup window is allowed to *overlap* the training
period (the model is frozen on disk, the bars are public market data —
no leakage vector exists). This is strictly weaker than the
walk-forward's ``validate_no_overlap`` check, which guards the
test-set OOS contract — irrelevant in a live setting where the model is
already validated.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from src.core import json_io
from src.core.exceptions import LeakageError, WarmupInsufficientError
from src.core.logging import get_logger
from src.core.persistence import (
    DEPLOYMENT_MANIFEST_JSON,
    DEPLOYMENT_SIGNALS_JSONL,
    DEPLOYMENTS_SUBDIR,
    EXPERIMENT_STRATEGY_SUBDIR,
    HPO_SUBDIR,
    HPO_TRIALS_RUNS_SUBDIR,
)
from src.core.types import Interval
from src.data.fingerprint import fingerprint_bars
from src.data.live_fetcher import resolve_fetcher
from src.optimization.checkpointing import TRIAL_ARTIFACTS_SUBDIR
from src.optimization.tuner import STUDY_DB_FILENAME, USER_ATTR_EXPERIMENT_ID, storage_url_for
from src.orchestration.holdout_eval import SourceKind
from src.orchestration.run_loader import (
    load_experiment_config_from_run,
    load_strategy_from_run_dir,
    resolve_run_dir,
)
from src.strategies.interface import IStrategy

_logger = get_logger(__name__)

_AUTO_DERIVED_MIN_WARMUP_BARS = 50


def recommend_warmup_bars(strategy: IStrategy) -> int:
    """
    Compute the smallest warmup window that still produces a stable signal.

    Composes the strategy's own ``required_warmup_bars`` (the indicator
    lookback floor — below this every signal is NaN at the last position)
    with its ``convergence_margin_bars`` (extra rows that let GARCH/ARMA
    leaves converge out of the fitted backcast). Padded by an absolute
    floor so a tiny-window strategy still ends up with enough bars to
    amortise the recursion's fitted backcast and absorb a long holiday
    cluster in the vendor's calendar.
    """

    return max(
        strategy.required_warmup_bars + strategy.convergence_margin_bars,
        _AUTO_DERIVED_MIN_WARMUP_BARS,
    )


@dataclass(frozen=True)
class Deployment:
    """
    Provenance + configuration for one live deployment.

    Immutable by construction — the source run id never changes, and the
    auto-generated ``name`` is user-editable by writing a new manifest
    with a different ``name`` (no in-place mutation). ``deployment_id``
    is opaque; ``name`` is the user-facing label.
    """

    deployment_id: str
    name: str
    source_kind: SourceKind
    source_id: str
    warmup_bars: int
    created_at: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            "deployment_id": self.deployment_id,
            "name": self.name,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "warmup_bars": self.warmup_bars,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Deployment:
        raw_kind = json_io.get_str(d, "source_kind")
        if raw_kind not in ("run", "hpo"):
            raise ValueError(
                f"deployment manifest has invalid source_kind={raw_kind!r}; "
                f"expected 'run' or 'hpo'."
            )
        return cls(
            deployment_id=json_io.get_str(d, "deployment_id"),
            name=json_io.get_str(d, "name"),
            source_kind=cast(SourceKind, raw_kind),
            source_id=json_io.get_str(d, "source_id"),
            warmup_bars=json_io.get_int(d, "warmup_bars"),
            created_at=json_io.get_timestamp(d, "created_at"),
        )


@dataclass(frozen=True)
class SignalRow:
    """
    One row appended to ``signals.jsonl`` per successful predict.

    ``submitted_at`` is the wall-clock instant the predict ran;
    ``bar_ts`` is the timestamp of the bar the signal is *for*. These
    are kept distinct so daily / hourly / scheduled cadences stay
    distinguishable without schema breaks.
    """

    submitted_at: pd.Timestamp
    bar_ts: pd.Timestamp
    signal: float
    warmup_fingerprint: str
    source_run_id: str
    warmup_bars_used: int

    def to_dict(self) -> dict[str, object]:
        return {
            "submitted_at": self.submitted_at.isoformat(),
            "bar_ts": self.bar_ts.isoformat(),
            "signal": self.signal,
            "warmup_fingerprint": self.warmup_fingerprint,
            "source_run_id": self.source_run_id,
            "warmup_bars_used": self.warmup_bars_used,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> SignalRow:
        return cls(
            submitted_at=json_io.get_timestamp(d, "submitted_at"),
            bar_ts=json_io.get_timestamp(d, "bar_ts"),
            signal=json_io.get_float(d, "signal"),
            warmup_fingerprint=json_io.get_str(d, "warmup_fingerprint"),
            source_run_id=json_io.get_str(d, "source_run_id"),
            warmup_bars_used=json_io.get_int(d, "warmup_bars_used"),
        )


def resolve_deployment_dir(store_root: Path, deployment_id: str) -> Path:
    """
    Resolve ``store_root / deployments / <deployment_id>``.
    """

    return store_root / DEPLOYMENTS_SUBDIR / deployment_id


def resolve_strategy_state_path(
    source_kind: SourceKind, source_id: str, store_root: Path
) -> Path:
    """
    Return the on-disk ``strategy_state/`` directory for a source.

    Single source of truth for "where does the saved strategy live"
    — the deployment layer never branches on ``source_kind`` outside
    this helper. Adding a new source kind is one new branch here and
    zero changes in the predict path.

    For ``hpo`` sources, the path points at the *best trial's* run dir,
    discovered via the Optuna study's ``best_trial.user_attrs[experiment_id]``
    user-attr.
    """

    if source_kind == "run":
        path = resolve_run_dir(store_root, source_id) / EXPERIMENT_STRATEGY_SUBDIR
        if not path.is_dir():
            raise FileNotFoundError(
                f"strategy state not found for run {source_id!r}: {path} does "
                f"not exist. Re-run the source experiment or pass a different "
                f"source_id."
            )
        return path

    if source_kind == "hpo":
        return _resolve_hpo_strategy_state_path(source_id, store_root)

    raise ValueError(
        f"unknown source_kind={source_kind!r}; expected 'run' or 'hpo'."
    )


def _resolve_hpo_strategy_state_path(study_name: str, store_root: Path) -> Path:
    """
    Walk an HPO study's artifacts to the best trial's ``strategy_state/``.

    Uses ``optuna.load_study`` to get the direction-aware best trial; the
    best trial's experiment id is recorded as a user-attr by the tuner
    callback. A study with no completed trials surfaces a pointed
    :class:`FileNotFoundError` so the caller can route the user to "wait
    for the study to finish" rather than a generic Optuna error.
    """

    import optuna

    study_dir = store_root / HPO_SUBDIR / study_name
    db_path = study_dir / STUDY_DB_FILENAME
    if not db_path.is_file():
        raise FileNotFoundError(
            f"HPO study {study_name!r} has no Optuna DB at {db_path}; the study "
            f"may not exist or may not have started. Fix by pointing at a "
            f"finished study under {store_root / HPO_SUBDIR}."
        )
    study = optuna.load_study(study_name=study_name, storage=storage_url_for(study_dir))
    try:
        best = study.best_trial
    except ValueError as exc:
        raise FileNotFoundError(
            f"HPO study {study_name!r} has no completed trials; no best trial "
            f"to deploy. Wait for the study to complete at least one trial."
        ) from exc
    experiment_id = best.user_attrs.get(USER_ATTR_EXPERIMENT_ID)
    if not isinstance(experiment_id, str):
        raise FileNotFoundError(
            f"HPO study {study_name!r} best trial #{best.number} is missing the "
            f"{USER_ATTR_EXPERIMENT_ID!r} user-attr; the trial may have been "
            f"recorded by an older tuner version. Re-run the study."
        )
    trial_run_dir = (
        study_dir / TRIAL_ARTIFACTS_SUBDIR / HPO_TRIALS_RUNS_SUBDIR / experiment_id
    )
    state_dir = trial_run_dir / EXPERIMENT_STRATEGY_SUBDIR
    if not state_dir.is_dir():
        raise FileNotFoundError(
            f"HPO best-trial strategy state not found at {state_dir}; the "
            f"trial artifacts may have been cleaned. Re-run the study or "
            f"choose a different source."
        )
    return state_dir


def _auto_generate_name(
    source_kind: SourceKind,
    source_id: str,
    source_run_dir: Path,
    strategy: IStrategy,
) -> str:
    """
    Build the default display name from the source's config + metadata.

    ``"<ticker>-<strategy>-<train_end>"`` for run sources;
    ``"<ticker>-<strategy>-HPO-<study_name>"`` for HPO sources (no train_end
    because each trial has its own boundary; the study name carries the
    full identity).
    """

    cfg = load_experiment_config_from_run(source_run_dir)
    ticker = "/".join(cfg.data.tickers)
    if source_kind == "run":
        metadata = strategy.training_metadata
        if metadata is None:
            raise RuntimeError(
                f"source run {source_id!r} loaded without training_metadata; "
                f"the saved strategy state may be corrupt."
            )
        train_end = metadata.train_end.strftime("%Y-%m-%d")
        return f"{ticker}-{cfg.strategy.name}-{train_end}"
    return f"{ticker}-{cfg.strategy.name}-HPO-{source_id}"


def create_deployment(
    *,
    source_kind: SourceKind,
    source_id: str,
    store_root: Path,
    name: str | None = None,
    warmup_bars: int | None = None,
    deployment_id: str | None = None,
) -> Deployment:
    """
    Create a new deployment directory pointing at a trained source.

    Side effects: validates the source exists (resolves its
    ``strategy_state/``), creates ``<store_root>/deployments/<id>/``,
    and writes the deployment manifest + empty signal log. Returns the
    typed :class:`Deployment` for the caller to surface to a UI or CLI.

    ``warmup_bars`` defaults to ``None``, in which case the value is
    auto-derived by loading the source's strategy and asking it via
    :func:`recommend_warmup_bars`. The derived value is frozen into the
    deployment manifest at create-time; subsequent predict calls use the
    stored count, so a model change post-create does not silently move
    the goalposts.

    ``deployment_id`` is auto-generated (UUID4 hex) unless supplied —
    callers in test harnesses pin it for determinism; everyday use
    accepts the default.
    """

    if warmup_bars is not None and warmup_bars < 1:
        raise ValueError(
            f"warmup_bars must be >= 1, got {warmup_bars}; fix by passing a "
            f"strictly positive bar count, or pass None to auto-derive from "
            f"the strategy."
        )

    source_run_dir = resolve_strategy_state_path(source_kind, source_id, store_root).parent
    strategy = load_strategy_from_run_dir(source_run_dir)

    final_name = name if name is not None else _auto_generate_name(
        source_kind, source_id, source_run_dir, strategy
    )
    final_id = deployment_id if deployment_id is not None else uuid.uuid4().hex
    if warmup_bars is None:
        resolved_warmup = recommend_warmup_bars(strategy)
        _logger.info(
            "deployment auto-derived warmup_bars=%d (required=%d + margin=%d, floor=%d)",
            resolved_warmup,
            strategy.required_warmup_bars,
            strategy.convergence_margin_bars,
            _AUTO_DERIVED_MIN_WARMUP_BARS,
        )
    else:
        resolved_warmup = warmup_bars

    deployment = Deployment(
        deployment_id=final_id,
        name=final_name,
        source_kind=source_kind,
        source_id=source_id,
        warmup_bars=resolved_warmup,
        created_at=pd.Timestamp.now(tz="UTC"),
    )

    dep_dir = resolve_deployment_dir(store_root, final_id)
    dep_dir.mkdir(parents=True, exist_ok=False)
    json_io.write(dep_dir / DEPLOYMENT_MANIFEST_JSON, deployment.to_dict())
    (dep_dir / DEPLOYMENT_SIGNALS_JSONL).touch()

    _logger.info(
        "deployment %s created: source=%s:%s warmup=%d",
        final_id, source_kind, source_id, resolved_warmup,
    )
    return deployment


def load_deployment(store_root: Path, deployment_id: str) -> Deployment:
    """
    Reconstruct a :class:`Deployment` from its persisted manifest.
    """

    manifest_path = resolve_deployment_dir(store_root, deployment_id) / DEPLOYMENT_MANIFEST_JSON
    try:
        raw = json_io.read_dict(manifest_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"deployment {deployment_id!r} not found at {manifest_path}; "
            f"fix by passing the id of a deployment created via create_deployment()."
        ) from exc
    return Deployment.from_dict(raw)


def read_signals(store_root: Path, deployment_id: str) -> tuple[SignalRow, ...]:
    """
    Read the deployment's signal log in append order.

    Returns an empty tuple for a newly created deployment whose
    ``signals.jsonl`` has not yet been appended to.
    """

    path = resolve_deployment_dir(store_root, deployment_id) / DEPLOYMENT_SIGNALS_JSONL
    try:
        rows = json_io.read_jsonl(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"deployment {deployment_id!r} has no signals.jsonl at {path}; "
            f"the deployment directory may be corrupt."
        ) from exc
    return tuple(SignalRow.from_dict(d) for d in rows)


def predict(
    *,
    deployment_id: str,
    store_root: Path,
    as_of: pd.Timestamp | None = None,
) -> SignalRow:
    """
    Generate (or recall) today's signal for ``deployment_id``.

    Workflow:

    1. Load the deployment manifest and resolve the source's run-dir +
       trained strategy via the registry-driven loader.
    2. Read the strategy's :class:`TrainingMetadata` for ``train_end``
       and ``interval``.
    3. Resolve ``as_of`` (default: wall-clock UTC now) and fetch a
       warmup window of bars through ``as_of`` via the cadence-specific
       :class:`~src.data.live_fetcher.LiveBarFetcher`. The window is
       allowed to overlap the training period — see the module docstring.
    4. **Anti-leakage guard**: assert the *last fetched bar* is strictly
       after ``train_end``. Doing the check on the fetched data (not on
       the user-supplied ``as_of``) handles the realistic case where
       the vendor only has data up to the previous session.
    5. Run ``strategy.generate_signals(bars)``; the signal at
       ``bars.index[-1]`` is today's value. NaN at that position means
       the warmup window is too short for the strategy's longest
       indicator — surface loudly as :class:`WarmupInsufficientError`.
    6. **Idempotent append**: if ``signals.jsonl`` already carries a row
       for ``bars.index[-1]``, return that row unchanged. Otherwise
       append a new row and return it.
    """

    deployment = load_deployment(store_root, deployment_id)
    source_run_dir = resolve_strategy_state_path(
        deployment.source_kind, deployment.source_id, store_root
    ).parent
    cfg = load_experiment_config_from_run(source_run_dir)
    if len(cfg.data.tickers) != 1:
        raise NotImplementedError(
            f"deployment {deployment_id!r} sources a {len(cfg.data.tickers)}-ticker "
            f"strategy; live predict for multi-ticker / pairs strategies is not "
            f"implemented in this build."
        )
    ticker = cfg.data.tickers[0]

    strategy = load_strategy_from_run_dir(source_run_dir)
    metadata = strategy.training_metadata
    if metadata is None:
        raise RuntimeError(
            f"deployment {deployment_id!r} source has no training_metadata; "
            f"the saved strategy state may be corrupt."
        )

    resolved_as_of = as_of if as_of is not None else pd.Timestamp.now(tz="UTC")
    fetcher = resolve_fetcher(metadata.interval)
    if metadata.interval is not Interval.DAILY:
        raise NotImplementedError(
            f"warmup window math not implemented for interval={metadata.interval}; "
            f"daily is the only supported cadence today."
        )
    window = pd.Timedelta(days=int(deployment.warmup_bars * 1.5) + 30)
    start = (resolved_as_of - window).to_pydatetime()
    end = resolved_as_of.to_pydatetime()
    bars = fetcher.fetch(ticker, start, end, metadata.interval)

    if bars.empty:
        raise WarmupInsufficientError(
            f"deployment {deployment_id!r}: live fetch returned no bars for "
            f"{ticker} over [{start}, {end}]; the vendor may have no data for "
            f"this ticker in the requested window. Verify the ticker is listed "
            f"and that the warmup window does not predate its first session."
        )

    last_bar_ts = _to_naive(pd.Timestamp(bars.index[-1]))
    train_end = _to_naive(metadata.train_end)
    if last_bar_ts <= train_end:
        raise LeakageError(
            f"deployment {deployment_id!r}: last fetched bar at {last_bar_ts} is "
            f"not strictly after train_end={metadata.train_end}; refusing to "
            f"predict on a bar the model saw during training. Wait for the next "
            f"completed session or train a fresher model."
        )

    signals = strategy.generate_signals(bars)
    last_signal = signals.iloc[-1]
    if pd.isna(last_signal):
        vendor_first_bar = pd.Timestamp(bars.index[0])
        raise WarmupInsufficientError(
            f"deployment {deployment_id!r}: strategy produced NaN at {last_bar_ts} "
            f"(requested {deployment.warmup_bars} warmup bars, vendor returned "
            f"{len(bars)}, earliest available bar {vendor_first_bar.date()}). "
            f"The strategy's longest indicator lookback exceeds what the vendor "
            f"could supply for {ticker}. Either pick a different ticker with "
            f"longer history, or wait for the vendor to accumulate more bars."
        )

    return _append_or_recall_signal(
        store_root=store_root,
        deployment_id=deployment_id,
        bars=bars,
        signal_value=float(last_signal),
        last_bar_ts=last_bar_ts,
        source_run_id=deployment.source_id,
        warmup_bars_used=deployment.warmup_bars,
    )


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _append_or_recall_signal(
    *,
    store_root: Path,
    deployment_id: str,
    bars: pd.DataFrame,
    signal_value: float,
    last_bar_ts: pd.Timestamp,
    source_run_id: str,
    warmup_bars_used: int,
) -> SignalRow:
    """
    Append a new signal row for ``last_bar_ts`` unless one already exists.

    Dedup is by ``bar_ts`` only — two predicts on different
    ``submitted_at`` wall-clocks but the same target bar produce one
    row, not two. Returns either the freshly written row or the prior
    row, byte-equivalent under :meth:`SignalRow.to_dict`.
    """

    existing = read_signals(store_root, deployment_id)
    for row in existing:
        if _to_naive(row.bar_ts) == last_bar_ts:
            return row

    new_row = SignalRow(
        submitted_at=pd.Timestamp.now(tz="UTC"),
        bar_ts=last_bar_ts,
        signal=signal_value,
        warmup_fingerprint=fingerprint_bars(bars),
        source_run_id=source_run_id,
        warmup_bars_used=warmup_bars_used,
    )
    path = resolve_deployment_dir(store_root, deployment_id) / DEPLOYMENT_SIGNALS_JSONL
    json_io.append_jsonl(path, new_row.to_dict())
    _logger.info(
        "deployment %s: signal=%.4f at bar_ts=%s",
        deployment_id, signal_value, last_bar_ts,
    )
    return new_row


__all__ = [
    "Deployment",
    "SignalRow",
    "create_deployment",
    "load_deployment",
    "predict",
    "read_signals",
    "recommend_warmup_bars",
    "resolve_deployment_dir",
    "resolve_strategy_state_path",
]
