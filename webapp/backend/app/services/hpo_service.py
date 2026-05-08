"""Read-only services for the persisted HPO-studies tree."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.core import json_io
from src.optimization.checkpointing import BEST_CONFIG_YAML_NAME, TRIALS_JSONL_NAME
from webapp.backend.app.infrastructure.store import (
    HpoStudyNotFoundError,
    find_hpo_study_dir,
    iter_hpo_study_dirs,
    store_label,
)
from webapp.backend.app.schemas.hpo import HpoDetail, HpoSummary, StudyDirection, TrialRow
from webapp.backend.app.schemas.jobs import TERMINAL_STATUSES, JobKind

__all__ = [
    "HpoStudyNotFoundError",
    "find_live_job_for",
    "get_hpo_study",
    "list_hpo_studies",
    "list_trials",
    "trial_row_from_record",
]

_COMPLETE_STATE = "COMPLETE"


def list_hpo_studies(root: Path) -> list[HpoSummary]:
    """List every HPO study under ``root``, newest first."""
    summaries: list[HpoSummary] = []
    for study_dir in iter_hpo_study_dirs(root):
        trials = json_io.read_jsonl(study_dir / TRIALS_JSONL_NAME)
        summaries.append(_summary_from_trials(study_dir, trials, root))
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def get_hpo_study(root: Path, name: str, *, live_job_id: str | None = None) -> HpoDetail:
    """Read the full detail payload for one HPO study.

    ``live_job_id`` is resolved by the router via :func:`find_live_job_for`
    against the jobs DB; passed through here to avoid coupling this layer
    to a sqlite connection.
    """
    study_dir = find_hpo_study_dir(root, name)
    trials = json_io.read_jsonl(study_dir / TRIALS_JSONL_NAME)
    summary = _summary_from_trials(study_dir, trials, root)
    return HpoDetail(
        name=summary.name,
        store=summary.store,
        created_at=summary.created_at,
        n_trials=summary.n_trials,
        n_complete=summary.n_complete,
        best_value=summary.best_value,
        best_trial_number=summary.best_trial_number,
        direction=summary.direction,
        best_config=_read_best_config(study_dir),
        live_job_id=live_job_id,
    )


def list_trials(root: Path, name: str, after_trial: int | None = None) -> list[TrialRow]:
    """Read the trial feed, optionally filtered to ``trial.number > after_trial``."""
    study_dir = find_hpo_study_dir(root, name)
    trials = json_io.read_jsonl(study_dir / TRIALS_JSONL_NAME)
    rows = [trial_row_from_record(t) for t in trials]
    if after_trial is not None:
        rows = [r for r in rows if r.number > after_trial]
    rows.sort(key=lambda r: r.number)
    return rows


def find_live_job_for(conn: sqlite3.Connection, study_name: str) -> str | None:
    """Return the id of a non-terminal TUNE job that's populating ``study_name``.

    TUNE jobs persist ``experiment_id = study_name`` at submission time
    (the directory name is known up front, unlike RUN jobs whose run
    dir basename is resolved post-completion). At most one non-terminal
    TUNE job per study is expected; we return the most recent.
    """
    terminal = tuple(s.value for s in TERMINAL_STATUSES)
    placeholders = ",".join("?" * len(terminal))
    row = conn.execute(
        f"SELECT id FROM jobs "
        f"WHERE kind = ? AND experiment_id = ? AND status NOT IN ({placeholders}) "
        f"ORDER BY id DESC LIMIT 1",
        (JobKind.TUNE.value, study_name, *terminal),
    ).fetchone()
    if row is None:
        return None
    return str(row["id"])


def trial_row_from_record(trial: dict[str, object]) -> TrialRow:
    """Materialise a ``TrialRow`` from one parsed ``trials.jsonl`` record.

    The Optuna callback stamps ``user_attrs.experiment_id`` after the
    per-trial ``Experiment.run()`` resolves a name, so the row can
    deep-link to the trial's run page.
    """
    user_attrs_raw = trial.get("user_attrs")
    user_attrs: dict[str, object] = user_attrs_raw if isinstance(user_attrs_raw, dict) else {}
    experiment_id = user_attrs.get("experiment_id")
    return TrialRow(
        number=json_io.get_int(trial, "number"),
        state=json_io.get_str(trial, "state"),
        value=json_io.get_optional_float(trial, "value"),
        params=json_io.get_dict(trial, "params"),
        datetime_start=json_io.get_optional_iso_datetime(trial, "datetime_start"),
        datetime_complete=json_io.get_optional_iso_datetime(trial, "datetime_complete"),
        experiment_id=experiment_id if isinstance(experiment_id, str) else None,
    )


def _summary_from_trials(
    study_dir: Path, trials: list[dict[str, object]], root: Path
) -> HpoSummary:
    n_complete = 0
    best_number: int | None = None
    best_value: float | None = None
    for t in trials:
        if json_io.get_str(t, "state") != _COMPLETE_STATE:
            continue
        n_complete += 1
        value = json_io.get_optional_float(t, "value")
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_number = json_io.get_int(t, "number")
    return HpoSummary(
        name=study_dir.name,
        store=store_label(study_dir, root),
        created_at=_mtime(study_dir / TRIALS_JSONL_NAME),
        n_trials=len(trials),
        n_complete=n_complete,
        best_value=best_value,
        best_trial_number=best_number,
        direction=StudyDirection.MAXIMIZE,
    )


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _read_best_config(study_dir: Path) -> dict[str, object]:
    path = study_dir / BEST_CONFIG_YAML_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return raw
