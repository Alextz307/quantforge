"""
Unit tests for the ``experiment importance`` metrics-reproduction oracle.

``_metrics_reproduced`` decides whether a re-run reproduced a finished run's
models: identical fold models yield bit-identical aggregate metrics, so a
match means "attach importance in place" and a mismatch means "save it as a
separate run". The end-to-end backfill / divergence routing is covered by the
CLI integration tests; these pin the comparison itself.
"""

from __future__ import annotations

from pathlib import Path

from scripts.experiment import _discard_prior_diverged_run, _metrics_reproduced
from src.core import json_io
from src.core.constants import IMPORTANCE_REPRODUCTION_ABS_TOL, IMPORTANCE_REPRODUCTION_RTOL
from src.core.persistence import (
    FEATURE_IMPORTANCE_DIVERGED_JSON,
    FEATURE_IMPORTANCE_JSON,
    RUNS_SUBDIR,
)

_BASE_METRICS: dict[str, object] = {
    "n_folds": 3,
    "sharpe_mean": 1.2345,
    "sharpe_std": 0.4,
    "max_drawdown_worst": -0.18,
    "total_return_mean": 0.31,
    "trade_count_total": 42,
}

_WITHIN_TOLERANCE_DELTA = IMPORTANCE_REPRODUCTION_RTOL / 10.0
_ABOVE_TOLERANCE_DELTA = 1e-4


def test_identical_metrics_reproduce() -> None:
    assert _metrics_reproduced(_BASE_METRICS, dict(_BASE_METRICS), IMPORTANCE_REPRODUCTION_RTOL)


def test_difference_above_tolerance_does_not_reproduce() -> None:
    candidate = dict(_BASE_METRICS)
    candidate["sharpe_mean"] = 1.2345 + _ABOVE_TOLERANCE_DELTA
    assert not _metrics_reproduced(_BASE_METRICS, candidate, IMPORTANCE_REPRODUCTION_RTOL)


def test_difference_within_tolerance_reproduces() -> None:
    candidate = dict(_BASE_METRICS)
    candidate["sharpe_mean"] = 1.2345 + _WITHIN_TOLERANCE_DELTA
    assert _metrics_reproduced(_BASE_METRICS, candidate, IMPORTANCE_REPRODUCTION_RTOL)


def test_matching_nans_reproduce() -> None:
    original = {**_BASE_METRICS, "sharpe_mean": float("nan")}
    candidate = {**_BASE_METRICS, "sharpe_mean": float("nan")}
    assert _metrics_reproduced(original, candidate, IMPORTANCE_REPRODUCTION_RTOL)


def test_nan_versus_finite_does_not_reproduce() -> None:
    original = {**_BASE_METRICS, "sharpe_mean": float("nan")}
    assert not _metrics_reproduced(original, dict(_BASE_METRICS), IMPORTANCE_REPRODUCTION_RTOL)


def test_differing_key_sets_do_not_reproduce() -> None:
    candidate = dict(_BASE_METRICS)
    del candidate["trade_count_total"]
    assert not _metrics_reproduced(_BASE_METRICS, candidate, IMPORTANCE_REPRODUCTION_RTOL)


def test_integer_trade_count_difference_does_not_reproduce() -> None:
    candidate = dict(_BASE_METRICS)
    candidate["trade_count_total"] = 43
    assert not _metrics_reproduced(_BASE_METRICS, candidate, IMPORTANCE_REPRODUCTION_RTOL)


def test_near_zero_within_abs_tol_reproduces() -> None:
    original = {**_BASE_METRICS, "total_return_mean": 0.0}
    candidate = {**_BASE_METRICS, "total_return_mean": IMPORTANCE_REPRODUCTION_ABS_TOL / 10.0}
    assert _metrics_reproduced(original, candidate, IMPORTANCE_REPRODUCTION_RTOL)


def test_near_zero_above_abs_tol_does_not_reproduce() -> None:
    original = {**_BASE_METRICS, "total_return_mean": 0.0}
    candidate = {**_BASE_METRICS, "total_return_mean": IMPORTANCE_REPRODUCTION_ABS_TOL * 10.0}
    assert not _metrics_reproduced(original, candidate, IMPORTANCE_REPRODUCTION_RTOL)


def test_bool_versus_equal_int_does_not_reproduce() -> None:
    original = {**_BASE_METRICS, "converged": True}
    candidate = {**_BASE_METRICS, "converged": 1}
    assert not _metrics_reproduced(original, candidate, IMPORTANCE_REPRODUCTION_RTOL)


def test_matching_bools_reproduce() -> None:
    original = {**_BASE_METRICS, "converged": True}
    candidate = {**_BASE_METRICS, "converged": True}
    assert _metrics_reproduced(original, candidate, IMPORTANCE_REPRODUCTION_RTOL)


_ORIGINAL_ID = "exp_original"


def _seed_diverged(tmp_path: Path, *, prior_id: str, source_run: str) -> tuple[Path, Path, Path]:
    store_root = tmp_path / "store"
    runs = store_root / RUNS_SUBDIR
    run_dir = runs / _ORIGINAL_ID
    run_dir.mkdir(parents=True)
    prior_dir = runs / prior_id
    prior_dir.mkdir(parents=True)
    json_io.write(prior_dir / FEATURE_IMPORTANCE_JSON, {"source_run": source_run, "aggregated": []})
    json_io.write(run_dir / FEATURE_IMPORTANCE_DIVERGED_JSON, {"diverged_run_id": prior_id})
    return store_root, run_dir, prior_dir


def test_discard_prior_diverged_run_removes_superseded_container(tmp_path: Path) -> None:
    store_root, run_dir, prior_dir = _seed_diverged(
        tmp_path, prior_id="exp_prior", source_run=_ORIGINAL_ID
    )

    _discard_prior_diverged_run(run_dir, store_root, _ORIGINAL_ID)

    assert not prior_dir.exists()


def test_discard_prior_diverged_run_spares_unrelated_run(tmp_path: Path) -> None:
    store_root, run_dir, prior_dir = _seed_diverged(
        tmp_path, prior_id="exp_other", source_run="someone_else"
    )

    _discard_prior_diverged_run(run_dir, store_root, _ORIGINAL_ID)

    assert prior_dir.exists()


def test_discard_prior_diverged_run_without_pointer_is_noop(tmp_path: Path) -> None:
    run_dir = tmp_path / "store" / RUNS_SUBDIR / _ORIGINAL_ID
    run_dir.mkdir(parents=True)

    _discard_prior_diverged_run(run_dir, tmp_path / "store", _ORIGINAL_ID)
