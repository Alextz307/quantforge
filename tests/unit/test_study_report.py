"""
Unit tests for the study-report consolidator.

Exercises :func:`consolidate_study` against a synthetic ``<study_dir>/``
tree built on ``tmp_path``. The tree mirrors the orchestrator's output
layout (``runs/``, ``holdout_evals/``, ``comparisons/``,
``study_state.json``) - handcrafting it here avoids running the
orchestrator and keeps these tests in unit-tier latency.

The reporter side (PNG/SVG/.tex emission) is exercised in
``test_study_report_reporter.py``; this file focuses on the JSON parsing
+ aggregation logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.metrics_aggregator import aggregate_folds
from src.core import json_io
from src.core.persistence import (
    COMPARISONS_SUBDIR,
    EXPERIMENT_MANIFEST_JSON,
    EXPERIMENT_METRICS_JSON,
    FOLD_RESULTS_JSONL,
    HOLDOUT_EVAL_JSON,
    HOLDOUT_EVALS_SUBDIR,
    RUNS_SUBDIR,
)
from src.core.types import Interval
from src.engine.scenarios import SlippageScenario
from src.orchestration.manifest import Manifest
from src.orchestration.study_report import (
    HoldoutSnapshot,
    aggregate_floor_bind_across_folds,
    consolidate_study,
)
from src.orchestration.study_state import (
    LegState,
    StudyState,
    write_study_state,
)
from src.orchestration.types import FoldRecord, PairwiseSignificance
from src.strategies.volatility_targeting import FLOOR_BIND_DIAGNOSTIC_KEY
from src.visualization.plots import MANIFEST_FILENAME
from tests.conftest import (
    make_log_return_equity_curve,
    make_stub_fold_record,
)

_BOOTSTRAP_SEED = 1337
_STUB_DATA_HASH = "a" * 64
_ANNUALIZATION_FACTOR = Interval.DAILY.annualization_factor()
_FOLD_TIME_RANGE_START = pd.Timestamp("2020-01-01")
_FOLD_TIME_RANGE_END = pd.Timestamp("2020-12-31")
_HOLDOUT_START = pd.Timestamp("2024-01-01")


def _write_fake_run(
    runs_dir: Path,
    *,
    run_id: str,
    name: str,
    sharpes: tuple[float, ...],
    write_metrics: bool = True,
) -> None:
    """
    Materialise a minimal valid ``runs/<id>/`` dir.

    Writes ``manifest.json`` and ``fold_results.jsonl`` so
    :func:`load_experiment_result` can reconstruct an
    :class:`ExperimentResult`. Skips ``config.yaml`` and per-fold
    artifacts - the consolidator doesn't read those. ``write_metrics=False``
    omits ``metrics.json`` to mimic a legacy/partial run the consolidator
    must recompute from folds rather than crash on.
    """

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    folds = tuple(
        make_stub_fold_record(
            i,
            sharpe=sharpe,
            equity_curve=make_log_return_equity_curve(sharpe, n=30, seed=_BOOTSTRAP_SEED + i),
        )
        for i, sharpe in enumerate(sharpes)
    )
    json_io.write_jsonl(run_dir / FOLD_RESULTS_JSONL, [f.to_dict() for f in folds])
    if write_metrics:
        json_io.write(
            run_dir / EXPERIMENT_METRICS_JSON,
            aggregate_folds(folds, annualization_factor=_ANNUALIZATION_FACTOR).to_dict(),
        )
    manifest = Manifest(
        experiment_id=run_id,
        name=name,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        git_sha="stubsha1",
        seed=42,
        data_hash=_STUB_DATA_HASH,
        slippage_scenario=SlippageScenario.NORMAL,
        interval=Interval.DAILY,
        risk_free_rate=0.0,
        holdout_start=_HOLDOUT_START,
    )
    json_io.write(run_dir / EXPERIMENT_MANIFEST_JSON, manifest.to_dict())


def _write_fake_holdout(
    holdout_dir: Path, *, leg_id: str, sharpe: float, dev_bars: int = 1000, holdout_bars: int = 250
) -> None:
    """
    Match ``HoldoutEvalResult.to_dict()`` schema for the consolidator's needs.
    """

    out_dir = holdout_dir / leg_id
    out_dir.mkdir(parents=True)
    payload: dict[str, object] = {
        "is_holdout_eval": True,
        "out_name": leg_id,
        "source_kind": "run",
        "source_id": f"stub_{leg_id}",
        "source_path": "/dev/null",
        "holdout_start": _HOLDOUT_START.isoformat(),
        "data_hash": _STUB_DATA_HASH,
        "git_sha": "stubsha1",
        "created_at": datetime.now(UTC).isoformat(),
        "n_dev_bars": dev_bars,
        "n_holdout_bars": holdout_bars,
        "slippage_scenario": "normal",
        "metrics": {
            "total_return": 0.05,
            "annualized_return": 0.10,
            "annualized_volatility": 0.15,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sharpe * 1.05,
            "calmar_ratio": sharpe * 0.9,
            "max_drawdown": -0.08,
            "win_rate": 0.55,
            "trade_count": 30,
            "sharpe_ci": {
                "point_estimate": sharpe,
                "lower": sharpe - 0.2,
                "upper": sharpe + 0.2,
                "confidence": 0.95,
                "n_resamples": 1000,
                "block_size": 5,
            },
        },
        "equity_curve": [1.0, 1.01, 1.02],
        "buy_and_hold": {
            "sharpe_ratio": 0.5,
            "sortino_ratio": 0.55,
            "calmar_ratio": 0.45,
            "max_drawdown": -0.10,
            "annualized_return": 0.07,
            "annualized_volatility": 0.14,
            "total_return": 0.04,
            "win_rate": 0.50,
            "trade_count": 1,
            "equity_curve": [1.0, 1.005, 1.01],
        },
    }
    json_io.write(out_dir / HOLDOUT_EVAL_JSON, payload)


def _write_fake_comparison(
    comparisons_dir: Path,
    *,
    universe: str,
    pairwise: tuple[PairwiseSignificance, ...],
) -> None:
    """
    Write the comparison manifest with the pairwise list the consolidator reads.
    """

    out_dir = comparisons_dir / universe
    out_dir.mkdir(parents=True)
    payload: dict[str, object] = {
        "out_name": universe,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": "stubsha1",
        "per_strategy_experiment_id": {},
        "per_strategy_stats": {},
        "pairwise": [p.to_dict() for p in pairwise],
    }
    json_io.write(out_dir / MANIFEST_FILENAME, payload)


def _build_study_dir(
    tmp_path: Path,
    *,
    legs: Sequence[tuple[str, str, str, tuple[float, ...]]],
    incomplete_leg_ids: tuple[str, ...] = (),
    legs_without_metrics: tuple[str, ...] = (),
    holdout_data: dict[str, float] | None = None,
    comparison_data: dict[str, tuple[PairwiseSignificance, ...]] | None = None,
) -> Path:
    """
    Assemble a full fake study tree on ``tmp_path``.

    ``legs`` is ``(leg_id, strategy, universe, sharpes)`` tuples - sharpes
    drives the synthesised fold records (one per element). ``incomplete_leg_ids``
    marks legs whose state stays ``is_complete=False``; their run dir is
    NOT written (mirrors a mid-leg failure).
    """

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    runs_dir = study_dir / RUNS_SUBDIR
    holdout_dir = study_dir / HOLDOUT_EVALS_SUBDIR
    comparisons_dir = study_dir / COMPARISONS_SUBDIR

    leg_states: list[LegState] = []
    for leg_id, strategy, universe, sharpes in legs:
        if leg_id in incomplete_leg_ids:
            leg_states.append(LegState.initial(leg_id, strategy, universe))
            continue
        run_id = f"stub_{leg_id}"
        _write_fake_run(
            runs_dir,
            run_id=run_id,
            name=leg_id,
            sharpes=sharpes,
            write_metrics=leg_id not in legs_without_metrics,
        )
        leg_states.append(
            LegState(
                leg_id=leg_id,
                strategy=strategy,
                universe=universe,
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                completed_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
                steps_completed=(),
                is_complete=True,
                error=None,
                run_experiment_id=run_id,
            )
        )

    if holdout_data:
        for leg_id, sharpe in holdout_data.items():
            _write_fake_holdout(holdout_dir, leg_id=leg_id, sharpe=sharpe)
    if comparison_data:
        for universe, pairwise in comparison_data.items():
            _write_fake_comparison(comparisons_dir, universe=universe, pairwise=pairwise)

    state = StudyState(
        spec_name="test_study",
        spec_hash="stubhash" * 8,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        legs=tuple(leg_states),
        cross_strategy_compares_done=tuple(comparison_data or {}),
    )
    write_study_state(study_dir / "study_state.json", state)
    return study_dir


def test_consolidate_study_full_tree(tmp_path: Path) -> None:
    """
    All artifact kinds present -> all maps populated.
    """

    legs = [
        ("StratA__uni1", "StratA", "uni1", (1.2, 1.3, 1.1)),
        ("StratA__uni2", "StratA", "uni2", (0.5, 0.6, 0.7)),
        ("StratB__uni1", "StratB", "uni1", (0.9, 1.0, 0.8)),
        ("StratB__uni2", "StratB", "uni2", (1.5, 1.6, 1.4)),
    ]
    pair_uni1 = (
        PairwiseSignificance(
            name_a="StratA",
            name_b="StratB",
            point_differential=0.3,
            lower=0.05,
            upper=0.55,
            confidence=0.95,
            significant=True,
        ),
    )
    study_dir = _build_study_dir(
        tmp_path,
        legs=legs,
        holdout_data={"StratA__uni1": 0.95, "StratB__uni2": 1.10},
        comparison_data={"uni1": pair_uni1},
    )

    report = consolidate_study(study_dir)

    assert report.study_name == "test_study"
    assert set(report.strategies) == {"StratA", "StratB"}
    assert set(report.universes) == {"uni1", "uni2"}
    assert len(report.per_leg_aggregate) == 4
    strat_a_uni1 = report.per_leg_aggregate[("StratA", "uni1")]
    assert strat_a_uni1.n_folds == 3
    assert strat_a_uni1.sharpe_mean == pytest.approx(
        aggregate_folds(
            _load_folds(study_dir, "stub_StratA__uni1"),
            annualization_factor=_ANNUALIZATION_FACTOR,
        ).sharpe_mean
    )
    assert ("StratA", "uni1") in report.per_leg_holdout
    assert report.per_leg_holdout[("StratA", "uni1")].sharpe_ratio == 0.95
    assert "uni1" in report.per_universe_pairwise
    assert report.per_universe_pairwise["uni1"][0].name_a == "StratA"
    assert report.incomplete_leg_ids == ()


def test_consolidate_study_recomputes_when_metrics_json_missing(tmp_path: Path) -> None:
    """
    A completed leg whose ``metrics.json`` is absent (legacy/partial run)
    must not crash consolidation; the aggregate is recomputed from the
    loaded folds using the manifest's interval/rate.
    """

    legs = [
        ("StratA__uni1", "StratA", "uni1", (1.2, 1.3, 1.1)),
        ("StratB__uni1", "StratB", "uni1", (0.9, 1.0, 0.8)),
    ]
    study_dir = _build_study_dir(
        tmp_path,
        legs=legs,
        legs_without_metrics=("StratB__uni1",),
    )

    report = consolidate_study(study_dir)

    assert len(report.per_leg_aggregate) == 2
    recomputed = report.per_leg_aggregate[("StratB", "uni1")]
    assert recomputed.n_folds == 3
    assert recomputed.sharpe_mean == pytest.approx(
        aggregate_folds(
            _load_folds(study_dir, "stub_StratB__uni1"),
            annualization_factor=_ANNUALIZATION_FACTOR,
        ).sharpe_mean
    )


def test_consolidate_study_skips_incomplete_legs(tmp_path: Path) -> None:
    """
    Incomplete legs surface in ``incomplete_leg_ids`` and don't pollute aggregates.
    """

    legs = [
        ("StratA__uni1", "StratA", "uni1", (1.0, 1.1)),
        ("StratA__uni2", "StratA", "uni2", (0.5,)),
    ]
    study_dir = _build_study_dir(tmp_path, legs=legs, incomplete_leg_ids=("StratA__uni2",))
    report = consolidate_study(study_dir)
    assert ("StratA", "uni1") in report.per_leg_aggregate
    assert ("StratA", "uni2") not in report.per_leg_aggregate
    assert report.incomplete_leg_ids == ("StratA__uni2",)


def test_consolidate_study_no_holdout_no_compare(tmp_path: Path) -> None:
    """
    Sparse tree: only runs/, no auxiliary artifacts.
    """

    legs = [("StratA__uni1", "StratA", "uni1", (1.0, 1.1, 1.2))]
    study_dir = _build_study_dir(tmp_path, legs=legs)
    report = consolidate_study(study_dir)
    assert len(report.per_leg_aggregate) == 1
    assert report.per_leg_holdout == {}
    assert report.per_universe_pairwise == {}


def test_consolidate_study_missing_state_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_study"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="study_state.json"):
        consolidate_study(empty_dir)


def test_holdout_snapshot_round_trip(tmp_path: Path) -> None:
    """
    Round-trip a real ``HoldoutEvalResult.to_dict()`` payload.
    """

    leg_dir = tmp_path / "holdout"
    _write_fake_holdout(leg_dir, leg_id="x", sharpe=0.42)
    snapshot = HoldoutSnapshot.from_holdout_json(leg_dir / "x" / HOLDOUT_EVAL_JSON)
    assert snapshot.sharpe_ratio == 0.42
    assert snapshot.holdout_start == _HOLDOUT_START
    assert snapshot.n_holdout_bars == 250


class TestAggregateFloorBindAcrossFolds:
    def test_returns_none_when_no_fold_carries_diagnostic(self) -> None:
        result = aggregate_floor_bind_across_folds([{}, {"other_key": 0.5}])
        assert result is None

    def test_aggregates_mean_max_min_across_folds(self) -> None:
        diagnostics = [
            {FLOOR_BIND_DIAGNOSTIC_KEY: 0.10},
            {FLOOR_BIND_DIAGNOSTIC_KEY: 0.30},
            {FLOOR_BIND_DIAGNOSTIC_KEY: 0.50},
        ]
        result = aggregate_floor_bind_across_folds(diagnostics)
        assert result is not None
        assert result.mean == pytest.approx(0.30)
        assert result.max == pytest.approx(0.50)
        assert result.min == pytest.approx(0.10)
        assert result.n_folds == 3

    def test_skips_non_finite_values(self) -> None:
        diagnostics = [
            {FLOOR_BIND_DIAGNOSTIC_KEY: 0.20},
            {FLOOR_BIND_DIAGNOSTIC_KEY: float("nan")},
            {FLOOR_BIND_DIAGNOSTIC_KEY: float("inf")},
            {FLOOR_BIND_DIAGNOSTIC_KEY: 0.40},
        ]
        result = aggregate_floor_bind_across_folds(diagnostics)
        assert result is not None
        assert result.n_folds == 2
        assert result.mean == pytest.approx(0.30)


def test_holdout_snapshot_rejects_non_holdout_payload(tmp_path: Path) -> None:
    """
    A regular run manifest must NOT be parsed as a holdout snapshot.
    """

    bad = tmp_path / "bad.json"
    json_io.write(bad, {"is_holdout_eval": False, "metrics": {}})
    with pytest.raises(ValueError, match="not a holdout-eval payload"):
        HoldoutSnapshot.from_holdout_json(bad)


def _load_folds(study_dir: Path, run_id: str) -> tuple[FoldRecord, ...]:
    """
    Re-read fold records from a fake run dir for direct AggregateStats comparison.
    """

    from src.orchestration.run_loader import load_experiment_result

    return load_experiment_result(study_dir / RUNS_SUBDIR / run_id).folds
