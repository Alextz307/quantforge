"""Shared pytest fixtures: per-test temp DB, test secret, fresh app + DB."""

from __future__ import annotations

import os
import secrets
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.core import json_io
from src.core.persistence import (
    COMPARISONS_SUBDIR,
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_MANIFEST_JSON,
    EXPERIMENT_METRICS_JSON,
    FOLD_RESULTS_JSONL,
    HOLDOUT_EVAL_JSON,
    HOLDOUT_EVALS_SUBDIR,
    HPO_SUBDIR,
    REGIME_REPORTS_SUBDIR,
)
from src.core.types import Interval
from src.engine.scenarios import SlippageScenario
from src.optimization.checkpointing import BEST_CONFIG_YAML_NAME, TRIALS_JSONL_NAME
from src.orchestration.study import STUDY_STATE_FILENAME
from src.orchestration.study_state import LEG_STEPS_ORDER, LegState, StudyState
from webapp.backend.app.core.rate_limit import login_limiter
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.core.types import Role
from webapp.backend.app.infrastructure.db import bootstrap_schema, open_db
from webapp.backend.app.main import create_app
from webapp.backend.app.services.plots import PLOTS_DIRNAME
from webapp.backend.app.services.user_service import create_user

TEST_SECRET_KEY = secrets.token_urlsafe(48)


def make_valid_experiment_payload() -> dict[str, object]:
    """Canonical fully-populated ExperimentConfig payload for B2 validate tests."""
    return {
        "name": "test_run",
        "seed": 42,
        "data": {
            "source": "yfinance",
            "tickers": ["SPY"],
            "start": "2020-01-01",
            "end": "2024-12-31",
            "interval": "daily",
        },
        "strategy": {
            "name": "AdaptiveBollinger",
            "params": {"window": 20, "k": 2.0, "trend_window": 100},
        },
        "validation": {"n_splits": 3, "test_size": 252, "gap": 5, "expanding": True},
    }


PLOT_FILENAME = "equity.png"
PLOT_BYTES = b"\x89PNG\r\n\x1a\n"
DEFAULT_TICKER = "SPY"
DEFAULT_INTERVAL = Interval.DAILY.value
TRIAL_STATE_COMPLETE = "COMPLETE"
TRIAL_STATE_FAIL = "FAIL"


@pytest.fixture(autouse=True)
def _webapp_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("WEBAPP_SECRET_KEY", TEST_SECRET_KEY)
    monkeypatch.setenv("WEBAPP_DB_PATH", str(tmp_path / "webapp.sqlite"))
    monkeypatch.setenv("WEBAPP_ENV", "local")
    get_settings.cache_clear()
    login_limiter.reset()
    yield
    get_settings.cache_clear()
    login_limiter.reset()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def db_conn() -> Iterator[sqlite3.Connection]:
    with open_db() as conn:
        bootstrap_schema(conn)
        yield conn


TEST_USERNAME = "alex"
TEST_PASSWORD = "password123"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "adminpass!"
SECONDARY_USERNAME = "bob"
SECONDARY_PASSWORD = "secondary!"


@pytest.fixture
def authed_client(client: TestClient, db_conn: sqlite3.Connection) -> TestClient:
    """A TestClient with an authenticated session cookie for a regular user."""
    create_user(db_conn, username=TEST_USERNAME, password=TEST_PASSWORD, role=Role.USER)
    response = client.post(
        "/api/auth/login", json={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    assert response.status_code == HTTPStatus.OK
    return client


@pytest.fixture
def _jobs_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Flip ``WEBAPP_JOBS_ENABLED`` on + point job artifacts at ``tmp_path``.

    Must run before any TestClient is constructed (lifespan reads settings on
    create_app). Tests requesting ``jobs_client`` get this fixture transitively.
    """
    monkeypatch.setenv("WEBAPP_JOBS_ENABLED", "true")
    monkeypatch.setenv("WEBAPP_JOB_TEMP_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("WEBAPP_STORE_ROOT", str(tmp_path / "experiment_results"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def jobs_client(_jobs_enabled: None) -> Iterator[TestClient]:
    """A jobs-enabled TestClient — sets the feature flag before app creation."""
    with TestClient(create_app()) as test_client:
        yield test_client


def _create_user_and_login(
    client: TestClient,
    db_conn: sqlite3.Connection,
    *,
    username: str,
    password: str,
    role: Role,
) -> None:
    create_user(db_conn, username=username, password=password, role=role)
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == HTTPStatus.OK


@pytest.fixture
def authed_jobs_client(jobs_client: TestClient, db_conn: sqlite3.Connection) -> TestClient:
    """``jobs_client`` authenticated as a regular user."""
    _create_user_and_login(
        jobs_client,
        db_conn,
        username=TEST_USERNAME,
        password=TEST_PASSWORD,
        role=Role.USER,
    )
    return jobs_client


@pytest.fixture
def admin_jobs_client(jobs_client: TestClient, db_conn: sqlite3.Connection) -> TestClient:
    """``jobs_client`` authenticated as an admin."""
    _create_user_and_login(
        jobs_client,
        db_conn,
        username=ADMIN_USERNAME,
        password=ADMIN_PASSWORD,
        role=Role.ADMIN,
    )
    return jobs_client


def make_synthetic_run(
    parent_runs_dir: Path,
    *,
    experiment_id: str,
    name: str = "synthetic",
    strategy: str = "AdaptiveBollinger",
    tickers: list[str] | None = None,
    interval: str = DEFAULT_INTERVAL,
    created_at: datetime | None = None,
    sharpe_mean: float = 0.5,
    n_folds: int = 3,
    write_metrics: bool = True,
    write_config: bool = True,
    write_plot: bool = True,
) -> Path:
    """Materialize a minimal valid run directory under ``parent_runs_dir``.

    Returns the run directory. The shape mirrors ``src.core.persistence``:
    ``manifest.json`` + ``metrics.json`` + ``config.yaml`` + ``fold_results.jsonl``
    + ``plots/equity.png``. Toggles let tests probe degenerate runs.
    """
    run_dir = parent_runs_dir / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ts = (created_at or datetime(2026, 1, 1, tzinfo=UTC)).isoformat()

    json_io.write(
        run_dir / EXPERIMENT_MANIFEST_JSON,
        {
            "experiment_id": experiment_id,
            "name": name,
            "created_at": ts,
            "git_sha": "abc1234",
            "seed": 42,
            "data_hash": "deadbeef",
            "slippage_scenario": SlippageScenario.NORMAL.value,
            "holdout_start": None,
            "pretrained_leaves": [],
        },
    )

    if write_metrics:
        json_io.write(
            run_dir / EXPERIMENT_METRICS_JSON,
            {
                "sharpe_mean": sharpe_mean,
                "calmar_mean": 1.2,
                "n_folds": n_folds,
            },
        )

    if write_config:
        config = {
            "name": name,
            "seed": 42,
            "data": {
                "source": {"name": "parquet", "params": {"data_dir": "tests/fixtures"}},
                "tickers": tickers or [DEFAULT_TICKER],
                "start": "2018-01-02T00:00:00",
                "end": "2024-12-31T00:00:00",
                "interval": interval,
            },
            "strategy": {"name": strategy, "params": {}},
        }
        (run_dir / EXPERIMENT_CONFIG_YAML).write_text(yaml.safe_dump(config), encoding="utf-8")

    folds_path = run_dir / FOLD_RESULTS_JSONL
    fold_lines = [
        {
            "fold_index": i,
            "train_start": "2020-01-01T00:00:00",
            "train_end": "2020-06-30T00:00:00",
            "test_start": "2020-07-01T00:00:00",
            "test_end": "2020-12-31T00:00:00",
            "total_return": 0.05,
            "annualized_return": 0.10,
            "annualized_volatility": 0.15,
            "sharpe_ratio": sharpe_mean,
            "sortino_ratio": 0.7,
            "calmar_ratio": 1.0,
            "max_drawdown": -0.05,
            "win_rate": 0.55,
            "trade_count": 20,
            "equity_curve": [10000.0, 10100.0, 10500.0],
        }
        for i in range(n_folds)
    ]
    json_io.write_jsonl(folds_path, fold_lines)

    if write_plot:
        plots = run_dir / PLOTS_DIRNAME
        plots.mkdir(exist_ok=True)
        (plots / PLOT_FILENAME).write_bytes(PLOT_BYTES)

    return run_dir


def _aggregate_stats(sharpe_mean: float = 0.5) -> dict[str, object]:
    """Synthetic per-strategy / per-regime aggregate-stats payload."""
    return {
        "n_folds": 3,
        "sharpe_mean": sharpe_mean,
        "sharpe_std": 0.1,
        "sharpe_ci95_low": sharpe_mean - 0.2,
        "sharpe_ci95_high": sharpe_mean + 0.2,
        "sortino_mean": 0.7,
        "sortino_std": 0.15,
        "sortino_ci95_low": 0.5,
        "sortino_ci95_high": 0.9,
        "calmar_mean": 1.0,
        "calmar_std": 0.2,
        "calmar_ci95_low": 0.6,
        "calmar_ci95_high": 1.4,
        "total_return_mean": 0.05,
        "total_return_std": 0.02,
        "max_drawdown_mean": -0.05,
        "max_drawdown_worst": -0.1,
        "win_rate_mean": 0.55,
        "trade_count_total": 60,
    }


def make_synthetic_comparison(
    parent_dir: Path,
    *,
    name: str,
    strategies: dict[str, str] | None = None,
    created_at: datetime | None = None,
    write_plot: bool = True,
) -> Path:
    """Materialize a minimal valid comparison directory under ``parent_dir``."""
    cmp_dir = parent_dir / name
    cmp_dir.mkdir(parents=True, exist_ok=True)
    ts = (created_at or datetime(2026, 4, 1, tzinfo=UTC)).isoformat()
    strategies = strategies or {
        "AdaptiveBollinger": "20260101_120000_AdaptiveBollinger_abc1234_deadbeef",
    }

    json_io.write(
        cmp_dir / EXPERIMENT_MANIFEST_JSON,
        {
            "out_name": name,
            "created_at": ts,
            "git_sha": "abc1234",
            "per_strategy_experiment_id": strategies,
            "per_strategy_stats": {s: _aggregate_stats() for s in strategies},
        },
    )

    if write_plot:
        plots = cmp_dir / PLOTS_DIRNAME
        plots.mkdir(exist_ok=True)
        (plots / PLOT_FILENAME).write_bytes(PLOT_BYTES)

    return cmp_dir


def make_synthetic_regime_report(
    parent_dir: Path,
    *,
    name: str,
    experiment_id: str = "20260101_120000_AdaptiveBollinger_abc1234_deadbeef",
    kind: str = "trend",
    detector_name: str = "trend",
    regime_labels: tuple[str, ...] = ("bull", "bear"),
    created_at: datetime | None = None,
    write_plot: bool = True,
) -> Path:
    """Materialize a minimal valid regime-report directory under ``parent_dir``."""
    report_dir = parent_dir / name
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = (created_at or datetime(2026, 4, 2, tzinfo=UTC)).isoformat()

    json_io.write(
        report_dir / EXPERIMENT_MANIFEST_JSON,
        {
            "out_name": name,
            "experiment_id": experiment_id,
            "kind": kind,
            "detector_name": detector_name,
            "created_at": ts,
            "git_sha": "abc1234",
            "per_regime_stats": {label: _aggregate_stats() for label in regime_labels},
            "per_regime_fold_indices": {label: [i] for i, label in enumerate(regime_labels)},
            "mixed_fold_indices": [],
            "slices": [
                {
                    "label": regime_labels[0],
                    "start": "2020-01-01T00:00:00",
                    "end": "2020-06-30T00:00:00",
                }
            ],
        },
    )

    if write_plot:
        plots = report_dir / PLOTS_DIRNAME
        plots.mkdir(exist_ok=True)
        (plots / PLOT_FILENAME).write_bytes(PLOT_BYTES)

    return report_dir


def make_synthetic_holdout_eval(
    parent_dir: Path,
    *,
    name: str,
    source_kind: str = "run",
    source_id: str = "20260101_120000_AdaptiveBollinger_abc1234_deadbeef",
    holdout_start: datetime | None = None,
    created_at: datetime | None = None,
    sharpe_ratio: float = 0.6,
    write_plot: bool = True,
) -> Path:
    """Materialize a minimal valid holdout-eval directory under ``parent_dir``."""
    eval_dir = parent_dir / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts = (created_at or datetime(2026, 4, 3, tzinfo=UTC)).isoformat()
    holdout_ts = (holdout_start or datetime(2024, 1, 1, tzinfo=UTC)).isoformat()

    json_io.write(
        eval_dir / HOLDOUT_EVAL_JSON,
        {
            "is_holdout_eval": True,
            "out_name": name,
            "source_kind": source_kind,
            "source_id": source_id,
            "source_path": f"experiment_results/runs/{source_id}",
            "holdout_start": holdout_ts,
            "data_hash": "deadbeef",
            "git_sha": "abc1234",
            "created_at": ts,
            "n_dev_bars": 1000,
            "n_holdout_bars": 200,
            "slippage_scenario": SlippageScenario.NORMAL.value,
            "metrics": {
                "total_return": 0.07,
                "annualized_return": 0.12,
                "annualized_volatility": 0.18,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": 0.85,
                "calmar_ratio": 1.1,
                "max_drawdown": -0.06,
                "win_rate": 0.56,
                "trade_count": 25,
            },
            "equity_curve": [10000.0, 10100.0, 10500.0],
        },
    )

    if write_plot:
        plots = eval_dir / PLOTS_DIRNAME
        plots.mkdir(exist_ok=True)
        (plots / PLOT_FILENAME).write_bytes(PLOT_BYTES)

    return eval_dir


def make_synthetic_study(
    parent_studies_dir: Path,
    *,
    name: str,
    spec_name: str = "demo_spec",
    spec_hash: str = "deadbeef" * 8,
    started_at: datetime | None = None,
    legs: tuple[tuple[str, str, bool], ...] = (
        ("AdaptiveBollinger", "spy_daily_5y", True),
        ("AdaptiveBollinger", "spy_daily_10y", False),
    ),
    cross_strategy_compares_done: tuple[str, ...] = (),
) -> Path:
    """Materialize a minimal valid study directory.

    ``legs`` is ``(strategy, universe, is_complete)``. Complete legs get a
    ``run_experiment_id`` and a ``completed_at``; incomplete legs match the
    ``LegState.initial`` shape.
    """
    study_dir = parent_studies_dir / name
    study_dir.mkdir(parents=True, exist_ok=True)
    ts = started_at or datetime(2026, 4, 1, tzinfo=UTC)
    leg_states: list[LegState] = []
    for strategy, universe, is_complete in legs:
        base = LegState.initial(f"{strategy}__{universe}", strategy, universe)
        if is_complete:
            base = replace(
                base,
                started_at=ts,
                completed_at=ts,
                steps_completed=LEG_STEPS_ORDER,
                is_complete=True,
                run_experiment_id=f"20260101_120000_{strategy}_abc1234_deadbeef",
            )
        leg_states.append(base)
    state = StudyState(
        spec_name=spec_name,
        spec_hash=spec_hash,
        started_at=ts,
        legs=tuple(leg_states),
        cross_strategy_compares_done=cross_strategy_compares_done,
    )
    json_io.write(study_dir / STUDY_STATE_FILENAME, state.to_dict())
    return study_dir


CONSOLIDATED_TABLE_FILENAME = "master_ranking.tex"
CONSOLIDATED_PLOT_FILENAME = "strategy_x_universe_heatmap.png"


def make_synthetic_consolidated_report(
    study_dir: Path,
    *,
    study_name: str,
    publish_label: str | None = None,
    created_at: datetime | None = None,
    git_sha: str = "deadbeef" * 5,
    strategies: tuple[str, ...] = ("AdaptiveBollinger", "PairsTrading"),
    universes: tuple[str, ...] = ("spy_daily_5y", "spy_daily_10y"),
    incomplete_leg_ids: tuple[str, ...] = (),
    n_legs_with_regime: int = 2,
    n_legs_with_holdout: int = 1,
    n_universes_with_pairwise: int = 1,
) -> Path:
    """Materialize a minimal valid consolidated-report tree under ``study_dir``."""
    from src.visualization.plots import MANIFEST_FILENAME

    plots_dir = study_dir / PLOTS_DIRNAME
    tables_dir = study_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / CONSOLIDATED_PLOT_FILENAME).write_bytes(PLOT_BYTES)
    (tables_dir / CONSOLIDATED_TABLE_FILENAME).write_text("% latex table stub", encoding="utf-8")
    ts = created_at or datetime(2026, 4, 5, tzinfo=UTC)
    manifest = {
        "study_name": study_name,
        "publish_label": publish_label or study_name,
        "study_dir": str(study_dir),
        "created_at": ts.isoformat(),
        "git_sha": git_sha,
        "strategies": list(strategies),
        "universes": list(universes),
        "incomplete_leg_ids": list(incomplete_leg_ids),
        "per_leg_run_id": {},
        "n_legs_with_regime": n_legs_with_regime,
        "n_legs_with_holdout": n_legs_with_holdout,
        "n_universes_with_pairwise": n_universes_with_pairwise,
    }
    json_io.write(study_dir / MANIFEST_FILENAME, manifest)
    return study_dir


def make_synthetic_hpo_study(
    parent_hpo_dir: Path,
    *,
    name: str,
    n_trials: int = 3,
    n_complete: int | None = None,
    best_value: float = 0.8,
    best_trial_number: int | None = None,
    created_at: datetime | None = None,
    write_best_config: bool = True,
) -> Path:
    """Materialize a minimal valid HPO-study directory.

    Writes ``trials.jsonl`` with ``n_trials`` records (the first
    ``n_complete`` are ``COMPLETE`` and carry monotonically-increasing
    ``value`` fields, peaking at ``best_value`` on ``best_trial_number``).
    The trials.jsonl mtime is stamped to ``created_at`` so summary tests
    can assert deterministic ordering.
    """
    if n_complete is None:
        n_complete = n_trials
    if best_trial_number is None:
        best_trial_number = max(0, n_complete - 1)
    study_dir = parent_hpo_dir / name
    study_dir.mkdir(parents=True, exist_ok=True)
    ts = (created_at or datetime(2026, 4, 1, tzinfo=UTC)).isoformat()

    trials: list[dict[str, object]] = []
    for i in range(n_trials):
        is_complete = i < n_complete
        value: float | None = None
        if is_complete:
            value = best_value if i == best_trial_number else best_value - 0.5
        trials.append(
            {
                "number": i,
                "state": TRIAL_STATE_COMPLETE if is_complete else TRIAL_STATE_FAIL,
                "value": value,
                "params": {"window": 30 + i, "k": 1.5 + 0.1 * i},
                "user_attrs": {"experiment_id": f"20260101_120000_synthetic_abc_{i:04d}"},
                "datetime_start": ts,
                "datetime_complete": ts if is_complete else None,
            }
        )
    json_io.write_jsonl(study_dir / TRIALS_JSONL_NAME, trials)

    if write_best_config:
        best_cfg = {
            "name": "demo",
            "seed": 42,
            "strategy": {"name": "AdaptiveBollinger", "params": {"window": 30, "k": 2.0}},
        }
        (study_dir / BEST_CONFIG_YAML_NAME).write_text(yaml.safe_dump(best_cfg), encoding="utf-8")

    if created_at is not None:
        epoch = created_at.timestamp()
        os.utime(study_dir / TRIALS_JSONL_NAME, (epoch, epoch))

    return study_dir


@pytest.fixture
def webapp_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthetic store with two runs: one flat layout, one study-nested layout."""
    root = tmp_path / "experiment_results"
    flat_runs = root / "thesis_demo" / "runs"
    study_runs = root / "studies" / "main" / "runs"
    make_synthetic_run(
        flat_runs,
        experiment_id="20260101_120000_AdaptiveBollinger_abc1234_deadbeef",
        strategy="AdaptiveBollinger",
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    make_synthetic_run(
        study_runs,
        experiment_id="20260201_090000_PairsTrading_def5678_cafebabe",
        strategy="PairsTrading",
        tickers=["IVV", "VOO"],
        created_at=datetime(2026, 2, 1, 9, 0, 0, tzinfo=UTC),
    )
    make_synthetic_comparison(
        root / "thesis_demo" / COMPARISONS_SUBDIR,
        name="flat_compare",
        strategies={
            "AdaptiveBollinger": "20260101_120000_AdaptiveBollinger_abc1234_deadbeef",
            "PairsTrading": "20260201_090000_PairsTrading_def5678_cafebabe",
        },
    )
    make_synthetic_regime_report(
        root / "thesis_demo" / REGIME_REPORTS_SUBDIR,
        name="flat_regime",
    )
    make_synthetic_holdout_eval(
        root / "studies" / "main" / HOLDOUT_EVALS_SUBDIR,
        name="study_holdout",
    )
    make_synthetic_study(
        root / "studies",
        name="main",
    )
    make_synthetic_hpo_study(
        root / "studies" / "main" / HPO_SUBDIR,
        name="AdaptiveBollinger__spy_daily_5y",
    )
    monkeypatch.setenv("WEBAPP_STORE_ROOT", str(root))
    get_settings.cache_clear()
    return root
