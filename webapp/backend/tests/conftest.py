"""Shared pytest fixtures: per-test temp DB, test secret, fresh app + DB."""

from __future__ import annotations

import secrets
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.core import json_io
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_MANIFEST_JSON,
    EXPERIMENT_METRICS_JSON,
    FOLD_RESULTS_JSONL,
)
from src.core.types import Interval
from src.engine.scenarios import SlippageScenario
from webapp.backend.app.core.rate_limit import login_limiter
from webapp.backend.app.core.settings import get_settings
from webapp.backend.app.infrastructure.db import bootstrap_schema, open_db
from webapp.backend.app.main import create_app
from webapp.backend.app.services.run_service import PLOTS_DIRNAME

TEST_SECRET_KEY = secrets.token_urlsafe(48)

PLOT_FILENAME = "equity.png"
PLOT_BYTES = b"\x89PNG\r\n\x1a\n"
DEFAULT_TICKER = "SPY"
DEFAULT_INTERVAL = Interval.DAILY.value


@pytest.fixture(autouse=True)
def _webapp_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("WEBAPP_SECRET_KEY", TEST_SECRET_KEY)
    monkeypatch.setenv("WEBAPP_DB_PATH", str(tmp_path / "webapp.sqlite"))
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


@pytest.fixture
def authed_client(client: TestClient, db_conn: sqlite3.Connection) -> TestClient:
    """A TestClient with an authenticated session cookie for a regular user."""
    from webapp.backend.app.core.types import Role
    from webapp.backend.app.services.user_service import create_user

    create_user(db_conn, username=TEST_USERNAME, password=TEST_PASSWORD, role=Role.USER)
    response = client.post(
        "/api/auth/login", json={"username": TEST_USERNAME, "password": TEST_PASSWORD}
    )
    assert response.status_code == HTTPStatus.OK
    return client


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
    monkeypatch.setenv("WEBAPP_STORE_ROOT", str(root))
    get_settings.cache_clear()
    return root
