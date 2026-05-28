"""
Round-trip + create/load behaviour for :class:`Deployment`.

Tests the persistence shape (``manifest.json`` + ``deployment.yaml`` +
empty ``signals.jsonl``) and the auto-name generation rules — the
``predict`` flow is exercised in ``test_deployment_predict.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.persistence import (
    DEPLOYMENT_MANIFEST_JSON,
    DEPLOYMENT_SIGNALS_JSONL,
    DEPLOYMENTS_SUBDIR,
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_STRATEGY_SUBDIR,
    RUNS_SUBDIR,
)
from src.orchestration.deployment import (
    Deployment,
    create_deployment,
    load_deployment,
    read_signals,
    recommend_warmup_bars,
    resolve_strategy_state_path,
)
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from src.strategies.interface import RECURSIVE_LEAF_CONVERGENCE_MARGIN_BARS
from tests.conftest import make_synthetic_ohlcv_df

_TRAIN_ROWS = 250
_BOLLINGER_WINDOW = 20
_BOLLINGER_TREND_WINDOW = 50
_GARCH_P_MAX = 1
_GARCH_Q_MAX = 1
_CUSTOM_WARMUP = 100
_DEPLOYMENT_ID = "fixed-test-id"
_ABSOLUTE_WARMUP_FLOOR = 50


def _materialise_run(store_root: Path, run_id: str) -> Path:
    """
    Train a tiny AdaptiveBollinger and write a minimal run dir.
    """

    from src.core.config import load_experiment_config, write_frozen_yaml

    run_dir = store_root / RUNS_SUBDIR / run_id
    run_dir.mkdir(parents=True)
    cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
    write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, cfg)

    df = make_synthetic_ohlcv_df(n_rows=_TRAIN_ROWS)
    strategy = AdaptiveBollingerStrategy(
        window=_BOLLINGER_WINDOW,
        trend_window=_BOLLINGER_TREND_WINDOW,
        garch_p_max=_GARCH_P_MAX,
        garch_q_max=_GARCH_Q_MAX,
    )
    strategy.train(df)
    strategy.save(run_dir / EXPERIMENT_STRATEGY_SUBDIR)
    return run_dir


def test_deployment_dict_round_trip() -> None:
    """
    ``to_dict`` / ``from_dict`` recover the same Deployment instance.
    """

    created = pd.Timestamp("2026-05-28T00:00:00Z")
    original = Deployment(
        deployment_id="abc123",
        name="SPY-Strategy-2026-01-01",
        source_kind="run",
        source_id="20260101_SPY_run",
        warmup_bars=_CUSTOM_WARMUP,
        created_at=created,
    )

    restored = Deployment.from_dict(original.to_dict())

    assert restored == original


def test_deployment_rejects_unknown_source_kind() -> None:
    bad = {
        "deployment_id": "x",
        "name": "x",
        "source_kind": "not_a_kind",
        "source_id": "y",
        "warmup_bars": 252,
        "created_at": "2026-05-28T00:00:00+00:00",
    }
    with pytest.raises(ValueError, match="invalid source_kind"):
        Deployment.from_dict(bad)


def test_create_writes_artifacts(tmp_path: Path) -> None:
    """
    Three artifacts land on disk under deployments/<id>/.
    """

    store = tmp_path / "store"
    _materialise_run(store, "run_abc")

    deployment = create_deployment(
        source_kind="run",
        source_id="run_abc",
        store_root=store,
        deployment_id=_DEPLOYMENT_ID,
    )

    dep_dir = store / DEPLOYMENTS_SUBDIR / _DEPLOYMENT_ID
    assert (dep_dir / DEPLOYMENT_MANIFEST_JSON).is_file()
    assert (dep_dir / DEPLOYMENT_SIGNALS_JSONL).is_file()
    assert (dep_dir / DEPLOYMENT_SIGNALS_JSONL).read_text(encoding="utf-8") == ""
    assert (
        deployment.warmup_bars == _BOLLINGER_TREND_WINDOW + RECURSIVE_LEAF_CONVERGENCE_MARGIN_BARS
    )
    assert deployment.source_kind == "run"
    assert deployment.source_id == "run_abc"


def test_create_auto_generates_name(tmp_path: Path) -> None:
    """
    Default name follows ``<ticker>-<strategy>-<train_end>``.
    """

    store = tmp_path / "store"
    _materialise_run(store, "run_abc")

    deployment = create_deployment(
        source_kind="run",
        source_id="run_abc",
        store_root=store,
        deployment_id=_DEPLOYMENT_ID,
    )

    parts = deployment.name.split("-")
    assert parts[0] == "SPY"
    assert parts[1] == "AdaptiveBollinger"
    pd.Timestamp(parts[2])


def test_create_honours_explicit_name(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _materialise_run(store, "run_abc")

    deployment = create_deployment(
        source_kind="run",
        source_id="run_abc",
        store_root=store,
        name="my custom label",
        deployment_id=_DEPLOYMENT_ID,
    )
    assert deployment.name == "my custom label"


def test_create_rejects_existing_dir(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _materialise_run(store, "run_abc")
    create_deployment(
        source_kind="run",
        source_id="run_abc",
        store_root=store,
        deployment_id=_DEPLOYMENT_ID,
    )

    with pytest.raises(FileExistsError):
        create_deployment(
            source_kind="run",
            source_id="run_abc",
            store_root=store,
            deployment_id=_DEPLOYMENT_ID,
        )


def test_create_rejects_missing_source(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    with pytest.raises(FileNotFoundError, match="strategy state not found"):
        create_deployment(
            source_kind="run",
            source_id="absent",
            store_root=store,
        )


def test_create_rejects_invalid_warmup(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _materialise_run(store, "run_abc")
    with pytest.raises(ValueError, match="warmup_bars must be"):
        create_deployment(
            source_kind="run",
            source_id="run_abc",
            store_root=store,
            warmup_bars=0,
        )


def test_load_round_trip(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _materialise_run(store, "run_abc")
    created = create_deployment(
        source_kind="run",
        source_id="run_abc",
        store_root=store,
        deployment_id=_DEPLOYMENT_ID,
    )

    loaded = load_deployment(store, _DEPLOYMENT_ID)
    assert loaded == created


def test_read_signals_empty(tmp_path: Path) -> None:
    store = tmp_path / "store"
    _materialise_run(store, "run_abc")
    create_deployment(
        source_kind="run",
        source_id="run_abc",
        store_root=store,
        deployment_id=_DEPLOYMENT_ID,
    )
    assert read_signals(store, _DEPLOYMENT_ID) == ()


def test_resolve_strategy_state_path_for_run(tmp_path: Path) -> None:
    store = tmp_path / "store"
    run_dir = _materialise_run(store, "run_abc")
    resolved = resolve_strategy_state_path("run", "run_abc", store)
    assert resolved == run_dir / EXPERIMENT_STRATEGY_SUBDIR


def test_resolve_strategy_state_path_missing(tmp_path: Path) -> None:
    store = tmp_path / "store"
    with pytest.raises(FileNotFoundError, match="strategy state not found"):
        resolve_strategy_state_path("run", "absent", store)


def test_resolve_strategy_state_path_rejects_unknown_kind(tmp_path: Path) -> None:
    from typing import cast

    from src.orchestration.holdout_eval import SourceKind

    with pytest.raises(ValueError, match="unknown source_kind"):
        resolve_strategy_state_path(cast(SourceKind, "comparison"), "x", tmp_path)


def test_resolve_hpo_study_dir_flat(tmp_path: Path) -> None:
    from src.core.persistence import HPO_SUBDIR
    from src.optimization.tuner import STUDY_DB_FILENAME
    from src.orchestration.deployment import _resolve_hpo_study_dir

    study_dir = tmp_path / HPO_SUBDIR / "study_x"
    study_dir.mkdir(parents=True)
    (study_dir / STUDY_DB_FILENAME).write_text("", encoding="utf-8")
    assert _resolve_hpo_study_dir(tmp_path, "study_x") == study_dir


def test_resolve_hpo_study_dir_study_nested(tmp_path: Path) -> None:
    from src.core.persistence import HPO_SUBDIR
    from src.optimization.tuner import STUDY_DB_FILENAME
    from src.orchestration.deployment import _resolve_hpo_study_dir

    study_dir = tmp_path / "studies" / "main" / HPO_SUBDIR / "study_x"
    study_dir.mkdir(parents=True)
    (study_dir / STUDY_DB_FILENAME).write_text("", encoding="utf-8")
    assert _resolve_hpo_study_dir(tmp_path, "study_x") == study_dir


def test_resolve_hpo_study_dir_absent_returns_flat(tmp_path: Path) -> None:
    from src.core.persistence import HPO_SUBDIR
    from src.orchestration.deployment import _resolve_hpo_study_dir

    resolved = _resolve_hpo_study_dir(tmp_path, "study_x")
    assert resolved == tmp_path / HPO_SUBDIR / "study_x"
    assert not resolved.exists()


def test_recommend_warmup_bars_includes_convergence_margin() -> None:
    """
    AdaptiveBollinger (GARCH leaf) → required + 100 margin.
    """

    strategy = AdaptiveBollingerStrategy(
        window=_BOLLINGER_WINDOW,
        trend_window=_BOLLINGER_TREND_WINDOW,
        garch_p_max=_GARCH_P_MAX,
        garch_q_max=_GARCH_Q_MAX,
    )
    assert strategy.convergence_margin_bars == RECURSIVE_LEAF_CONVERGENCE_MARGIN_BARS
    expected = strategy.required_warmup_bars + RECURSIVE_LEAF_CONVERGENCE_MARGIN_BARS
    assert recommend_warmup_bars(strategy) == expected


def test_recommend_warmup_bars_respects_absolute_floor() -> None:
    """
    A strategy with a tiny required_warmup_bars + zero margin still hits
    the absolute floor — empirically 50 today.
    """

    class _TinyStubStrategy:
        required_warmup_bars: int = 5
        convergence_margin_bars: int = 0

    from typing import cast

    from src.strategies.interface import IStrategy

    derived = recommend_warmup_bars(cast(IStrategy, _TinyStubStrategy()))
    assert derived >= _ABSOLUTE_WARMUP_FLOOR


def test_create_honours_explicit_warmup_bars(tmp_path: Path) -> None:
    """
    Passing ``warmup_bars=<int>`` opts out of auto-derive verbatim.
    """

    store = tmp_path / "store"
    _materialise_run(store, "run_abc")
    deployment = create_deployment(
        source_kind="run",
        source_id="run_abc",
        store_root=store,
        warmup_bars=_CUSTOM_WARMUP,
        deployment_id=_DEPLOYMENT_ID,
    )
    assert deployment.warmup_bars == _CUSTOM_WARMUP
