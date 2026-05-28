"""
Tests for the ``.save_complete`` marker backfill migration.

Reproduces the legacy condition (a complete save tree with the markers
stripped, as models persisted before the marker existed look on disk) and
asserts the backfill re-certifies and re-marks them so they load again — and
that a genuinely corrupt save is left untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.backfill_save_markers import backfill_run, backfill_store
from src.core.config import load_experiment_config, write_frozen_yaml
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_STRATEGY_SUBDIR,
    SAVE_COMPLETE_MARKER,
)
from src.orchestration.run_loader import load_strategy_from_run_dir
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from tests.conftest import make_synthetic_ohlcv_df

_TRAIN_ROWS = 300
_BOLLINGER_WINDOW = 20
_BOLLINGER_TREND_WINDOW = 50
_GARCH_P_MAX = 1
_GARCH_Q_MAX = 1


def _materialise_run(tmp_path: Path) -> Path:
    """
    Train + save an AdaptiveBollinger run, then strip every marker.

    The stripped tree is byte-identical to a model persisted before the
    marker mechanism existed.
    """

    run_dir = tmp_path / "trained"
    run_dir.mkdir()
    cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
    write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, cfg)
    strategy = AdaptiveBollingerStrategy(
        window=_BOLLINGER_WINDOW,
        trend_window=_BOLLINGER_TREND_WINDOW,
        garch_p_max=_GARCH_P_MAX,
        garch_q_max=_GARCH_Q_MAX,
    )
    strategy.train(make_synthetic_ohlcv_df(n_rows=_TRAIN_ROWS))
    strategy.save(run_dir / EXPERIMENT_STRATEGY_SUBDIR)
    for marker in (run_dir / EXPERIMENT_STRATEGY_SUBDIR).rglob(SAVE_COMPLETE_MARKER):
        marker.unlink()
    return run_dir


def test_legacy_run_fails_to_load_then_backfill_fixes_it(tmp_path: Path) -> None:
    run_dir = _materialise_run(tmp_path)

    with pytest.raises(FileNotFoundError, match="save_complete"):
        load_strategy_from_run_dir(run_dir)

    result = backfill_run(run_dir, dry_run=False)

    assert result is not None
    assert result.status == "backfilled"
    # strategy_state + the garch leaf both get a marker.
    assert result.marked == 2
    load_strategy_from_run_dir(run_dir)


def test_dry_run_reports_without_writing(tmp_path: Path) -> None:
    run_dir = _materialise_run(tmp_path)

    result = backfill_run(run_dir, dry_run=True)

    assert result is not None
    assert result.status == "planned"
    assert not list((run_dir / EXPERIMENT_STRATEGY_SUBDIR).rglob(SAVE_COMPLETE_MARKER))


def test_already_marked_run_is_skipped(tmp_path: Path) -> None:
    run_dir = _materialise_run(tmp_path)
    backfill_run(run_dir, dry_run=False)

    assert backfill_run(run_dir, dry_run=False) is None


def test_corrupt_save_is_left_untouched(tmp_path: Path) -> None:
    run_dir = _materialise_run(tmp_path)
    leaf_weights = next((run_dir / EXPERIMENT_STRATEGY_SUBDIR).rglob("weights.json"))
    leaf_weights.write_text("{ this is not valid json", encoding="utf-8")

    result = backfill_run(run_dir, dry_run=False)

    assert result is not None
    assert result.status == "failed"
    assert not list((run_dir / EXPERIMENT_STRATEGY_SUBDIR).rglob(SAVE_COMPLETE_MARKER))


def test_backfill_store_walks_nested_runs(tmp_path: Path) -> None:
    store = tmp_path / "store"
    nested = store / "studies" / "main" / "runs" / "legacy_run"
    nested.mkdir(parents=True)
    materialised = _materialise_run(tmp_path)
    (nested / EXPERIMENT_CONFIG_YAML).write_bytes(
        (materialised / EXPERIMENT_CONFIG_YAML).read_bytes()
    )
    src_state = materialised / EXPERIMENT_STRATEGY_SUBDIR
    dst_state = nested / EXPERIMENT_STRATEGY_SUBDIR
    dst_state.mkdir()
    for item in src_state.rglob("*"):
        target = dst_state / item.relative_to(src_state)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.write_bytes(item.read_bytes())

    results = backfill_store(store, dry_run=False)

    assert len(results) == 1
    assert results[0].status == "backfilled"
    load_strategy_from_run_dir(nested)
