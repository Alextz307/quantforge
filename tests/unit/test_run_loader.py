"""
Round-trip + error-shape tests for :mod:`src.orchestration.run_loader`.

Behavioural surface:
* ``load_experiment_result`` reconstructs an :class:`ExperimentResult`
  (experiment_id + folds + manifest) byte-equivalent to the writer's
  output (modulo dataclass identity).
* ``load_experiment_config_from_run`` returns the frozen
  :class:`ExperimentConfig` from the run's ``config.yaml``.
* ``load_strategy_from_run_dir`` resolves the registered concrete
  class via ``strategy_registry`` and dispatches to ``cls.load``;
  the returned instance is fully trained and produces signals
  byte-identical to the source strategy.
* Missing dir or missing artifact raises :class:`FileNotFoundError`
  with a pointed message - partial run dirs (mid-crash) must not look
  like analysable runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.core import json_io
from src.core.config import write_frozen_yaml
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_MANIFEST_JSON,
    EXPERIMENT_STRATEGY_SUBDIR,
    FOLD_RESULTS_JSONL,
    RUNS_SUBDIR,
)
from src.orchestration.run_loader import (
    load_experiment_config_from_run,
    load_experiment_result,
    load_strategy_from_run_dir,
    resolve_run_dir,
)
from src.strategies.adaptive_bollinger import AdaptiveBollingerStrategy
from tests.conftest import (
    comparison_curve_seed,
    make_log_return_equity_curve,
    make_stub_experiment_result,
    make_stub_fold_record,
    make_synthetic_ohlcv_df,
)

_STRATEGY_TRAIN_ROWS = 300
_STRATEGY_BOLLINGER_WINDOW = 20
_STRATEGY_BOLLINGER_TREND_WINDOW = 50
_STRATEGY_GARCH_P_MAX = 1
_STRATEGY_GARCH_Q_MAX = 1

_N_FOLDS = 3
_CURVE_LENGTH = 32
_SHARPE = 1.1


def _write_stub_run_dir(run_dir: Path, name: str) -> None:
    """
    Materialise a minimal but valid run dir using the stub helpers.
    """

    run_dir.mkdir(parents=True)
    folds = tuple(
        make_stub_fold_record(
            i,
            sharpe=_SHARPE,
            equity_curve=make_log_return_equity_curve(
                _SHARPE, n=_CURVE_LENGTH, seed=comparison_curve_seed(name, i)
            ),
        )
        for i in range(_N_FOLDS)
    )
    result = make_stub_experiment_result(name, folds=folds)
    json_io.write(run_dir / EXPERIMENT_MANIFEST_JSON, result.manifest.to_dict())
    with (run_dir / FOLD_RESULTS_JSONL).open("w", encoding="utf-8") as f:
        for fold in result.folds:
            f.write(json.dumps(fold.to_dict(), sort_keys=True))
            f.write("\n")


def test_round_trip_recovers_manifest_and_folds(tmp_path: Path) -> None:
    run_dir = tmp_path / "stub_run"
    _write_stub_run_dir(run_dir, "Alpha")

    loaded = load_experiment_result(run_dir)
    assert loaded.experiment_id == "stub_Alpha"
    assert loaded.manifest.name == "Alpha"
    assert len(loaded.folds) == _N_FOLDS
    assert all(fold.sharpe_ratio == _SHARPE for fold in loaded.folds)
    assert all(len(fold.equity_curve) == _CURVE_LENGTH for fold in loaded.folds)


def test_missing_run_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run directory not found"):
        load_experiment_result(tmp_path / "nope")


def test_missing_manifest_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "incomplete"
    run_dir.mkdir()
    (run_dir / FOLD_RESULTS_JSONL).write_text("")
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_experiment_result(run_dir)


def test_missing_folds_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "incomplete"
    run_dir.mkdir()
    # Manifest only (placeholder JSON; loader fails before parsing it).
    (run_dir / EXPERIMENT_MANIFEST_JSON).write_text("{}")
    with pytest.raises(FileNotFoundError, match="fold_results.jsonl"):
        load_experiment_result(run_dir)


def test_load_config_from_run(tmp_path: Path) -> None:
    """
    Frozen ``config.yaml`` round-trips through the loader.
    """

    from src.core.config import load_experiment_config

    run_dir = tmp_path / "with_config"
    run_dir.mkdir()
    cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
    write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, cfg)

    loaded = load_experiment_config_from_run(run_dir)
    assert loaded.name == cfg.name
    assert loaded.data.tickers == cfg.data.tickers


def test_load_config_missing_yaml_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "no_yaml"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=EXPERIMENT_CONFIG_YAML):
        load_experiment_config_from_run(run_dir)


def _materialise_trained_run_dir(
    tmp_path: Path,
) -> tuple[Path, AdaptiveBollingerStrategy, pd.DataFrame]:
    """
    Train AdaptiveBollinger on synthetic bars; persist a minimal run dir.

    Layout matches the experiment runner's output: ``config.yaml`` next to
    a ``strategy_state/`` produced by ``strategy.save()``. Returns the run
    dir, the source strategy (still in memory) and the data the loader
    will use to verify byte-identical signal output.
    """

    from src.core.config import load_experiment_config

    run_dir = tmp_path / "trained"
    run_dir.mkdir()

    cfg = load_experiment_config("config/strategies/adaptive_bollinger.yaml")
    write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, cfg)

    df = make_synthetic_ohlcv_df(n_rows=_STRATEGY_TRAIN_ROWS)
    strategy = AdaptiveBollingerStrategy(
        window=_STRATEGY_BOLLINGER_WINDOW,
        trend_window=_STRATEGY_BOLLINGER_TREND_WINDOW,
        garch_p_max=_STRATEGY_GARCH_P_MAX,
        garch_q_max=_STRATEGY_GARCH_Q_MAX,
    )
    strategy.train(df)
    strategy.save(run_dir / EXPERIMENT_STRATEGY_SUBDIR)
    return run_dir, strategy, df


def test_load_strategy_round_trip(tmp_path: Path) -> None:
    """
    Loader returns a trained strategy whose signals match the source.
    """

    run_dir, source, df = _materialise_trained_run_dir(tmp_path)

    loaded = load_strategy_from_run_dir(run_dir)

    assert type(loaded) is type(source)
    assert loaded.training_metadata is not None
    assert loaded.training_metadata.train_end == source.training_metadata.train_end  # type: ignore[union-attr]

    expected = source.generate_signals(df)
    actual = loaded.generate_signals(df)
    pd.testing.assert_series_equal(actual, expected)


def test_load_strategy_missing_run_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run directory not found"):
        load_strategy_from_run_dir(tmp_path / "absent")


def test_load_strategy_missing_state_dir_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "no_state"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="strategy state directory not found"):
        load_strategy_from_run_dir(run_dir)


_RUN_ID = "20260522_143448_AdaptiveBollinger_f8bee0a_b102397e"
_STUDY_NAME = "main"


def test_resolve_run_dir_flat(tmp_path: Path) -> None:
    run_dir = tmp_path / RUNS_SUBDIR / _RUN_ID
    run_dir.mkdir(parents=True)
    assert resolve_run_dir(tmp_path, _RUN_ID) == run_dir


def test_resolve_run_dir_study_nested(tmp_path: Path) -> None:
    run_dir = tmp_path / "studies" / _STUDY_NAME / RUNS_SUBDIR / _RUN_ID
    run_dir.mkdir(parents=True)
    assert resolve_run_dir(tmp_path, _RUN_ID) == run_dir


def test_resolve_run_dir_absent_returns_flat_path(tmp_path: Path) -> None:
    resolved = resolve_run_dir(tmp_path, _RUN_ID)
    assert resolved == tmp_path / RUNS_SUBDIR / _RUN_ID
    assert not resolved.exists()
