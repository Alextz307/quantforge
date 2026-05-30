"""
StrategyTuner integration tests.

Mocks ``build_experiment`` + ``aggregate_folds`` so the tuner's own
logic (config materialisation, SQLite study creation, trial-to-objective
plumbing, resume, best-config refresh) is exercised without paying for
real ML training. The model/strategy layers have their own tests -
repeating them here would just make the suite slow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import optuna
import pytest

from src.analysis.metrics_aggregator import AggregateStats
from src.core.config import ExperimentConfig
from src.core.hpo_config import HPOConfig, ObjectiveKind, SamplerKind
from src.optimization import tuner as tuner_mod
from src.optimization.checkpointing import BEST_CONFIG_YAML_NAME, TRIALS_JSONL_NAME
from src.optimization.tuner import (
    EXPERIMENT_CONFIG_YAML,
    HPO_CONFIG_YAML,
    USER_ATTR_EXPERIMENT_ID,
    StrategyTuner,
)
from src.orchestration.experiment import RunOptions
from tests.conftest import make_stub_aggregate_stats

if TYPE_CHECKING:
    from collections.abc import Callable

_SPY_DATA = {
    "source": "csv",
    "tickers": ["SPY"],
    "start": "2020-01-01",
    "end": "2024-01-01",
    "interval": "daily",
}
_TARGET_WINDOW = 30
_SHARPE_PENALTY_PER_UNIT = 0.05


class _FakeExperiment:
    """
    Stand-in for a wired ``Experiment`` - ``run()`` returns a fake result.
    """

    def __init__(self, experiment_id: str) -> None:
        self._experiment_id = experiment_id

    def run(self, options: RunOptions | None = None) -> object:
        class _Result:
            experiment_id = self._experiment_id
            folds: tuple[object, ...] = ()
            manifest = None

        return _Result()


def _build_base_cfg() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "tuner_test",
            "seed": 42,
            "data": _SPY_DATA,
            "strategy": {
                "name": "AdaptiveBollinger",
                "params": {},
            },
        }
    )


def _hpo_cfg(study_name: str, n_trials: int) -> HPOConfig:
    return HPOConfig.model_validate(
        {
            "study_name": study_name,
            "n_trials": n_trials,
            "sampler": SamplerKind.RANDOM.value,
            "objective": ObjectiveKind.SHARPE.value,
            "seed": 1,
        }
    )


@pytest.fixture
def mocked_tuner_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], None]:
    """
    Monkeypatch ``build_experiment`` + ``aggregate_folds``.

    Returns a callable that resets the per-test trial counter so each
    test starts from the same deterministic state.
    """

    counter = {"n": 0}

    def _fake_build(cfg: ExperimentConfig) -> _FakeExperiment:
        return _FakeExperiment(experiment_id=f"fake_exp_{counter['n']}")

    def _fake_aggregate(folds: tuple[object, ...]) -> AggregateStats:
        trial_cfg_window = _window_for_trial(counter["n"])
        sharpe = 1.0 - abs(trial_cfg_window - _TARGET_WINDOW) * _SHARPE_PENALTY_PER_UNIT
        counter["n"] += 1
        return make_stub_aggregate_stats(sharpe=sharpe)

    monkeypatch.setattr(tuner_mod, "build_experiment", _fake_build)
    monkeypatch.setattr(tuner_mod, "aggregate_folds", _fake_aggregate)

    def _reset() -> None:
        counter["n"] = 0

    return _reset


def _window_for_trial(n: int) -> int:
    """
    Deterministic rotation across the AdaptiveBollinger window search space
    [10, 50] giving monotone coverage around the peak."""

    return 10 + (n * 5) % 41


class TestStrategyTunerSmoke:
    def test_run_produces_study_with_n_trials(
        self, tmp_path: Path, mocked_tuner_env: Callable[[], None]
    ) -> None:
        mocked_tuner_env()
        tuner = StrategyTuner(
            experiment_cfg=_build_base_cfg(),
            hpo_cfg=_hpo_cfg("smoke", n_trials=3),
            store_root=tmp_path,
        )
        study = tuner.run()
        assert len(study.trials) == 3
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(completed) == 3

    def test_persists_configs_once(
        self, tmp_path: Path, mocked_tuner_env: Callable[[], None]
    ) -> None:
        mocked_tuner_env()
        tuner = StrategyTuner(
            experiment_cfg=_build_base_cfg(),
            hpo_cfg=_hpo_cfg("persist", n_trials=2),
            store_root=tmp_path,
        )
        tuner.run()
        assert (tuner.study_dir / EXPERIMENT_CONFIG_YAML).is_file()
        assert (tuner.study_dir / HPO_CONFIG_YAML).is_file()

    def test_user_attr_records_experiment_id(
        self, tmp_path: Path, mocked_tuner_env: Callable[[], None]
    ) -> None:
        mocked_tuner_env()
        tuner = StrategyTuner(
            experiment_cfg=_build_base_cfg(),
            hpo_cfg=_hpo_cfg("userattr", n_trials=2),
            store_root=tmp_path,
        )
        study = tuner.run()
        for trial in study.trials:
            assert USER_ATTR_EXPERIMENT_ID in trial.user_attrs
            assert trial.user_attrs[USER_ATTR_EXPERIMENT_ID].startswith("fake_exp_")


class TestStrategyTunerBestConfig:
    def test_best_config_yaml_written_and_round_trips(
        self, tmp_path: Path, mocked_tuner_env: Callable[[], None]
    ) -> None:
        mocked_tuner_env()
        tuner = StrategyTuner(
            experiment_cfg=_build_base_cfg(),
            hpo_cfg=_hpo_cfg("best", n_trials=5),
            store_root=tmp_path,
        )
        tuner.run()

        best_path = tuner.study_dir / BEST_CONFIG_YAML_NAME
        assert best_path.is_file()
        import yaml

        with best_path.open() as f:
            raw = yaml.safe_load(f)
        reloaded = ExperimentConfig.model_validate(raw)
        assert reloaded.strategy.name == "AdaptiveBollinger"
        assert reloaded.name == "tuner_test"

    def test_trials_jsonl_line_per_trial(
        self, tmp_path: Path, mocked_tuner_env: Callable[[], None]
    ) -> None:
        mocked_tuner_env()
        tuner = StrategyTuner(
            experiment_cfg=_build_base_cfg(),
            hpo_cfg=_hpo_cfg("jsonl", n_trials=4),
            store_root=tmp_path,
        )
        tuner.run()

        lines = (tuner.study_dir / TRIALS_JSONL_NAME).read_text().splitlines()
        assert len(lines) == 4


class TestStrategyTunerResume:
    def test_second_run_extends_study(
        self, tmp_path: Path, mocked_tuner_env: Callable[[], None]
    ) -> None:
        mocked_tuner_env()
        cfg = _build_base_cfg()
        first = StrategyTuner(
            experiment_cfg=cfg,
            hpo_cfg=_hpo_cfg("resume", n_trials=2),
            store_root=tmp_path,
        )
        first.run()
        assert (first.study_dir / TRIALS_JSONL_NAME).is_file()

        mocked_tuner_env()
        second = StrategyTuner(
            experiment_cfg=cfg,
            hpo_cfg=_hpo_cfg("resume", n_trials=3),
            store_root=tmp_path,
        )
        study = second.run()
        # Optuna's n_trials is ADDITIONAL per call - 2 + 3 = 5.
        assert len(study.trials) == 5


class TestStrategyTunerPruning:
    """
    ``optuna.TrialPruned`` raised from the objective (as LSTM/XGBoost
    leaves do under a live pruner) must be caught by Optuna - marking the
    trial PRUNED instead of FAILED - and must not short-circuit the rest
    of the study.
    """

    def test_pruned_trial_recorded_and_study_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.optimization.tuner as tm

        # Trial 0 completes; trial 1 raises TrialPruned from aggregate_folds -
        # the closest mid-trial stand-in without plumbing through a real model.
        counter = {"n": 0}

        def _fake_build(cfg: ExperimentConfig) -> _FakeExperiment:
            return _FakeExperiment(experiment_id=f"prune_exp_{counter['n']}")

        def _fake_aggregate(folds: tuple[object, ...]) -> AggregateStats:
            n = counter["n"]
            counter["n"] += 1
            if n == 1:
                raise optuna.TrialPruned("fixture: prune trial 1")
            return make_stub_aggregate_stats(
                sharpe=0.4, max_drawdown_worst=-0.05, total_return_mean=0.02
            )

        monkeypatch.setattr(tm, "build_experiment", _fake_build)
        monkeypatch.setattr(tm, "aggregate_folds", _fake_aggregate)

        tuner = StrategyTuner(
            experiment_cfg=_build_base_cfg(),
            hpo_cfg=_hpo_cfg("pruning", n_trials=3),
            store_root=tmp_path,
        )
        study = tuner.run()
        assert len(study.trials) == 3
        states = [t.state for t in study.trials]
        assert optuna.trial.TrialState.PRUNED in states
        assert optuna.trial.TrialState.COMPLETE in states


class TestStrategyTunerConfigDrift:
    def test_second_run_with_different_config_rejected(
        self, tmp_path: Path, mocked_tuner_env: Callable[[], None]
    ) -> None:
        mocked_tuner_env()
        cfg_a = _build_base_cfg()
        StrategyTuner(
            experiment_cfg=cfg_a,
            hpo_cfg=_hpo_cfg("drift", n_trials=1),
            store_root=tmp_path,
        ).run()

        payload = cfg_a.model_dump(mode="json")
        payload["seed"] = 999
        cfg_b = ExperimentConfig.model_validate(payload)

        mocked_tuner_env()
        with pytest.raises(ValueError, match="different experiment config"):
            StrategyTuner(
                experiment_cfg=cfg_b,
                hpo_cfg=_hpo_cfg("drift", n_trials=1),
                store_root=tmp_path,
            ).run()
