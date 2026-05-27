"""Direct tests for :class:`TrialCallback`.

The tuner's own smoke test exercises the callback indirectly; here we
construct a callback in isolation and invoke it against controlled
``optuna.trial.FrozenTrial`` fixtures so the jsonl append + best-config
refresh rules are verified independently of Optuna's optimize loop.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import optuna
import yaml

from src.core.config import ExperimentConfig
from src.optimization.checkpointing import (
    BEST_CONFIG_YAML_NAME,
    TRIALS_JSONL_NAME,
    TrialCallback,
)

_SPY_DATA = {
    "source": "csv",
    "tickers": ["SPY"],
    "start": "2020-01-01",
    "end": "2024-01-01",
    "interval": "daily",
}


def _base_cfg() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "cb_test",
            "seed": 42,
            "data": _SPY_DATA,
            "strategy": {"name": "AdaptiveBollinger", "params": {}},
        }
    )


def _make_callback(tmp_path: Path) -> TrialCallback:
    return TrialCallback(experiment_cfg=_base_cfg(), study_dir=tmp_path)


def _run_study_with_callback(tmp_path: Path, values: list[float]) -> optuna.Study:
    """Drive a study that asks AdaptiveBollinger's search space and
    returns caller-specified values per trial.

    Realistic enough that ``sample_trial_params`` — which the best-config
    refresh replays via :class:`FixedTrial` — sees the same parameter
    surface it would see in a live tuner.
    """
    from src.optimization.sampling import sample_trial_params

    cfg = _base_cfg()
    study = optuna.create_study(direction="maximize")
    callback = _make_callback(tmp_path)
    value_iter = iter(values)

    def objective(trial: optuna.Trial) -> float:
        sample_trial_params(cfg, trial)
        return next(value_iter)

    study.optimize(objective, n_trials=len(values), callbacks=[callback])
    return study


class TestJsonlAppend:
    def test_one_line_per_trial(self, tmp_path: Path) -> None:
        _run_study_with_callback(tmp_path, [0.1, 0.5, 0.3])
        lines = (tmp_path / TRIALS_JSONL_NAME).read_text().splitlines()
        assert len(lines) == 3

    def test_each_line_is_valid_json_with_expected_keys(self, tmp_path: Path) -> None:
        import json

        _run_study_with_callback(tmp_path, [0.1])
        line = (tmp_path / TRIALS_JSONL_NAME).read_text().splitlines()[0]
        record = json.loads(line)
        for key in (
            "number",
            "state",
            "value",
            "params",
            "user_attrs",
            "datetime_start",
            "datetime_complete",
        ):
            assert key in record
        assert record["state"] == "COMPLETE"
        assert record["value"] == 0.1


class TestBestConfigRefresh:
    def test_best_config_written_on_first_complete_trial(self, tmp_path: Path) -> None:
        _run_study_with_callback(tmp_path, [0.5])
        assert (tmp_path / BEST_CONFIG_YAML_NAME).is_file()

    def test_best_config_updated_on_new_best(self, tmp_path: Path) -> None:
        _run_study_with_callback(tmp_path, [0.1, 0.5, 0.3])
        raw = yaml.safe_load((tmp_path / BEST_CONFIG_YAML_NAME).read_text())
        cfg = ExperimentConfig.model_validate(raw)
        assert cfg.strategy.name == "AdaptiveBollinger"

    def test_best_config_preserves_original_name(self, tmp_path: Path) -> None:
        _run_study_with_callback(tmp_path, [0.5])
        raw = yaml.safe_load((tmp_path / BEST_CONFIG_YAML_NAME).read_text())
        assert raw["name"] == "cb_test"


class TestNonCompleteStatesSkipBestRefresh:
    def test_failed_trial_does_not_write_best_config(self, tmp_path: Path) -> None:
        """A trial that raises is logged in jsonl but not promoted to best.

        Uses ``catch=(RuntimeError,)`` so Optuna marks the trial FAILED
        (vs. re-raising out of optimize entirely) and the callback fires.
        """
        from src.optimization.sampling import sample_trial_params

        cfg = _base_cfg()
        study = optuna.create_study(direction="maximize")
        callback = _make_callback(tmp_path)

        def failing_objective(trial: optuna.Trial) -> float:
            sample_trial_params(cfg, trial)
            raise RuntimeError("fixture: trial fails")

        study.optimize(
            failing_objective,
            n_trials=1,
            callbacks=[callback],
            catch=(RuntimeError,),
        )

        assert (tmp_path / TRIALS_JSONL_NAME).is_file()
        lines = (tmp_path / TRIALS_JSONL_NAME).read_text().splitlines()
        assert len(lines) == 1
        assert not (tmp_path / BEST_CONFIG_YAML_NAME).exists()


class TestCallbackDirectInvocation:
    """Tests the callback surface without going through study.optimize."""

    def test_pruned_trial_skips_best_config_but_logs_jsonl(self, tmp_path: Path) -> None:
        cb = _make_callback(tmp_path)

        class _FakeStudy:
            @property
            def best_trial(self) -> optuna.trial.FrozenTrial:
                raise ValueError("no completed trials")

        pruned = optuna.trial.create_trial(
            params={"bollinger_window": 20},
            distributions={"bollinger_window": optuna.distributions.IntDistribution(10, 50)},
            value=None,
            state=optuna.trial.TrialState.PRUNED,
        )
        # create_trial doesn't set datetimes; attach them manually.
        object.__setattr__(pruned, "datetime_start", datetime(2026, 1, 1, 12, 0))
        object.__setattr__(pruned, "datetime_complete", datetime(2026, 1, 1, 12, 5))

        cb(_FakeStudy(), pruned)  # type: ignore[arg-type]

        assert (tmp_path / TRIALS_JSONL_NAME).is_file()
        assert not (tmp_path / BEST_CONFIG_YAML_NAME).exists()


class TestEndToEndPruning:
    """Integration-lite: study.optimize with an objective that raises
    ``TrialPruned`` mid-flight mirrors how the real LSTM/XGBoost leaves
    abort trials under a live pruner. The callback must log the PRUNED
    record without promoting it to best.
    """

    def test_pruned_exception_flows_through_callback(self, tmp_path: Path) -> None:
        import json

        from src.optimization.sampling import sample_trial_params

        cfg = _base_cfg()
        study = optuna.create_study(direction="maximize")
        callback = _make_callback(tmp_path)

        def objective(trial: optuna.Trial) -> float:
            sample_trial_params(cfg, trial)
            if trial.number == 0:
                raise optuna.TrialPruned("fixture: prune trial 0")
            return 0.3

        study.optimize(objective, n_trials=2, callbacks=[callback])

        lines = (tmp_path / TRIALS_JSONL_NAME).read_text().splitlines()
        assert len(lines) == 2
        states = [json.loads(line)["state"] for line in lines]
        assert states == ["PRUNED", "COMPLETE"]
        assert (tmp_path / BEST_CONFIG_YAML_NAME).is_file()


class TestBestValueCache:
    """Incremental best tracking avoids scanning ``study.best_trial`` on
    every non-improving COMPLETE trial (would otherwise be O(n²) for a
    study whose best improves often).
    """

    def test_cache_populated_after_first_complete_trial(self, tmp_path: Path) -> None:
        callback = _make_callback(tmp_path)
        _run_study_with_callback_from(tmp_path, [0.5], callback=callback)
        assert callback._last_best_value == 0.5

    def test_cache_advances_only_when_best_improves(self, tmp_path: Path) -> None:
        callback = _make_callback(tmp_path)
        _run_study_with_callback_from(tmp_path, [0.1, 0.5, 0.3, 0.4], callback=callback)
        assert callback._last_best_value == 0.5

    def test_non_improving_trial_skips_study_best_trial_scan(self, tmp_path: Path) -> None:
        """After a best is seen, a worse COMPLETE trial must short-circuit
        without touching ``study.best_trial`` — we detect this by patching
        the study with a ``best_trial`` that raises if accessed.
        """
        callback = _make_callback(tmp_path)
        _run_study_with_callback_from(tmp_path, [0.5], callback=callback)
        assert callback._last_best_value == 0.5

        worse = optuna.trial.create_trial(
            params={"bollinger_window": 20},
            distributions={"bollinger_window": optuna.distributions.IntDistribution(10, 50)},
            value=0.1,
            state=optuna.trial.TrialState.COMPLETE,
        )
        object.__setattr__(worse, "datetime_start", datetime(2026, 1, 1, 12, 0))
        object.__setattr__(worse, "datetime_complete", datetime(2026, 1, 1, 12, 5))

        class _TrapStudy:
            @property
            def best_trial(self) -> optuna.trial.FrozenTrial:
                raise AssertionError("callback should short-circuit without scanning best_trial")

        callback(_TrapStudy(), worse)  # type: ignore[arg-type]


def _run_study_with_callback_from(
    tmp_path: Path, values: list[float], *, callback: TrialCallback
) -> optuna.Study:
    """Variant of ``_run_study_with_callback`` that takes a pre-built
    callback — needed by tracker tests that read callback state after.
    """
    from src.optimization.sampling import sample_trial_params

    cfg = _base_cfg()
    study = optuna.create_study(direction="maximize")
    value_iter = iter(values)

    def objective(trial: optuna.Trial) -> float:
        sample_trial_params(cfg, trial)
        return next(value_iter)

    study.optimize(objective, n_trials=len(values), callbacks=[callback])
    return study
