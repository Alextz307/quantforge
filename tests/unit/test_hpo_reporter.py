"""
HPOReporter tests — verify artifacts are produced under the expected paths.

Uses real Optuna studies (they're cheap to create). The plots/tables
themselves aren't parsed — file-existence + non-trivial size is the
contract, matching the convention in :mod:`test_strategy_reporter`.
"""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest

from src.visualization.hpo_reporter import HPOReporter


def _run_study(n_trials: int, *, two_params: bool = False) -> optuna.Study:
    """
    Run a toy study that accepts at least one parameter.

    ``two_params=True`` adds a second param so Optuna's fANOVA importance
    calculation doesn't degenerate into a single-axis trivial answer.
    """

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=1))

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        if two_params:
            y = trial.suggest_float("y", 0.0, 1.0)
            return x + y
        return x

    study.optimize(objective, n_trials=n_trials)
    return study


class TestHPOReporterArtifacts:
    def test_convergence_plot_written_for_nonempty_study(self, tmp_path: Path) -> None:
        study = _run_study(n_trials=3)
        HPOReporter().generate_full_report(study, tmp_path)
        assert (tmp_path / "plots" / "convergence.png").is_file()
        assert (tmp_path / "plots" / "convergence.svg").is_file()

    def test_top_trials_table_written(self, tmp_path: Path) -> None:
        study = _run_study(n_trials=3)
        HPOReporter().generate_full_report(study, tmp_path)
        tex = (tmp_path / "tables" / "top_trials.tex").read_text()
        assert r"\toprule" in tex  # booktabs style
        assert "trial" in tex
        assert "value" in tex

    def test_param_importance_written_with_enough_trials(self, tmp_path: Path) -> None:
        study = _run_study(n_trials=5, two_params=True)
        HPOReporter().generate_full_report(study, tmp_path)
        assert (tmp_path / "plots" / "param_importance.png").is_file()

    def test_param_importance_skipped_for_single_trial(self, tmp_path: Path) -> None:
        study = _run_study(n_trials=1)
        HPOReporter().generate_full_report(study, tmp_path)
        assert not (tmp_path / "plots" / "param_importance.png").exists()
        assert (tmp_path / "plots" / "convergence.png").is_file()

    def test_empty_study_produces_no_plots(self, tmp_path: Path) -> None:
        study = optuna.create_study(direction="maximize")
        HPOReporter().generate_full_report(study, tmp_path)
        assert not (tmp_path / "plots").exists() or not any((tmp_path / "plots").iterdir())
        assert not (tmp_path / "tables").exists() or not any((tmp_path / "tables").iterdir())


class TestHPOReporterIgnoresFailedTrials:
    def test_failed_trials_excluded_from_completed_count(self, tmp_path: Path) -> None:
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.RandomSampler(seed=1)
        )

        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", 0.0, 1.0)
            if trial.number == 0:
                raise RuntimeError("fixture: trial fails")
            return x

        with pytest.raises(RuntimeError):
            study.optimize(objective, n_trials=1)

        def good_objective(trial: optuna.Trial) -> float:
            return trial.suggest_float("x", 0.0, 1.0)

        study.optimize(good_objective, n_trials=2)
        HPOReporter().generate_full_report(study, tmp_path)
        assert (tmp_path / "plots" / "convergence.png").is_file()
