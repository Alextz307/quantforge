"""
Behavioural tests for the seeded fANOVA importance helper.

Real Optuna studies are cheap to build, so each test runs a toy study
rather than mocking the evaluator.
"""

from __future__ import annotations

import optuna
import pytest
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator, get_param_importances
from optuna.trial import TrialState, create_trial

from src.optimization.importance import param_importances

_IMPORTANCE_TRIALS = 10
_SAMPLER_SEED = 1
_SECOND_EVALUATOR_SEED = 0
_DOMINANT_PARAM = "x"
_NOISE_PARAM = "y"
# Wide enough that x's importance dominates y's by a large margin (~0.81 fANOVA
# on the seed=1 draw); the directional-agreement test asserts a bare max() over
# the importances, so shrinking this could silently flip the ranking.
_DOMINANT_COEF = 5.0
_PARAM_BOUNDS = (0.0, 1.0)
_NORMALIZE_TOLERANCE = 1e-6


def _two_param_study(n_trials: int, *, dominant: bool = False) -> optuna.Study:
    """
    Two-param study; ``dominant`` makes ``x`` drive the objective and ``y`` noise.

    The directional-agreement test uses ``dominant`` for an unambiguous x-over-y
    ranking; the balanced variant is enough for the value-shape invariants.
    """

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=_SAMPLER_SEED)
    )

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float(_DOMINANT_PARAM, *_PARAM_BOUNDS)
        y = trial.suggest_float(_NOISE_PARAM, *_PARAM_BOUNDS)
        return _DOMINANT_COEF * x if dominant else x + y

    study.optimize(objective, n_trials=n_trials)
    return study


def test_nonnegative_and_normalized() -> None:
    importances = param_importances(_two_param_study(_IMPORTANCE_TRIALS))
    assert importances
    assert all(score >= 0.0 for score in importances.values())
    assert sum(importances.values()) == pytest.approx(1.0, abs=_NORMALIZE_TOLERANCE)


def test_stable_across_repeated_calls() -> None:
    study = _two_param_study(_IMPORTANCE_TRIALS)
    assert param_importances(study) == param_importances(study)


def test_uses_complete_trials_only() -> None:
    study = _two_param_study(_IMPORTANCE_TRIALS)
    before = param_importances(study)
    distributions: dict[str, BaseDistribution] = {
        _DOMINANT_PARAM: FloatDistribution(*_PARAM_BOUNDS),
        _NOISE_PARAM: FloatDistribution(*_PARAM_BOUNDS),
    }
    midpoint = sum(_PARAM_BOUNDS) / len(_PARAM_BOUNDS)
    study.add_trial(
        create_trial(
            state=TrialState.FAIL,
            params={_DOMINANT_PARAM: midpoint, _NOISE_PARAM: midpoint},
            distributions=distributions,
        )
    )
    assert param_importances(study) == before


def test_directional_agreement_with_second_evaluator() -> None:
    study = _two_param_study(_IMPORTANCE_TRIALS, dominant=True)
    fanova = param_importances(study)
    mdi = get_param_importances(
        study,
        evaluator=MeanDecreaseImpurityImportanceEvaluator(seed=_SECOND_EVALUATOR_SEED),
    )
    assert max(fanova, key=lambda name: fanova[name]) == _DOMINANT_PARAM
    assert max(mdi, key=lambda name: mdi[name]) == _DOMINANT_PARAM
