"""
Dispatch + seeding tests for :mod:`src.optimization.samplers` and
:mod:`src.optimization.pruners`.

These are shallow factories — the Optuna classes themselves are
upstream-tested. What we verify here is (a) every enum value wires to the
correct class, and (b) samplers respect the seed so two HPOConfigs with
the same ``seed`` produce reproducible first-trial draws.
"""

from __future__ import annotations

import optuna
import pytest
from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PercentilePruner,
)
from optuna.samplers import (
    BaseSampler,
    CmaEsSampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)

from src.core.hpo_config import PrunerKind, SamplerKind
from src.optimization.pruners import build_pruner
from src.optimization.samplers import build_sampler

_SEED = 42
_SEARCH_LO = 0.0
_SEARCH_HI = 1.0


class TestBuildSamplerDispatch:
    @pytest.mark.parametrize(
        "kind,expected_cls",
        [
            (SamplerKind.TPE, TPESampler),
            (SamplerKind.RANDOM, RandomSampler),
            (SamplerKind.CMAES, CmaEsSampler),
            (SamplerKind.QMC, QMCSampler),
        ],
    )
    def test_dispatch(self, kind: SamplerKind, expected_cls: type[BaseSampler]) -> None:
        sampler = build_sampler(kind, seed=_SEED)
        assert isinstance(sampler, expected_cls)


class TestSamplerSeedReproducibility:
    @pytest.mark.parametrize(
        "kind",
        [SamplerKind.TPE, SamplerKind.RANDOM, SamplerKind.QMC],
    )
    def test_same_seed_same_first_draw(self, kind: SamplerKind) -> None:
        """
        A fresh study + same-seed sampler draws the same first value.

        CMA-ES is skipped here: it needs at least two numerical
        parameters to warm up, which makes the one-parameter smoke
        fixture a bad match. Seed-reproducibility for CMA-ES is
        Optuna's own test surface.
        """

        def _first_value() -> float:
            study = optuna.create_study(
                sampler=build_sampler(kind, seed=_SEED),
                direction="maximize",
            )
            trial = study.ask()
            return trial.suggest_float("x", _SEARCH_LO, _SEARCH_HI)

        assert _first_value() == _first_value()


class TestBuildPrunerDispatch:
    @pytest.mark.parametrize(
        "kind,expected_cls",
        [
            (PrunerKind.MEDIAN, MedianPruner),
            (PrunerKind.HYPERBAND, HyperbandPruner),
            (PrunerKind.PERCENTILE, PercentilePruner),
            (PrunerKind.NONE, NopPruner),
        ],
    )
    def test_dispatch(self, kind: PrunerKind, expected_cls: type[BasePruner]) -> None:
        pruner = build_pruner(kind)
        assert isinstance(pruner, expected_cls)
