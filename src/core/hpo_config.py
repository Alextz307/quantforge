"""
Typed config for ``StrategyTuner`` / ``experiment tune``.

Kept deliberately separate from :class:`ExperimentConfig`:

* ``Experiment.run()`` consumes none of these fields - embedding them in
  every ``run`` config would force unused HPO knobs into every YAML.
* The tuner takes ``(experiment_cfg, hpo_cfg)`` explicitly; a single
  experiment config can be reused across multiple studies (different
  sampler / objective / n_trials) without rewriting the base YAML.

StrEnum choices mirror Optuna's built-in sampler / pruner classes so YAML
values like ``sampler: tpe`` round-trip without hand-written unions.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.config import load_yaml_config


class SamplerKind(StrEnum):
    """
    Optuna sampler selector for :func:`build_sampler`.
    """

    TPE = "tpe"
    RANDOM = "random"
    CMAES = "cmaes"
    QMC = "qmc"


class PrunerKind(StrEnum):
    """
    Optuna pruner selector for :func:`build_pruner`.

    ``NONE`` maps to ``optuna.pruners.NopPruner`` - keeps the field
    uniformly enum-typed instead of carrying an ``Optional`` in every
    downstream signature.
    """

    MEDIAN = "median"
    HYPERBAND = "hyperband"
    PERCENTILE = "percentile"
    NONE = "none"


class ObjectiveKind(StrEnum):
    """
    Which aggregate metric the study maximises.

    All three read from ``ExperimentResult.aggregate_metrics`` keys - the
    objective layer is a thin adapter, no per-fold maths lives in the
    tuner.
    """

    SHARPE = "sharpe"
    CALMAR = "calmar"
    SORTINO_MINUS_DRAWDOWN = "sortino_minus_drawdown"


class HPOConfig(BaseModel):
    """
    Typed knobs for one Optuna study.

    ``study_name`` doubles as the on-disk directory under
    ``experiment_results/hpo/<study_name>/`` - keep it filesystem-safe.
    Resume works by re-running with the same ``study_name`` against the
    same SQLite file; Optuna replays completed trials automatically.

    ``n_jobs`` is a positive integer. The CLI layer resolves a
    convenience ``-1`` / ``"auto"`` to ``os.cpu_count()`` before
    constructing this model - keeping the pydantic side strictly ``ge=1``
    means the in-process invariant "parallelism count is a known int"
    holds everywhere downstream.

    ``seed`` is the sampler seed; it is NOT the experiment seed. Each
    trial seeds its own experiment from ``ExperimentConfig.seed``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    study_name: str = Field(min_length=1)
    n_trials: int = Field(default=50, ge=1)
    n_jobs: int = Field(default=1, ge=1)
    sampler: SamplerKind = SamplerKind.TPE
    pruner: PrunerKind = PrunerKind.MEDIAN
    objective: ObjectiveKind = ObjectiveKind.SHARPE
    timeout_s: float | None = Field(default=None, gt=0.0)
    seed: int = 42

    @model_validator(mode="after")
    def _validate_study_name(self) -> Self:
        if "/" in self.study_name or "\\" in self.study_name:
            raise ValueError(
                f"hpo.study_name must not contain path separators; got "
                f"{self.study_name!r}. Use a plain identifier like "
                f"'spy_bollinger_sharpe' - the runner creates the "
                f"directory under experiment_results/hpo/ automatically."
            )
        return self


def load_hpo_config(path: str | Path) -> HPOConfig:
    """
    Read a YAML file and validate it as an :class:`HPOConfig`.

    Delegates to :func:`src.core.config.load_yaml_config` so the
    "not found / empty / validation failed" framing stays identical
    across every ``experiment`` subcommand.
    """

    return load_yaml_config(path, HPOConfig, "hpo")
