"""
Factory mapping :class:`SamplerKind` to Optuna ``BaseSampler`` instances.

Every sampler is instantiated with an explicit seed so reruns with the
same ``HPOConfig.seed`` reproduce bit-for-bit — required for thesis
defensibility. CMA-ES requires numerical parameters; when a study has
categorical dimensions (e.g. ``lstm_loss_fn``, ``arma_ic``) Optuna
automatically falls through to an independent sampler for the
categorical bits, so the user-facing contract stays "all strategies
tunable with any sampler".
"""

from __future__ import annotations

import warnings
from typing import assert_never

from optuna.exceptions import ExperimentalWarning
from optuna.samplers import (
    BaseSampler,
    CmaEsSampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)

from src.core.hpo_config import SamplerKind


def build_sampler(kind: SamplerKind, seed: int) -> BaseSampler:
    """
    Instantiate the requested Optuna sampler with a deterministic seed.
    """

    match kind:
        case SamplerKind.TPE:
            return TPESampler(seed=seed)
        case SamplerKind.RANDOM:
            return RandomSampler(seed=seed)
        case SamplerKind.CMAES:
            return CmaEsSampler(seed=seed)
        case SamplerKind.QMC:
            # QMCSampler emits an ExperimentalWarning on construction; QMC is
            # intentionally a first-class option in ``SamplerKind``, so the
            # upstream "interface may change" notice is noise both in tests and
            # in CLI output. Silenced narrowly around the constructor only.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ExperimentalWarning)
                return QMCSampler(seed=seed)
        case _ as unknown:
            assert_never(unknown)
