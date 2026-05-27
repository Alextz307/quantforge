"""Factory mapping :class:`PrunerKind` to Optuna ``BasePruner`` instances.

Pruners need intermediate values reported via ``trial.report(value, step)``
to actually prune anything. The LSTM and XGBoost leaves re-raise
``optuna.TrialPruned`` from their training loops when a pruner decides
mid-training that a trial is unpromising — the tuner lets that exception
propagate so Optuna marks the trial pruned rather than failed.

``PrunerKind.NONE`` maps to ``NopPruner`` so the pruner field stays
enum-typed end-to-end (no Optional plumbing in downstream signatures).
"""

from __future__ import annotations

from typing import assert_never

from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PercentilePruner,
)

from src.core.hpo_config import PrunerKind

_DEFAULT_PERCENTILE_PRUNER_VALUE = 25.0


def build_pruner(kind: PrunerKind) -> BasePruner:
    """Instantiate the requested Optuna pruner."""
    match kind:
        case PrunerKind.MEDIAN:
            return MedianPruner()
        case PrunerKind.HYPERBAND:
            return HyperbandPruner()
        case PrunerKind.PERCENTILE:
            return PercentilePruner(percentile=_DEFAULT_PERCENTILE_PRUNER_VALUE)
        case PrunerKind.NONE:
            return NopPruner()
        case _ as unknown:
            assert_never(unknown)
