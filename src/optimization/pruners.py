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

from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PercentilePruner,
)

from src.core.hpo_config import PrunerKind

# 25% keeps the top three quarters of trials; tighter than median (50%)
# but looser than the most aggressive Optuna defaults. Sane thesis-demo
# default — users who care can construct a custom PercentilePruner.
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
