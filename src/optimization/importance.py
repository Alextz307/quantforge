"""
Deterministic fANOVA hyperparameter importance for a completed study.

The single seam both the HPO report renderer and the webapp live-monitor
read through, so the figure and the panel always agree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.constants import HPO_IMPORTANCE_FANOVA_SEED

if TYPE_CHECKING:
    import optuna

__all__ = ["param_importances"]


def param_importances(study: optuna.Study) -> dict[str, float]:
    """
    Seeded fANOVA hyperparameter importances for ``study``.

    Optuna restricts the fANOVA fit to COMPLETE trials, so FAIL / PRUNED
    trials never enter the computation. The evaluator fits a random forest
    whose split sampling is RNG-driven; pinning the seed makes repeated calls
    on the same study return the identical mapping (a reproducible figure and
    a stable panel). Scores are non-negative and normalised to sum to 1.

    Optuna's importance submodule pulls in scikit-learn, so it is imported
    here at call time rather than at module load - importing this module
    therefore costs neither Optuna nor scikit-learn until an importance is
    actually requested.
    """

    from optuna.importance import FanovaImportanceEvaluator, get_param_importances

    evaluator = FanovaImportanceEvaluator(seed=HPO_IMPORTANCE_FANOVA_SEED)
    return get_param_importances(study, evaluator=evaluator)
