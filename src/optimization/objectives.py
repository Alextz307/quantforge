"""
The HPO objective: mean-of-folds Sharpe.

The tuner hands each trial's aggregate-metrics dict to
:class:`SharpeObjective` and returns the scalar to Optuna. Keeping the
adapter boundary at ``dict[str, object]`` (the exact shape
:meth:`src.analysis.metrics_aggregator.AggregateStats.to_dict` produces)
means the objective stays mockable without depending on the full
``AggregateStats`` type graph - the tuner converts via ``to_dict()``
immediately before dispatch.

The objective is MAXIMIZED - Optuna studies are configured with
``direction="maximize"``.
"""

from __future__ import annotations

from collections.abc import Mapping


def _read_float_metric(aggregate_metrics: Mapping[str, object], key: str) -> float:
    """
    Pull a numeric entry out of ``aggregate_metrics`` with clear errors.

    The tuner surfaces these errors to Optuna as failed trials; a missing
    key usually means the experiment ran on zero folds (empty dev slice)
    and the aggregator short-circuited - the error message points that
    out so the user doesn't go hunting for a bug in the objective itself.
    """

    if key not in aggregate_metrics:
        raise KeyError(
            f"objective needs aggregate_metrics['{key}'] but the experiment did "
            f"not produce it; available keys: {sorted(aggregate_metrics)}. "
            f"Most common cause: the dev slice had zero folds after holdout "
            f"reservation - shorten validation.holdout_pct or widen data.end."
        )
    value = aggregate_metrics[key]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(
            f"objective expected aggregate_metrics['{key}'] as numeric, got "
            f"{type(value).__name__}={value!r}; this usually means the "
            f"aggregator path changed shape - regenerate the experiment."
        )
    return float(value)


class SharpeObjective:
    """
    Mean-of-folds Sharpe - the sole HPO objective.
    """

    def __call__(self, aggregate_metrics: Mapping[str, object]) -> float:
        return _read_float_metric(aggregate_metrics, "sharpe_mean")
