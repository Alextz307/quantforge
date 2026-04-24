"""Objective adapters that map ``ExperimentResult.aggregate_metrics`` to a scalar.

The tuner hands each trial's aggregate-metrics dict to the selected
objective and returns the scalar to Optuna. Keeping the adapter boundary
at ``dict[str, object]`` (the exact shape
:mod:`src.orchestration.experiment._aggregate_metrics` produces) means
objectives stay mockable without depending on the full
``ExperimentResult`` type graph — the tuner passes
``result.aggregate_metrics`` directly.

Every objective is MAXIMIZED — Optuna studies are configured with
``direction="maximize"``. Drawdown / loss-style metrics are negated at
the objective layer so the study semantics stay uniform.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, assert_never, runtime_checkable

from src.core.hpo_config import ObjectiveKind

# Penalty on |max_drawdown_worst|. Chosen so a 10-pt drawdown costs 0.2
# units of Sortino — aggressive enough to flag deep-drawdown trials but
# not so punitive that a stable-Sortino strategy with a short dip gets
# discarded. Configurable via the dataclass kwarg when users want to
# override via an explicitly-constructed objective.
_DEFAULT_DRAWDOWN_PENALTY = 2.0


@runtime_checkable
class IObjective(Protocol):
    """Callable that collapses aggregate metrics to a single maximization target."""

    def __call__(self, aggregate_metrics: Mapping[str, object]) -> float: ...


def _read_float_metric(aggregate_metrics: Mapping[str, object], key: str) -> float:
    """Pull a numeric entry out of ``aggregate_metrics`` with clear errors.

    The tuner surfaces these errors to Optuna as failed trials; a missing
    key usually means the experiment ran on zero folds (empty dev slice)
    and the aggregator short-circuited — the error message points that
    out so the user doesn't go hunting for a bug in the objective itself.
    """
    if key not in aggregate_metrics:
        raise KeyError(
            f"objective needs aggregate_metrics['{key}'] but the experiment did "
            f"not produce it; available keys: {sorted(aggregate_metrics)}. "
            f"Most common cause: the dev slice had zero folds after holdout "
            f"reservation — shorten validation.holdout_pct or widen data.end."
        )
    value = aggregate_metrics[key]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(
            f"objective expected aggregate_metrics['{key}'] as numeric, got "
            f"{type(value).__name__}={value!r}; this usually means the "
            f"aggregator path changed shape — regenerate the experiment."
        )
    return float(value)


class SharpeObjective:
    """Mean Sharpe across folds — the thesis-default single-objective."""

    def __call__(self, aggregate_metrics: Mapping[str, object]) -> float:
        return _read_float_metric(aggregate_metrics, "sharpe_mean")


class CalmarObjective:
    """Mean Calmar across folds — rewards return-per-unit-drawdown."""

    def __call__(self, aggregate_metrics: Mapping[str, object]) -> float:
        return _read_float_metric(aggregate_metrics, "calmar_mean")


@dataclass(frozen=True)
class SortinoMinusDrawdownPenaltyObjective:
    """Sortino mean minus a penalty on the worst observed drawdown.

    ``max_drawdown_worst`` is reported as a negative number (the min
    across folds). We apply ``abs()`` before the penalty so the
    arithmetic reads as "Sortino minus penalty-times-depth-of-drawdown"
    and setting ``penalty=0.0`` degenerates cleanly to pure Sortino.

    Configurable via the ``penalty`` kwarg; ``build_objective`` uses the
    default. If a study needs a non-default penalty, construct the
    objective directly and pass it into :class:`StrategyTuner`.
    """

    penalty: float = _DEFAULT_DRAWDOWN_PENALTY

    def __call__(self, aggregate_metrics: Mapping[str, object]) -> float:
        sortino = _read_float_metric(aggregate_metrics, "sortino_mean")
        max_dd = _read_float_metric(aggregate_metrics, "max_drawdown_worst")
        return sortino - self.penalty * abs(max_dd)


def build_objective(kind: ObjectiveKind) -> IObjective:
    """Dispatch :class:`ObjectiveKind` to a concrete objective instance."""
    match kind:
        case ObjectiveKind.SHARPE:
            return SharpeObjective()
        case ObjectiveKind.CALMAR:
            return CalmarObjective()
        case ObjectiveKind.SORTINO_MINUS_DRAWDOWN:
            return SortinoMinusDrawdownPenaltyObjective()
        case _ as unknown:
            assert_never(unknown)
