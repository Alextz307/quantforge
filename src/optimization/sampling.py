"""Trial-parameter sampling for :class:`StrategyTuner`.

The strategy's own ``suggest_params(trial)`` is the single source of truth
for the ctor-kwarg search space — every strategy and model exposes one,
and the strategy-level method flattens any leaf-owned knobs it passes
through (e.g. ``ReturnForecast.suggest_params`` returns ``arma_p_max``,
``lstm_hidden_dim`` alongside its own ``position_scale``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import strategy_registry

if TYPE_CHECKING:
    import optuna

    from src.core.config import ExperimentConfig


def sample_trial_params(cfg: ExperimentConfig, trial: optuna.trial.BaseTrial) -> dict[str, object]:
    """Draw one set of strategy ctor kwargs for ``trial``.

    Returns a fresh dict each call; the caller merges it into the base
    ``ExperimentConfig.strategy.params`` with sampled values winning —
    that merge is the tuner's job, not the sampler's.
    """
    strategy_cls = strategy_registry.get(cfg.strategy.name)
    suggested: dict[str, object] = strategy_cls.suggest_params(trial)
    return suggested
