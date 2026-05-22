"""Trial-parameter sampling for :class:`StrategyTuner`.

The strategy's own ``suggest_params(trial)`` is the single source of truth
for the ctor-kwarg search space — every strategy and model exposes one,
and the strategy-level method flattens any leaf-owned knobs it passes
through (e.g. ``ReturnForecast.suggest_params`` returns ``arma_p_max``,
``lstm_hidden_dim`` alongside its own ``position_scale``).

What this module adds on top:

* **Pretrained-leaf filter**: when
  ``ExperimentConfig.pretrained_leaves`` pins one or more leaves, the
  keys those leaves freeze at the HPO boundary (per
  ``_LEAF_KEY_HPO_OVERRIDE_FROZEN_PARAMS`` in :mod:`src.core.config`)
  get dropped from the sampled dict. That set is a superset of the
  collision-check ``_LEAF_KEY_OWNED_PARAMS``: it additionally covers
  user-handshake kwargs like ``lstm_lookback`` that the strategy
  legitimately reads from ``strategy.params`` (so the collision check
  allows them there) but that HPO must not overwrite (the leaf's
  trained lookback is what feeds inference, and a sampled-then-passed
  mismatch fails the ``validate_pretrained_leaf`` window check
  mid-trial and burns the whole leg). Letting a trial override a
  frozen artifact's hyperparameters would be silent noise at best and
  a misleading tuned-params record at worst; the artifact wins, full
  stop.

A note on "wasted" Optuna suggestions
-------------------------------------
The filter runs AFTER ``strategy_cls.suggest_params(trial)``, so
``trial.suggest_*`` is still called for every search-space entry — even
keys we then discard. TPE ends up modelling irrelevant dimensions when
leaves are pinned. In practice this is a minor sampler-efficiency loss
(uncorrelated dimensions degrade to uniform sampling); the alternative
would be per-strategy ``suggest_params(skip=...)`` plumbing in five
strategies, which is heavier than the bias it removes. If this becomes
a real problem for large studies, the cheap fix is to pass ``skip``
through; the expensive-but-correct fix is per-leaf ``suggest_params``
composition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.config import _LEAF_KEY_HPO_OVERRIDE_FROZEN_PARAMS
from src.core.registry import strategy_registry

if TYPE_CHECKING:
    import optuna

    from src.core.config import ExperimentConfig


def sample_trial_params(cfg: ExperimentConfig, trial: optuna.trial.BaseTrial) -> dict[str, object]:
    """Draw one set of strategy ctor kwargs for ``trial``.

    Delegates to the registered ``strategy_cls.suggest_params(trial)``,
    then filters out any kwargs that a pinned pretrained leaf owns (see
    module docstring).

    Returns a fresh dict each call; the caller merges it into the base
    ``ExperimentConfig.strategy.params`` with sampled values winning —
    that merge is the tuner's job, not the sampler's.
    """
    strategy_cls = strategy_registry.get(cfg.strategy.name)
    suggested: dict[str, object] = strategy_cls.suggest_params(trial)

    if not cfg.pretrained_leaves:
        return suggested

    frozen_map = _LEAF_KEY_HPO_OVERRIDE_FROZEN_PARAMS.get(cfg.strategy.name, {})
    pinned_frozen: set[str] = set()
    for leaf_key in cfg.pretrained_leaves:
        pinned_frozen.update(frozen_map.get(leaf_key, ()))

    return {k: v for k, v in suggested.items() if k not in pinned_frozen}
