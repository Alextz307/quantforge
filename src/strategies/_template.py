"""
Skeleton single-asset strategy. Copy this file (drop the underscore) and fill in the stubs.

The leading underscore in the filename is intentional:
``src.core.registry.autoload_package`` skips ``_``-prefixed modules so this
template never registers itself on ``strategy_registry``. Renaming the copy
to e.g. ``mean_reversion.py`` makes the autoloader import the module - to
actually register the class, also uncomment the ``@strategy_registry.register(...)``
decorator below.

What this skeleton demonstrates:

* The atomic fitted-state commit (``_set_fitted_with_metadata`` at the end of
  ``train``; ``_assert_fitted_with_metadata`` at the top of ``generate_signals``).
* The 5-method ``IStrategy`` surface (``train``, ``generate_signals``, ``name``,
  ``required_warmup_bars``, ``suggest_params``).
* Where to register on the ``strategy_registry`` (commented out below).

What this skeleton does NOT demonstrate (look at the listed exemplars):

* Pairs strategies (``is_pairs_strategy = True``) -> ``pairs_trading.py``.
* Multi-feature single-asset (``is_multi_feature_strategy = True`` +
  ``primary_ticker``) -> ``cross_asset_momentum.py``.
* Composite strategies that own ML leaves -> ``momentum_gatekeeper.py``
  (pipeline + classifier), ``return_forecast.py`` /
  ``volatility_targeting.py`` (passthrough bundle).

See ``src/strategies/README.md`` for the full extension checklist.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.core.types import Interval
from src.strategies.interface import IStrategy

if TYPE_CHECKING:
    import optuna


# Uncomment + import + decorate the class below once you rename this file.
# The string here is the public name used in YAML configs and HPO output.
#
#     from src.core.registry import strategy_registry
#
#     @strategy_registry.register("MyStrategyName")
class _TemplateStrategy(IStrategy):
    """
    Replace this docstring with one paragraph: WHAT signal this strategy emits and WHY.

    State the asset shape (single-asset / pairs / multi-feature), the
    methodology lineage if any (paper or textbook reference), and the position
    semantics (binary 0/1, signed -1/0/+1, continuous).
    """

    def __init__(
        self,
        # Replace these with your strategy's actual hyperparameters. Use
        # plain Python types (int / float / str / Enum) - Pydantic + YAML
        # will coerce strings into Enum members automatically. The kwarg
        # names listed here MUST match the keys returned by suggest_params.
        window: int = 20,
        threshold: float = 1.0,
        interval: Interval = Interval.DAILY,
    ) -> None:
        # Validate every numeric/string param at the boundary - catches
        # mis-typed YAML and bad HPO trials before they corrupt fold state.
        if window < 2:
            raise ValueError(
                f"window must be >= 2, got {window}; fix by passing a window of at least 2 bars."
            )
        if threshold <= 0:
            raise ValueError(
                f"threshold must be > 0, got {threshold}; fix by passing a "
                f"strictly positive threshold."
            )

        self._window = window  # read by required_warmup_bars below
        self._threshold = threshold  # read by your generate_signals fill-in
        self._interval = interval  # read by the TrainingMetadata commit at end of train

    def train(
        self,
        train_data: pd.DataFrame,
        *,
        checkpoint_path: Path | None = None,  # noqa: ARG002 - accept for NN-style leaves; ignore here
        **kwargs: object,
    ) -> None:
        """
        Fit any per-strategy state on ``train_data``. Called once before backtesting.
        """

        # Fill in this method by following the two numbered steps below, then
        # delete the trailing ``raise NotImplementedError`` that marks the stub.
        #
        # 1. Run your fitting work here (estimate parameters, fit a scaler,
        #    etc.). Use ONLY columns from train_data. Do NOT touch any data
        #    after train_data.index[-1] - the backtest engine will re-call
        #    generate_signals() on the test window separately.
        #
        # 2. Atomic fitted-state commit. MUST be the last line of train().
        #    _set_fitted_with_metadata refuses None and is the only legal
        #    mutator of self._training_metadata - never assign that slot
        #    directly. The feature_columns tuple should list every column
        #    your strategy READS from generate_signals() data.
        #
        # from src.core.temporal import TrainingMetadata
        #
        # self._set_fitted_with_metadata(
        #     TrainingMetadata.from_fit(train_data, self._interval, ("close",))
        # )
        raise NotImplementedError(
            "_TemplateStrategy.train() is a stub; delete this raise once your "
            "fitting logic + final _set_fitted_with_metadata(...) are in place."
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Return position signals at time t (the engine shifts to t+1 - do NOT shift here).
        """

        # Read-side guard: raises a descriptive RuntimeError if train() was
        # not called. Returns the metadata narrowed to non-None for use
        # below if you need it (e.g. comparing intervals).
        self._assert_fitted_with_metadata()

        # Compute your signal series here. Position semantics are your
        # choice (binary 0/1, signed -1/0/+1, continuous size); document
        # them in the class docstring above. Leading bars during warmup
        # MUST stay NaN - never fillna(0) and never bfill().
        raise NotImplementedError(
            "_TemplateStrategy.generate_signals() is a stub; replace this "
            "NotImplementedError with your actual signal computation. The "
            "returned Series must share data.index, and the leading "
            "self.required_warmup_bars rows MUST be NaN."
        )

    @property
    def name(self) -> str:
        # Public string used in registry lookups and report tables. Keep
        # consistent with the @strategy_registry.register("...") decorator.
        return "Template"

    @property
    def required_warmup_bars(self) -> int:
        # Maximum number of leading bars before generate_signals can produce
        # a valid (non-NaN) value. The walk-forward dispatcher uses this to
        # extend test windows so signal coverage starts at the fold boundary.
        return self._window

    @staticmethod
    def suggest_params(trial: optuna.trial.BaseTrial) -> dict[str, object]:
        """
        Declare this strategy's Optuna search space.

        The returned dict's KEYS must exactly match this strategy's __init__
        kwarg names - ``StrategyTuner`` merges this dict into the ctor.

        The Optuna trial parameter NAMES (the strings passed to
        trial.suggest_*) are global identifiers across an Optuna study;
        prefix-namespace them by strategy (e.g. ``"template_window"``) so a
        shared study running multiple strategies doesn't collide.
        """

        return {
            "window": trial.suggest_int("template_window", 10, 50),
            "threshold": trial.suggest_float("template_threshold", 0.5, 2.0),
        }
