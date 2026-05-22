"""Test-only strategy stubs registered globally on ``strategy_registry``.

The multi-feature dispatch tests need a registered strategy to drive
config-driven instantiation paths (validator, fetch_bars, walk_forward).
No production strategy carries ``is_multi_feature_strategy=True`` yet, so
these stubs fill the gap until the first real multi-feature strategy
lands.

Importing this module registers the stubs as side effect.
``tests/conftest.py`` does this once at pytest collection so any test —
unit, integration, smoke — can reference the stubs by name through the
registry without an explicit import.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Self

import pandas as pd

from src.core.registry import strategy_registry
from src.core.temporal import TrainingMetadata
from src.core.types import Interval
from src.strategies.interface import IStrategy

_DEFAULT_PRIMARY = "AAA"


@strategy_registry.register("_MultiFeatureTestStub")
class MultiFeatureTestStub(IStrategy):
    """Minimal multi-feature strategy: emits flat-zero signals.

    The dispatch path is what's under test, not signal logic. Reading a
    feature-ticker column inside ``generate_signals`` proves the strategy
    saw the wide frame; the returned series is constant so the engine
    just compounds zero PnL.
    """

    is_multi_feature_strategy: ClassVar[bool] = True

    def __init__(
        self,
        primary_ticker: str = _DEFAULT_PRIMARY,
        feature_tickers: Sequence[str] = (),
        interval: Interval = Interval.DAILY,
    ) -> None:
        self._primary = primary_ticker
        self._features = tuple(feature_tickers)
        self._interval = interval

    def train(
        self,
        train_data: pd.DataFrame,
        *,
        checkpoint_path: Path | None = None,  # noqa: ARG002
        **kwargs: object,
    ) -> None:
        for ticker in (self._primary, *self._features):
            _ = train_data[f"close_{ticker}"]
        cols = tuple(f"close_{t}" for t in (self._primary, *self._features))
        self._set_fitted_with_metadata(TrainingMetadata.from_fit(train_data, self._interval, cols))

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self._assert_fitted_with_metadata()
        _ = data[f"close_{self._primary}"]
        return pd.Series(0.0, index=data.index, name="signal")

    @property
    def name(self) -> str:
        return "_MultiFeatureTestStub"

    @property
    def primary_ticker(self) -> str:
        return self._primary

    @property
    def required_warmup_bars(self) -> int:
        return 0

    @staticmethod
    def suggest_params(trial: object) -> dict[str, object]:
        return {}

    @classmethod
    def load(cls, path: str | Path) -> Self:  # pragma: no cover — not exercised
        raise NotImplementedError


@strategy_registry.register("_BothFlagsStub")
class BothFlagsStub(IStrategy):
    """Class-level config error: BOTH capability flags True simultaneously."""

    is_pairs_strategy: ClassVar[bool] = True
    is_multi_feature_strategy: ClassVar[bool] = True

    def train(self, train_data: pd.DataFrame, **kwargs: object) -> None:  # pragma: no cover
        raise NotImplementedError

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # pragma: no cover
        raise NotImplementedError

    @property
    def name(self) -> str:  # pragma: no cover
        return "_BothFlagsStub"

    @property
    def required_warmup_bars(self) -> int:  # pragma: no cover
        return 0

    @staticmethod
    def suggest_params(trial: object) -> dict[str, object]:  # pragma: no cover
        return {}
