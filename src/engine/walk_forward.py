"""Walk-forward orchestrator: strategy → engine → metrics, per fold.

The runtime leakage tripwire lives here, not inside the engine. Each
fold:

1. (optional) Feature-pipeline factory builds a FRESH pipeline, fits on
   ``fold.train`` only, and transforms both train/test into feature frames.
2. ``strategy.train(train_frame)``
3. Deep metadata check: iterate ``strategy.get_all_training_metadata()``
   and call ``validate_no_overlap(fold.test)`` on every non-None entry.
   Composite strategies expose both their own metadata and each wrapped
   model's — a drift inside the composite surfaces here, not silently.
4. ``signals = strategy.generate_signals(test_frame)``
5. ``raw = engine.run(fold.test, signals, slippage)``
6. ``metrics = MetricsCalculator.compute(raw.equity_curve, ...)``

Returned ``FoldResult`` bundles fold metadata + raw engine output +
computed performance metrics so callers don't need to recompute.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_engine import (
    BacktestResult,
    MetricsCalculator,
    PerformanceMetrics,
    SlippageConfig,
)
from src.core.constants import OHLCV_COLUMNS, PAIRS_LEG_SUFFIXES
from src.core.exceptions import LeakageError
from src.core.logging import get_logger
from src.core.persistence import FOLD_DIR_PREFIX
from src.core.temporal import TemporalSplit, WalkForwardValidator
from src.core.types import Interval
from src.engine.interface import IBacktestEngine
from src.features.interface import IFeaturePipeline
from src.strategies.interface import IStrategy

logger = get_logger(__name__)


@dataclass(frozen=True)
class FoldResult:
    """Per-fold orchestration output: metadata + raw + metrics."""

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    backtest: BacktestResult
    metrics: PerformanceMetrics


_LEG_A_RENAME: dict[str, str] = {f"{c}{PAIRS_LEG_SUFFIXES[0]}": c for c in OHLCV_COLUMNS}
_LEG_B_RENAME: dict[str, str] = {f"{c}{PAIRS_LEG_SUFFIXES[1]}": c for c in OHLCV_COLUMNS}


def _split_pairs_frame(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a wide-format pairs frame into two single-leg OHLCV frames."""
    missing_a = [c for c in _LEG_A_RENAME if c not in bars.columns]
    missing_b = [c for c in _LEG_B_RENAME if c not in bars.columns]
    if missing_a or missing_b:
        raise ValueError(
            f"pairs walk-forward dispatch expected wide-format columns "
            f"{sorted(_LEG_A_RENAME) + sorted(_LEG_B_RENAME)}, missing "
            f"{sorted(missing_a + missing_b)}; fix by ensuring the multi-ticker "
            f"fetch path produced both legs before invoking walk-forward."
        )
    bars_a = bars[list(_LEG_A_RENAME)].rename(columns=_LEG_A_RENAME)
    bars_b = bars[list(_LEG_B_RENAME)].rename(columns=_LEG_B_RENAME)
    return bars_a, bars_b


def _validate_deep_metadata(
    strategy: IStrategy,
    *,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Run the leakage invariant across every tracked metadata exposed by
    the strategy (composite leaves included).

    Two invariants, one per-entry:

    * ``train_end < test_data.index[0]`` — always enforced. Catches the
      canonical lookahead-leakage path: a leaf (or the strategy itself)
      that trained through the fold's test window would have seen the
      future it's now being evaluated on.
    * ``train_end < train_data.index[0]`` — enforced ONLY when
      ``tracked.is_pretrained``. A pretrained leaf frozen-injected by the
      user should NOT have seen the strategy's fold train window: if it
      did, strategy-level state fits on bars where the leaf is in-sample
      and produces inflated backtest numbers at eval. Fresh (non-
      pretrained) leaves legitimately train on the fold train window
      every fold — skipping this check for them preserves the normal
      walk-forward semantics.

    A ``LeakageError`` is re-raised with the strategy class name + origin
    prefixed so the failing component is obvious. A ``None`` metadata
    entry means the component never completed ``fit()`` — logged at WARN
    level and skipped, so the remaining tracked entries still provide
    partial coverage rather than swallowing the whole check.
    """
    strategy_cls = type(strategy).__name__
    # Hoist fold boundaries once; every tracked entry compares scalars
    # rather than re-scanning the fold DataFrame inside ``validate_no_overlap``.
    test_start: pd.Timestamp = test_data.index[0]
    train_start: pd.Timestamp = train_data.index[0]
    saw_any = False
    for tracked in strategy.get_all_training_metadata():
        meta = tracked.metadata
        if meta is None:
            logger.warning(
                "%s.%s has no training metadata — skipping leakage check for this component",
                strategy_cls,
                tracked.origin,
            )
            continue
        saw_any = True
        if test_start <= meta.train_end:
            raise LeakageError(
                f"{strategy_cls}.{tracked.origin}: Evaluation data starts at "
                f"{test_start} but model was trained through {meta.train_end}; "
                f"this would constitute data leakage. Fix by widening the "
                f"embargo gap or by ensuring the leaf was trained on a window "
                f"strictly preceding the test fold."
            )
        if tracked.is_pretrained and train_start <= meta.train_end:
            raise LeakageError(
                f"{strategy_cls}.{tracked.origin}: pretrained leaf overlaps "
                f"fold train window (leaf.train_end={meta.train_end} >= "
                f"fold.train_start={train_start}); fix by using a leaf whose "
                f"train_end precedes this fold's train_start."
            )
    if not saw_any:
        raise RuntimeError(
            f"{strategy_cls}.get_all_training_metadata() returned no populated "
            "metadata — at least one component must have completed fit(); "
            "fix by calling strategy.train() before walk-forward evaluation."
        )


def evaluate_walk_forward(
    strategy: IStrategy,
    bars: pd.DataFrame,
    validator: WalkForwardValidator,
    engine: IBacktestEngine,
    slippage: SlippageConfig,
    interval: Interval,
    risk_free_rate: float = 0.0,
    *,
    feature_pipeline_factory: Callable[[], IFeaturePipeline] | None = None,
    progress: bool = False,
    checkpoint_root: Path | None = None,
) -> list[FoldResult]:
    """Run train → leakage check → signals → engine → metrics, per fold.

    Args:
        strategy: A trained-from-scratch ``IStrategy`` instance. The
            orchestrator calls ``train()`` for each fold, so the same
            strategy instance is reused across folds (fresh fit each time).
        bars: Full OHLCV DataFrame to be split by ``validator``.
        validator: Walk-forward splitter producing ``TemporalSplit``s.
        engine: Backtest engine adapter (typically ``CppBacktestEngine``).
        slippage: Slippage scenario applied uniformly across folds.
        interval: Bar interval, used to pick the annualization factor
            for ``MetricsCalculator``.
        risk_free_rate: Per-period risk-free rate for Sharpe / Sortino
            (default 0.0).
        feature_pipeline_factory: Optional zero-arg callable producing a
            fresh ``IFeaturePipeline`` on each call. When provided, a new
            pipeline is built PER FOLD, fit on ``fold.train`` only, and
            applied to both train/test frames before the strategy sees
            them. Passing a single pre-fit instance would either leak the
            scaler's statistics across folds or trigger the fit-once
            scaler guard on the second fold; the factory shape makes the
            per-fold refit explicit.
        checkpoint_root: When set, the strategy's ``train()`` is called
            with ``checkpoint_path=<checkpoint_root>/fold_<i>`` so any
            wrapped LSTM / XGBoost leaf can dump best-so-far weights
            mid-fit. ``None`` disables mid-fit checkpointing entirely
            (the strategy's ``train()`` call is unaffected).

    Returns:
        One ``FoldResult`` per validator fold, in fold order.

    Raises:
        LeakageError: From ``validate_no_overlap`` if a fold's test data
            overlaps the training period of the strategy or any wrapped
            model. The raised exception's message names the failing
            component's origin label.
        RuntimeError: If a strategy fails to populate any training
            metadata after ``train()`` (contract violation).
    """
    annualization = interval.annualization_factor()
    n_folds = validator.n_splits
    results: list[FoldResult] = []
    fold_iter: Iterable[TemporalSplit] = validator.split(bars)
    if progress:
        # ``disable=None`` lets tqdm auto-detect non-TTY environments and
        # silently no-op (e.g. CI logs, redirected output).
        from tqdm.auto import tqdm

        fold_iter = tqdm(fold_iter, total=n_folds, desc="folds", disable=None)
    for fold in fold_iter:
        fold_logger = get_logger(__name__, fold=f"{fold.fold_index + 1}/{n_folds}")
        fold_logger.info(
            "train=[%s..%s] test=[%s..%s]",
            fold.train.index[0],
            fold.train.index[-1],
            fold.test.index[0],
            fold.test.index[-1],
        )
        if feature_pipeline_factory is not None:
            pipeline = feature_pipeline_factory()
            # fit_transform(train) does the fit AND the train-window transform
            # in one pass instead of fit() + transform(train) == two passes.
            train_frame = pipeline.fit_transform(fold.train)
            test_frame = pipeline.transform(fold.test)
        else:
            train_frame = fold.train
            test_frame = fold.test

        if checkpoint_root is not None:
            fold_ckpt: Path | None = checkpoint_root / f"{FOLD_DIR_PREFIX}{fold.fold_index + 1}"
        else:
            fold_ckpt = None
        strategy.train(train_frame, checkpoint_path=fold_ckpt)
        _validate_deep_metadata(strategy, train_data=train_frame, test_data=test_frame)

        signals = strategy.generate_signals(test_frame)
        if strategy.is_pairs_strategy:
            bars_a, bars_b = _split_pairs_frame(fold.test)
            raw = engine.run_pairs(bars_a, bars_b, signals, strategy.hedge_ratio, slippage)
        else:
            raw = engine.run(fold.test, signals, slippage)
        metrics = MetricsCalculator.compute(
            raw.equity_curve,
            annualization,
            risk_free_rate,
        )
        fold_logger.info(
            "done sharpe=%.4f ann_return=%.4f max_dd=%.4f",
            metrics.sharpe_ratio,
            metrics.annualized_return,
            metrics.max_drawdown,
        )
        results.append(
            FoldResult(
                fold_index=fold.fold_index,
                train_start=fold.train.index[0],
                train_end=fold.train.index[-1],
                test_start=fold.test.index[0],
                test_end=fold.test.index[-1],
                backtest=raw,
                metrics=metrics,
            )
        )
    return results
