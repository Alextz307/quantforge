"""
The wired, ready-to-run experiment primitive.

``Experiment`` is a frozen bundle of every component that participates in a
single walk-forward run: data source, strategy, validator, engine, slippage,
and an optional feature-pipeline FACTORY. It is produced by
:func:`src.orchestration.builder.build_experiment` from a validated
:class:`ExperimentConfig`.

Why the feature pipeline is a factory, not an instance
------------------------------------------------------
Feature pipelines (e.g. :class:`FeatureEngineeringPipeline`) enforce a
``fit_once`` guard on their scaler - a second ``fit()`` raises
``LeakageError``. A walk-forward run needs to fit the scaler PER FOLD on
``fold.train`` only; fitting once on the full dev region would leak later
folds' test-window statistics into earlier folds' features. A single
instance cannot satisfy both constraints. A factory closure captures the
config-derived kwargs and produces a fresh instance whenever the caller
asks - one per fold.

The strategy stays as an instance because each ``IStrategy.train()``
implementation is contracted to reset its own fit state from scratch.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_engine import SlippageConfig
from src.analysis.feature_importance import build_importance_artifact
from src.analysis.metrics_aggregator import aggregate_folds
from src.core import json_io
from src.core.config import ExperimentConfig, write_frozen_yaml
from src.core.constants import PAIRS_LEG_SUFFIXES
from src.core.logging import attach_run_log_file, get_logger
from src.core.persistence import (
    EXPERIMENT_CHECKPOINTS_SUBDIR,
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_METRICS_JSON,
    EXPERIMENT_STRATEGY_SUBDIR,
    FEATURE_IMPORTANCE_JSON,
    FOLD_RESULTS_JSONL,
    RUNS_SUBDIR,
    ensure_model_dir,
    write_experiment_manifest,
)
from src.core.seeding import seed_all
from src.core.temporal import WalkForwardValidator, resolve_holdout_boundary
from src.data.fingerprint import (
    fingerprint_bars,
    fingerprint_multi_bars,
    fingerprint_pair_bars,
)
from src.data.interface import IDataSource
from src.engine.interface import IBacktestEngine
from src.engine.walk_forward import FoldResult, evaluate_walk_forward
from src.features.interface import IFeaturePipeline
from src.orchestration.git_info import read_git_sha
from src.orchestration.manifest import Manifest
from src.orchestration.types import ExperimentResult, FoldRecord
from src.strategies.interface import IStrategy

# ``StrategyReporter`` is lazy-imported inside ``run()`` when ``write_report``
# is True - matplotlib's cold-import tree (~4s incl. pyplot + PIL + numpy
# cascades) is substantial and `--no-report` runs (e.g. HPO trials where the
# tuner drives reporting at the study level) shouldn't pay it. The lazy
# import mirrors ``seed_all``'s lazy torch import for the same reason.

_module_logger = get_logger(__name__)

_DEFAULT_STORE_ROOT = Path("experiment_results")
_EXPERIMENT_ID_SUFFIX_BYTES = 4  # -> 8 hex chars, 2^32 combos; low collision risk


def _make_experiment_id(strategy_name: str, created_at: datetime, git_sha: str) -> str:
    """
    Compose a unique experiment id: ``{utc_ts}_{strategy}_{sha}_{rand}``.

    Random suffix (hex-encoded cryptographic bytes) disambiguates two
    invocations in the same second + same strategy + same sha - matters for
    HPO parallelism and ``experiment compare`` subprocess fan-out.
    """

    ts = created_at.strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(_EXPERIMENT_ID_SUFFIX_BYTES)
    return f"{ts}_{strategy_name}_{git_sha}_{suffix}"


def fetch_bars(
    data_source: IDataSource,
    cfg: ExperimentConfig,
    strategy: IStrategy,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars dispatched by ``strategy``'s capability flags.

    Three shapes:

    * **single-asset** (1 ticker, no capability flag): direct fetch,
      canonical OHLCV columns.
    * **pairs** (2 tickers, ``is_pairs_strategy=True``): inner-join +
      ``_a`` / ``_b`` suffix; consumed by ``PairsTradingStrategy``.
    * **multi-feature** (N>=1 tickers, ``is_multi_feature_strategy=True``):
      inner-join + ``_<TICKER>`` suffix; consumed by single-asset traded
      strategies that use the other tickers as feature inputs only.

    Strategy-driven dispatch keeps the pairs vs. multi-feature distinction
    at N=2 unambiguous - the strategy class is the source of truth, not a
    bool flag the caller has to remember to pass.
    """

    tickers = cfg.data.tickers
    if len(tickers) == 0:
        raise ValueError(
            "ExperimentConfig.data.tickers must be non-empty; fix by listing at least one ticker."
        )
    if strategy.is_multi_feature_strategy:
        return _fetch_multi_bars(data_source, cfg, tickers)
    if len(tickers) == 1:
        return data_source.fetch(tickers[0], cfg.data.start, cfg.data.end, cfg.data.interval)
    if len(tickers) == 2:
        return _fetch_pair_bars(data_source, cfg, tickers[0], tickers[1])
    raise ValueError(
        f"ExperimentConfig.data.tickers supports 1 ticker (single-asset) or 2 "
        f"tickers (pairs) for non-multi-feature strategies; got {len(tickers)}: "
        f"{tickers}. Fix by trimming data.tickers, or - if the strategy reads "
        f"a wide multi-ticker frame - by setting "
        f"is_multi_feature_strategy=True on the strategy class so the caller "
        f"routes through the multi-feature fetch path."
    )


def _fetch_pair_bars(
    data_source: IDataSource,
    cfg: ExperimentConfig,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    """
    Fetch both legs of a pair, inner-join, suffix the OHLCV columns.
    """

    bars_a = data_source.fetch(ticker_a, cfg.data.start, cfg.data.end, cfg.data.interval)
    bars_b = data_source.fetch(ticker_b, cfg.data.start, cfg.data.end, cfg.data.interval)
    suffix_a, suffix_b = PAIRS_LEG_SUFFIXES
    # Inner join drops bars present on only one leg (typical for cross-region
    # holiday mismatches). Outer join would leak NaN OHLC into the engine's
    # ``Bar.is_valid`` check; left/right joins would silently bias the
    # calendar to one leg.
    joined = bars_a.add_suffix(suffix_a).join(bars_b.add_suffix(suffix_b), how="inner")
    if joined.empty:
        raise ValueError(
            f"pair fetch for ({ticker_a}, {ticker_b}) produced zero overlapping "
            f"bars after inner join over [{cfg.data.start}, {cfg.data.end}]; "
            f"fix by widening the date range or by choosing a pair traded on "
            f"the same calendar."
        )
    return joined


def _fetch_multi_bars(
    data_source: IDataSource,
    cfg: ExperimentConfig,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """
    Fetch N legs, inner-join on shared timestamps, suffix columns ``_<TICKER>``.

    Same inner-join rationale as ``_fetch_pair_bars`` (NaN poisoning on the
    engine's bar-validity check rules out outer / left / right joins). The
    suffix is the literal ticker name - strategies read e.g. ``close_SPY``
    directly, which is more readable than ``_a / _b`` once N exceeds two.
    """

    suffixed = [
        data_source.fetch(ticker, cfg.data.start, cfg.data.end, cfg.data.interval).add_suffix(
            f"_{ticker}"
        )
        for ticker in tickers
    ]
    joined = suffixed[0]
    for other in suffixed[1:]:
        joined = joined.join(other, how="inner")
    if joined.empty:
        raise ValueError(
            f"multi-feature fetch for {list(tickers)} produced zero overlapping "
            f"bars after inner join over [{cfg.data.start}, {cfg.data.end}]; "
            f"fix by widening the date range or by choosing tickers that trade "
            f"on the same calendar."
        )
    return joined


def compute_data_hash(strategy: IStrategy, bars: pd.DataFrame, tickers: Sequence[str]) -> str:
    """
    Dispatch to the correct fingerprint helper based on strategy shape.

    Single source of truth for "which fingerprint applies" - call sites in
    ``Experiment.run`` and ``holdout_eval`` route through this so a future
    fourth shape only adds one branch here, not three.
    """

    if strategy.is_multi_feature_strategy:
        return fingerprint_multi_bars(bars, tickers)
    if strategy.is_pairs_strategy:
        return fingerprint_pair_bars(bars)
    return fingerprint_bars(bars)


def _slice_dev(bars: pd.DataFrame, boundary: pd.Timestamp | None) -> pd.DataFrame:
    """
    Return the dev region - everything strictly before ``boundary``.

    ``boundary`` is the first bar OF the holdout (see
    ``resolve_holdout_boundary``). ``None`` disables the reservation.
    """

    if boundary is None:
        return bars
    return bars.loc[bars.index < boundary]


@dataclass(frozen=True)
class RunOptions:
    """
    Per-invocation knobs for :meth:`Experiment.run`.

    Bundled so the run() signature doesn't grow a flag per concern. Defaults
    match the no-arguments behaviour: write to ``experiment_results/``, emit
    the strategy report, no progress bar, no per-fold checkpoints.

    ``publish_label`` is forwarded to the strategy reporter so thesis-prose
    citations stay stable across reruns; ``None`` keeps the legacy
    ``experiment_id``-based caption + label.
    """

    store_root: Path | None = None
    write_report: bool = True
    progress: bool = False
    checkpoint: bool = False
    publish_label: str | None = None
    compute_feature_importance: bool = False


@dataclass(frozen=True)
class Experiment:
    """
    A fully-wired walk-forward experiment.

    Prefer constructing via :func:`build_experiment` - direct instantiation
    is intentional for tests that want to inject mocks per component.
    """

    config: ExperimentConfig
    data_source: IDataSource
    strategy: IStrategy
    validator: WalkForwardValidator
    engine: IBacktestEngine
    slippage: SlippageConfig
    feature_pipeline_factory: Callable[[], IFeaturePipeline] | None = None

    def run(self, options: RunOptions | None = None) -> ExperimentResult:
        """
        Execute the full walk-forward loop and persist every artifact.

        Pipeline:
        1. Seed numpy/torch/random deterministically from ``config.seed``.
        2. Fetch bars via the wired ``IDataSource`` (with cache).
        3. Resolve the holdout boundary per the config's ``validation`` block.
        4. Slice dev (bars strictly before the boundary) - the walk-forward
           splitter sees dev only. The holdout region is reserved for the
           post-thesis OOS evaluation and is NEVER touched here.
        5. Compute ``data_hash = fingerprint_bars(bars_full)`` so future
           holdout-eval commands can refuse on vendor drift.
        6. Create ``store_root/runs/<experiment_id>/`` and write the frozen
           ``config.yaml`` + ``manifest.json`` BEFORE any compute - so a
           mid-run crash still leaves a record of what was attempted.
        7. Run walk-forward (deep metadata check wired in per-fold via
           ``evaluate_walk_forward``) and convert each ``FoldResult`` into a
           serialisable ``FoldRecord``.
        8. Write ``fold_results.jsonl`` + ``metrics.json`` +
           ``strategy_state/`` (the last-fold strategy state; see caveat).
        9. Return the ``ExperimentResult`` so in-memory callers (tests,
           future ``compare`` command) don't have to re-read from disk.

        Persistence notes
        -----------------
        ``strategy_state/`` holds the strategy state produced by the LAST
        fold's ``train()`` call - not a canonical "final model". Holdout
        eval / HPO materialisation deliberately re-train fresh from
        ``best_config.yaml`` on the full dev region; the saved state here
        is a convenience artifact for post-hoc inspection. A strategy whose
        ``save()`` isn't implemented raises
        ``NotImplementedError`` from inside the save call - caught here and
        downgraded to a warning so the rest of the artifact tree still
        lands on disk.
        """

        opts = options if options is not None else RunOptions()
        store = opts.store_root if opts.store_root is not None else _DEFAULT_STORE_ROOT
        created_at = datetime.now(UTC)
        git_sha = read_git_sha()
        experiment_id = _make_experiment_id(self.strategy.name, created_at, git_sha)
        run_dir = Path(store) / RUNS_SUBDIR / experiment_id

        seed_all(self.config.seed)

        bars_full = fetch_bars(self.data_source, self.config, self.strategy)
        boundary = resolve_holdout_boundary(
            bars_full,
            holdout_pct=self.config.validation.holdout_pct,
            holdout_start=(
                pd.Timestamp(self.config.validation.holdout_start)
                if self.config.validation.holdout_start is not None
                else None
            ),
        )
        dev = _slice_dev(bars_full, boundary)
        if len(dev) == 0:
            raise ValueError(
                "Experiment.run(): dev slice is empty after holdout reservation; "
                "fix by reducing validation.holdout_pct or widening data.start/end."
            )

        data_hash = compute_data_hash(self.strategy, bars_full, self.config.data.tickers)
        manifest = Manifest(
            experiment_id=experiment_id,
            name=self.config.name,
            created_at=created_at,
            git_sha=git_sha,
            seed=self.config.seed,
            data_hash=data_hash,
            slippage_scenario=self.config.slippage.scenario,
            interval=self.config.data.interval,
            risk_free_rate=self.config.risk_free_rate,
            holdout_start=boundary,
        )

        ensure_model_dir(run_dir)
        with attach_run_log_file(run_dir):
            write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, self.config)
            write_experiment_manifest(run_dir, manifest)
            logger = get_logger(__name__, experiment_id=experiment_id, strategy=self.strategy.name)
            logger.info(
                "fetched %d bars, dev=%d, holdout_start=%s",
                len(bars_full),
                len(dev),
                boundary.isoformat() if boundary is not None else None,
            )

            checkpoint_root = run_dir / EXPERIMENT_CHECKPOINTS_SUBDIR if opts.checkpoint else None
            fold_results: list[FoldResult] = evaluate_walk_forward(
                strategy=self.strategy,
                bars=dev,
                validator=self.validator,
                engine=self.engine,
                slippage=self.slippage,
                interval=self.config.data.interval,
                risk_free_rate=self.config.risk_free_rate,
                feature_pipeline_factory=self.feature_pipeline_factory,
                progress=opts.progress,
                checkpoint_root=checkpoint_root,
                compute_feature_importance=opts.compute_feature_importance,
            )
            folds = tuple(FoldRecord.from_fold_result(fr) for fr in fold_results)

            json_io.write_jsonl(run_dir / FOLD_RESULTS_JSONL, (f.to_dict() for f in folds))
            json_io.write(
                run_dir / EXPERIMENT_METRICS_JSON,
                aggregate_folds(
                    folds,
                    annualization_factor=self.config.data.interval.annualization_factor(),
                    risk_free_rate=self.config.risk_free_rate,
                ).to_dict(),
            )

            fold_importances = [
                fr.feature_importance for fr in fold_results if fr.feature_importance is not None
            ]
            if fold_importances:
                json_io.write(
                    run_dir / FEATURE_IMPORTANCE_JSON,
                    build_importance_artifact(fold_importances),
                )

            _maybe_save_strategy(self.strategy, run_dir / EXPERIMENT_STRATEGY_SUBDIR)

            result = ExperimentResult(
                experiment_id=experiment_id,
                folds=folds,
                manifest=manifest,
            )

            if opts.write_report:
                from src.visualization.strategy_reporter import StrategyReporter

                StrategyReporter().generate_full_report(
                    result, run_dir, publish_label=opts.publish_label
                )
                logger.info("report generated under %s", run_dir)

            if folds:
                last_curve = folds[-1].equity_curve
                logger.info(
                    "%d folds complete, last equity=%.4f",
                    len(folds),
                    last_curve[-1] if last_curve else float("nan"),
                )

            return result


def _maybe_save_strategy(strategy: IStrategy, path: Path) -> None:
    """
    Call ``strategy.save(path)`` if supported; log + skip on NotImplementedError.

    Strategies without a ``save()`` override (and tests' ad-hoc strategies)
    shouldn't block the rest of the artifact tree from landing.
    The full strategy state is never the user's only handle - fold_results
    + config are enough to reproduce via ``experiment run``.
    """

    try:
        strategy.save(path)
    except NotImplementedError:
        _module_logger.warning(
            "%s.save() not implemented - skipping strategy_state/ artifact.",
            type(strategy).__name__,
        )
    except RuntimeError as e:
        _module_logger.warning("strategy save failed (%s); continuing run.", e)
