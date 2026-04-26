"""The wired, ready-to-run experiment primitive.

``Experiment`` is a frozen bundle of every component that participates in a
single walk-forward run: data source, strategy, validator, engine, slippage,
and an optional feature-pipeline FACTORY. It is produced by
:func:`src.orchestration.builder.build_experiment` from a validated
:class:`ExperimentConfig`.

Why the feature pipeline is a factory, not an instance
------------------------------------------------------
Feature pipelines (e.g. :class:`FeatureEngineeringPipeline`) enforce a
``fit_once`` guard on their scaler — a second ``fit()`` raises
``LeakageError``. A walk-forward run needs to fit the scaler PER FOLD on
``fold.train`` only; fitting once on the full dev region would leak later
folds' test-window statistics into earlier folds' features. A single
instance cannot satisfy both constraints. A factory closure captures the
config-derived kwargs and produces a fresh instance whenever the caller
asks — one per fold.

The strategy stays as an instance because each ``IStrategy.train()``
implementation is contracted to reset its own fit state from scratch.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_engine import SlippageConfig
from src.analysis.metrics_aggregator import aggregate_folds
from src.core import json_io
from src.core.config import ExperimentConfig, write_frozen_yaml
from src.core.constants import PAIRS_LEG_SUFFIXES
from src.core.logging import get_logger
from src.core.persistence import (
    EXPERIMENT_CHECKPOINTS_SUBDIR,
    EXPERIMENT_CONFIG_YAML,
    EXPERIMENT_METRICS_JSON,
    EXPERIMENT_STRATEGY_SUBDIR,
    FOLD_RESULTS_JSONL,
    RUNS_SUBDIR,
    ensure_model_dir,
    write_experiment_manifest,
)
from src.core.seeding import seed_all
from src.core.temporal import WalkForwardValidator, resolve_holdout_boundary
from src.data.fingerprint import fingerprint_bars, fingerprint_pair_bars
from src.data.interface import IDataSource
from src.engine.interface import IBacktestEngine
from src.engine.walk_forward import FoldResult, evaluate_walk_forward
from src.features.interface import IFeaturePipeline
from src.orchestration.git_info import read_git_sha
from src.orchestration.manifest import Manifest, PretrainedLeafRecord
from src.orchestration.types import ExperimentResult, FoldRecord
from src.strategies.interface import IStrategy

# ``StrategyReporter`` is lazy-imported inside ``run()`` when ``write_report``
# is True — matplotlib's cold-import tree (~4s incl. pyplot + PIL + numpy
# cascades) is substantial and `--no-report` runs (e.g. HPO trials where the
# tuner drives reporting at the study level) shouldn't pay it. The lazy
# import mirrors ``seed_all``'s lazy torch import for the same reason.

_module_logger = get_logger(__name__)

_DEFAULT_STORE_ROOT = Path("experiment_results")
_EXPERIMENT_ID_SUFFIX_BYTES = 4  # → 8 hex chars, 2^32 combos; low collision risk at ≤10³ runs/s


def _make_experiment_id(strategy_name: str, created_at: datetime, git_sha: str) -> str:
    """Compose a unique experiment id: ``{utc_ts}_{strategy}_{sha}_{rand}``.

    Random suffix (hex-encoded cryptographic bytes) disambiguates two
    invocations in the same second + same strategy + same sha — matters for
    HPO parallelism and ``experiment compare`` subprocess fan-out.
    """
    ts = created_at.strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(_EXPERIMENT_ID_SUFFIX_BYTES)
    return f"{ts}_{strategy_name}_{git_sha}_{suffix}"


def _fetch_bars(data_source: IDataSource, cfg: ExperimentConfig) -> pd.DataFrame:
    """Fetch OHLCV bars for a 1-ticker (single-asset) or 2-ticker (pairs) run.

    Two-ticker mode inner-joins on shared timestamps and suffixes the OHLCV
    columns ``_a`` / ``_b``; >2 tickers is rejected — no current strategy
    consumes more than two legs.
    """
    tickers = cfg.data.tickers
    if len(tickers) == 0:
        raise ValueError(
            "ExperimentConfig.data.tickers must be non-empty; fix by listing at least one ticker."
        )
    if len(tickers) == 1:
        return data_source.fetch(tickers[0], cfg.data.start, cfg.data.end, cfg.data.interval)
    if len(tickers) == 2:
        return _fetch_pair_bars(data_source, cfg, tickers[0], tickers[1])
    raise ValueError(
        f"ExperimentConfig.data.tickers supports 1 ticker (single-asset) or 2 "
        f"tickers (pairs); got {len(tickers)}: {tickers}. Fix by trimming "
        f"data.tickers — three-leg strategies are not in scope."
    )


def _fetch_pair_bars(
    data_source: IDataSource,
    cfg: ExperimentConfig,
    ticker_a: str,
    ticker_b: str,
) -> pd.DataFrame:
    """Fetch both legs of a pair, inner-join, suffix the OHLCV columns."""
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


def _slice_dev(bars: pd.DataFrame, boundary: pd.Timestamp | None) -> pd.DataFrame:
    """Return the dev region — everything strictly before ``boundary``.

    ``boundary`` is the first bar OF the holdout (see
    ``resolve_holdout_boundary``). ``None`` disables the reservation.
    """
    if boundary is None:
        return bars
    return bars.loc[bars.index < boundary]


def _write_fold_jsonl(path: Path, folds: tuple[FoldRecord, ...]) -> None:
    """Write one JSON object per fold, newline-separated.

    JSONL over one-big-JSON-array so ``tail -f`` works during long runs and
    ``wc -l`` gives you the fold count without parsing.
    """
    import json

    with path.open("w", encoding="utf-8") as f:
        for fold in folds:
            f.write(json.dumps(fold.to_dict(), sort_keys=True))
            f.write("\n")


@dataclass(frozen=True)
class RunOptions:
    """Per-invocation knobs for :meth:`Experiment.run`.

    Bundled so the run() signature doesn't grow a flag per concern. Defaults
    match the no-arguments behaviour: write to ``experiment_results/``, emit
    the strategy report, no progress bar, no per-fold checkpoints.
    """

    store_root: Path | None = None
    write_report: bool = True
    progress: bool = False
    checkpoint: bool = False


@dataclass(frozen=True)
class Experiment:
    """A fully-wired walk-forward experiment.

    Prefer constructing via :func:`build_experiment` — direct instantiation
    is intentional for tests that want to inject mocks per component.
    """

    config: ExperimentConfig
    data_source: IDataSource
    strategy: IStrategy
    validator: WalkForwardValidator
    engine: IBacktestEngine
    slippage: SlippageConfig
    feature_pipeline_factory: Callable[[], IFeaturePipeline] | None = None
    pretrained_leaf_records: tuple[PretrainedLeafRecord, ...] = ()

    def run(self, options: RunOptions | None = None) -> ExperimentResult:
        """Execute the full walk-forward loop and persist every artifact.

        Pipeline:
        1. Seed numpy/torch/random deterministically from ``config.seed``.
        2. Fetch bars via the wired ``IDataSource`` (with cache).
        3. Resolve the holdout boundary per the config's ``validation`` block.
        4. Slice dev (bars strictly before the boundary) — the walk-forward
           splitter sees dev only. The holdout region is reserved for the
           post-thesis OOS evaluation and is NEVER touched here.
        5. Compute ``data_hash = fingerprint_bars(bars_full)`` so future
           holdout-eval commands can refuse on vendor drift.
        6. Create ``store_root/runs/<experiment_id>/`` and write the frozen
           ``config.yaml`` + ``manifest.json`` BEFORE any compute — so a
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
        fold's ``train()`` call — not a canonical "final model". Holdout
        eval / HPO materialisation deliberately re-train fresh from
        ``best_config.yaml`` on the full dev region; the saved state here
        is a convenience artifact for post-hoc inspection. A strategy whose
        ``save()`` isn't implemented raises
        ``NotImplementedError`` from inside the save call — caught here and
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

        bars_full = _fetch_bars(self.data_source, self.config)
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

        data_hash = (
            fingerprint_pair_bars(bars_full)
            if len(self.config.data.tickers) == 2
            else fingerprint_bars(bars_full)
        )
        manifest = Manifest(
            experiment_id=experiment_id,
            name=self.config.name,
            created_at=created_at,
            git_sha=git_sha,
            seed=self.config.seed,
            data_hash=data_hash,
            slippage_scenario=self.config.slippage.scenario,
            holdout_start=boundary,
            pretrained_leaves=self.pretrained_leaf_records,
        )

        ensure_model_dir(run_dir)
        write_frozen_yaml(run_dir / EXPERIMENT_CONFIG_YAML, self.config)
        write_experiment_manifest(run_dir, manifest)
        logger = get_logger(__name__, experiment_id=experiment_id, strategy=self.strategy.name)
        logger.info(
            "fetched %d bars, dev=%d, holdout_start=%s",
            len(bars_full),
            len(dev),
            boundary.isoformat() if boundary is not None else None,
        )

        # Mid-fit checkpoints land under ``<run_dir>/checkpoints/fold_<i>/``
        # only when the caller opts in — most runs don't need them and skip
        # the per-epoch / per-round disk writes entirely.
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
        )
        folds = tuple(FoldRecord.from_fold_result(fr) for fr in fold_results)

        _write_fold_jsonl(run_dir / FOLD_RESULTS_JSONL, folds)
        json_io.write(run_dir / EXPERIMENT_METRICS_JSON, aggregate_folds(folds).to_dict())
        _maybe_save_strategy(self.strategy, run_dir / EXPERIMENT_STRATEGY_SUBDIR)

        result = ExperimentResult(
            experiment_id=experiment_id,
            folds=folds,
            manifest=manifest,
        )

        if opts.write_report:
            # Lazy: matplotlib's cold import is ~4s; paying it only when
            # reports are actually requested keeps the no-report path light.
            from src.visualization.strategy_reporter import StrategyReporter

            StrategyReporter().generate_full_report(result, run_dir)
            logger.info("report generated under %s", run_dir)

        # Summary line after reports so the INFO trail reads
        # "fetched → walked → reported → summary" in order.
        if folds:
            last_curve = folds[-1].equity_curve
            logger.info(
                "%d folds complete, last equity=%.4f",
                len(folds),
                last_curve[-1] if last_curve else float("nan"),
            )

        return result


def _maybe_save_strategy(strategy: IStrategy, path: Path) -> None:
    """Call ``strategy.save(path)`` if supported; log + skip on NotImplementedError.

    Strategies without a ``save()`` override (and tests' ad-hoc strategies)
    shouldn't block the rest of the artifact tree from landing.
    The full strategy state is never the user's only handle — fold_results
    + config are enough to reproduce via ``experiment run``.
    """
    try:
        strategy.save(path)
    except NotImplementedError:
        _module_logger.warning(
            "%s.save() not implemented — skipping strategy_state/ artifact.",
            type(strategy).__name__,
        )
    except RuntimeError as e:
        _module_logger.warning("strategy save failed (%s); continuing run.", e)
