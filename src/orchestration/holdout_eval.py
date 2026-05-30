"""
End-to-end driver for ``experiment holdout-eval``.

A post-HPO/post-run command. Loads a completed source (a single
``experiment run`` directory or an ``experiment tune`` study), refits a
fresh strategy on the FULL dev region, evaluates once on the holdout
region, and writes a one-shot result bundle. Walk-forward is *not*
re-run here - holdout-eval produces the single honest OOS number that
the dev/HPO loop deliberately never sees.

The CLI layer in ``scripts/experiment.py`` is a thin click wrapper
around :func:`run_holdout_eval`.

Why fresh fit on full dev (not reuse the source run's trained state)
--------------------------------------------------------------------
The source run's ``strategy_state/`` reflects only the *last fold*'s
training window - strictly less data than full dev. Reusing it would
trade away a few minutes of compute against measurable model quality.
Full-dev refit gives the strongest honest-OOS signal the framework can
produce.

Anti-leakage tripwires fired here, in this exact order
------------------------------------------------------
1. ``manifest.holdout_start`` is non-None (else the source had no
   holdout reservation - refuse).
2. Re-fetched ``data_hash == manifest.data_hash`` (else vendor drift -
   refuse, the boundary timestamp would slide silently).
3. ``resolve_holdout_boundary`` validates the pinned timestamp is
   present in the re-fetched index (a missing bar at the pin would
   trip a separate :class:`LeakageError` from the resolver).
4. ``TemporalSplit(train=dev, test=holdout, split_date=boundary)``
   re-asserts ``dev.index[-1] < holdout.index[0]`` after the slice.
5. Deep metadata check (same loop walk-forward uses) - every tracked
   training metadata validates that no component saw the holdout window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from quant_engine import MetricsCalculator
from src.analysis.baselines import BaselineResult, compute_buy_and_hold
from src.analysis.significance import BootstrapCI, bootstrap_sharpe_ci
from src.core import json_io
from src.core.config import load_experiment_config
from src.core.logging import get_logger
from src.core.persistence import (
    EXPERIMENT_CONFIG_YAML,
    HOLDOUT_EVAL_JSON,
    HOLDOUT_EVALS_SUBDIR,
    HPO_TRIALS_RUNS_SUBDIR,
    ensure_model_dir,
    read_experiment_manifest,
)
from src.core.temporal import TemporalSplit, resolve_holdout_boundary
from src.data.fingerprint import assert_data_hash_matches
from src.engine.scenarios import SlippageScenario
from src.engine.walk_forward import (
    dispatch_engine_run,
    dispatch_primary_ohlcv,
    validate_deep_metadata,
)
from src.optimization.checkpointing import BEST_CONFIG_YAML_NAME, TRIAL_ARTIFACTS_SUBDIR
from src.orchestration.builder import build_experiment
from src.orchestration.experiment import compute_data_hash, fetch_bars
from src.orchestration.git_info import read_git_sha

_logger = get_logger(__name__)

SourceKind = Literal["run", "hpo"]


@dataclass(frozen=True)
class HoldoutEvalResult:
    """
    In-memory output of :func:`run_holdout_eval`.

    Mirrors :class:`ExperimentResult`'s shape (one record + provenance)
    so callers - tests, future reporters, the Streamlit workbench -
    don't have to round-trip through disk to read the metrics.

    ``sharpe_ci`` carries the stationary-bootstrap 95% CI on the
    strategy's holdout Sharpe; ``buy_and_hold`` is the long-only
    reference baseline computed on the same holdout window under the
    same slippage scenario, for excess-over-baseline framing.
    """

    out_name: str
    source_kind: SourceKind
    source_id: str
    source_path: str
    holdout_start: pd.Timestamp
    data_hash: str
    git_sha: str
    created_at: datetime
    n_dev_bars: int
    n_holdout_bars: int
    slippage_scenario: SlippageScenario
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    equity_curve: tuple[float, ...]
    sharpe_ci: BootstrapCI
    buy_and_hold: BaselineResult

    def to_dict(self) -> dict[str, object]:
        """
        Serialise to the canonical ``holdout_eval.json`` payload.

        ``is_holdout_eval: true`` is the discriminator - automated tooling
        (study orchestrator, future ``experiment study`` consolidator)
        uses it to refuse feeding this bundle back into HPO.
        """

        return {
            "is_holdout_eval": True,
            "out_name": self.out_name,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "holdout_start": self.holdout_start.isoformat(),
            "data_hash": self.data_hash,
            "git_sha": self.git_sha,
            "created_at": self.created_at.isoformat(),
            "n_dev_bars": self.n_dev_bars,
            "n_holdout_bars": self.n_holdout_bars,
            "slippage_scenario": self.slippage_scenario.value,
            "metrics": {
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "annualized_volatility": self.annualized_volatility,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate,
                "trade_count": self.trade_count,
                "sharpe_ci": self.sharpe_ci.to_dict(),
            },
            "equity_curve": list(self.equity_curve),
            "buy_and_hold": self.buy_and_hold.to_dict(),
        }


@dataclass(frozen=True)
class _ResolvedSource:
    """
    Internal: a source kind + the two paths the workflow needs.

    ``config_path`` points at the source's frozen ExperimentConfig YAML
    (``config.yaml`` for a run, ``best_config.yaml`` for an HPO study).
    ``manifest_dir`` is the directory whose ``manifest.json`` carries the
    canonical ``holdout_start`` + ``data_hash`` (the run dir directly, or
    one of the HPO study's per-trial artifact dirs).
    """

    kind: SourceKind
    source_id: str
    source_path: Path
    config_path: Path
    manifest_dir: Path


def resolve_source(
    *,
    run_dir: Path | None,
    hpo_dir: Path | None,
) -> _ResolvedSource:
    """
    Resolve a CLI source pair into the two on-disk anchors the workflow needs.

    Exactly one of ``run_dir`` / ``hpo_dir`` must be set - the CLI layer
    enforces mutual exclusion and this guard is defence-in-depth.

    For ``--hpo-best``: discovers the per-trial manifest by reading the
    first lexicographic entry under ``<hpo_dir>/trials_artifacts/runs/``.
    All trials under one study share the same dev/holdout boundary (same
    base config, same validation block), so any single trial's manifest
    is canonical for ``data_hash`` + ``holdout_start``. A future
    study-level manifest writer in the tuner would supersede this, but
    today walking the trial dirs avoids cross-module surgery.
    """

    if (run_dir is None) == (hpo_dir is None):
        raise ValueError(
            "resolve_source requires exactly one of run_dir / hpo_dir; "
            "fix the caller (the CLI guard should prevent this)."
        )
    if run_dir is not None:
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"--run-dir {run_dir} does not exist or is not a directory; "
                f"check the path against your experiment_results/runs/ tree."
            )
        return _ResolvedSource(
            kind="run",
            source_id=run_dir.name,
            source_path=run_dir,
            config_path=run_dir / EXPERIMENT_CONFIG_YAML,
            manifest_dir=run_dir,
        )

    assert hpo_dir is not None
    if not hpo_dir.is_dir():
        raise FileNotFoundError(
            f"--hpo-best {hpo_dir} does not exist or is not a directory; "
            f"check the path against your experiment_results/hpo/ tree."
        )
    best_config_path = hpo_dir / BEST_CONFIG_YAML_NAME
    if not best_config_path.is_file():
        raise FileNotFoundError(
            f"missing {BEST_CONFIG_YAML_NAME} under {hpo_dir}; the HPO study "
            f"may have no completed trials yet - run the study to completion "
            f"or pass --run-dir against a finished run instead."
        )
    trials_root = hpo_dir / TRIAL_ARTIFACTS_SUBDIR / HPO_TRIALS_RUNS_SUBDIR
    if not trials_root.is_dir():
        raise FileNotFoundError(
            f"missing {trials_root} under {hpo_dir}; no trial has produced "
            f"artifacts yet - run the study to completion."
        )
    first_trial = next((p for p in trials_root.iterdir() if p.is_dir()), None)
    if first_trial is None:
        raise FileNotFoundError(
            f"no trial subdirectories under {trials_root}; HPO produced no "
            f"completed-trial artifacts to anchor the boundary against."
        )
    return _ResolvedSource(
        kind="hpo",
        source_id=hpo_dir.name,
        source_path=hpo_dir,
        config_path=best_config_path,
        manifest_dir=first_trial,
    )


def run_holdout_eval(
    *,
    source: _ResolvedSource,
    out_name: str,
    store_root: Path,
) -> tuple[HoldoutEvalResult, Path]:
    """
    Drive the full one-shot honest-OOS evaluation pipeline.

    The five anti-leakage tripwires (see module docstring) fire in order
    along the workflow below; the rest is straight-line. Returns the
    in-memory result + the artifact directory so the CLI layer can echo
    the path without recomputing it.
    """

    out_dir = store_root / HOLDOUT_EVALS_SUBDIR / out_name
    cfg = load_experiment_config(source.config_path)
    manifest = read_experiment_manifest(source.manifest_dir)
    if manifest.holdout_start is None:
        raise ValueError(
            f"source manifest at {source.manifest_dir} has holdout_start=None; "
            f"the source run reserved no holdout region, so honest OOS "
            f"evaluation is undefined. Re-run the source with "
            f"validation.holdout_pct > 0 (or validation.holdout_start pinned)."
        )

    experiment = build_experiment(cfg)
    bars_full = fetch_bars(experiment.data_source, cfg, experiment.strategy)

    actual_data_hash = compute_data_hash(experiment.strategy, bars_full, cfg.data.tickers)
    assert_data_hash_matches(
        actual_data_hash, manifest.data_hash, context="holdout boundary anchor"
    )

    boundary = resolve_holdout_boundary(bars_full, holdout_start=manifest.holdout_start)
    if boundary is None:
        raise RuntimeError(
            f"resolve_holdout_boundary returned None for a pinned "
            f"holdout_start={manifest.holdout_start}; this is a contract "
            f"violation in the resolver."
        )

    dev = bars_full.loc[bars_full.index < boundary]
    holdout = bars_full.loc[bars_full.index >= boundary]
    if len(dev) == 0:
        raise ValueError(
            f"holdout-eval: dev slice is empty (boundary {boundary} is at "
            f"or before the data start); the source run cannot have produced "
            f"valid dev folds."
        )
    if len(holdout) == 0:
        raise ValueError(
            f"holdout-eval: holdout slice is empty (boundary {boundary} is "
            f"after the data end); refetched bars may have been truncated."
        )
    TemporalSplit(train=dev, test=holdout, split_date=boundary)

    _logger.info(
        "holdout-eval %s: dev=%d bars [%s..%s], holdout=%d bars [%s..%s]",
        out_name,
        len(dev),
        dev.index[0],
        dev.index[-1],
        len(holdout),
        holdout.index[0],
        holdout.index[-1],
    )

    if experiment.feature_pipeline_factory is not None:
        pipeline = experiment.feature_pipeline_factory()
        train_frame = pipeline.fit_transform(dev)
        test_frame = pipeline.transform(holdout)
    else:
        train_frame = dev
        test_frame = holdout

    experiment.strategy.train(train_frame)
    validate_deep_metadata(experiment.strategy, test_data=test_frame)

    signals = experiment.strategy.generate_signals(test_frame)
    raw = dispatch_engine_run(
        experiment.engine, experiment.strategy, holdout, signals, experiment.slippage
    )

    annualization = cfg.data.interval.annualization_factor()
    metrics = MetricsCalculator.compute(raw.equity_curve, annualization, cfg.risk_free_rate)

    equity_arr = np.asarray(raw.equity_curve, dtype=np.float64)
    if np.any(equity_arr <= 0.0):
        raise ValueError(
            f"holdout-eval {out_name}: equity curve contains a non-positive bar; "
            f"the simple-return divisor would be zero or negative and poison the "
            f"bootstrap. Investigate the strategy for blow-up behaviour."
        )
    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe_ci = bootstrap_sharpe_ci(returns)

    bah_bars = dispatch_primary_ohlcv(experiment.strategy, holdout)
    bah = compute_buy_and_hold(
        bah_bars,
        slippage=experiment.slippage,
        interval=cfg.data.interval,
        engine=experiment.engine,
        risk_free_rate=cfg.risk_free_rate,
    )

    result = HoldoutEvalResult(
        out_name=out_name,
        source_kind=source.kind,
        source_id=source.source_id,
        source_path=str(source.source_path),
        holdout_start=boundary,
        data_hash=actual_data_hash,
        git_sha=read_git_sha(),
        created_at=datetime.now(UTC),
        n_dev_bars=len(dev),
        n_holdout_bars=len(holdout),
        slippage_scenario=cfg.slippage.scenario,
        total_return=raw.total_return,
        annualized_return=metrics.annualized_return,
        annualized_volatility=metrics.annualized_volatility,
        sharpe_ratio=metrics.sharpe_ratio,
        sortino_ratio=metrics.sortino_ratio,
        calmar_ratio=metrics.calmar_ratio,
        max_drawdown=metrics.max_drawdown,
        win_rate=metrics.win_rate,
        trade_count=raw.trade_count,
        equity_curve=tuple(raw.equity_curve.tolist()),
        sharpe_ci=sharpe_ci,
        buy_and_hold=bah,
    )

    ensure_model_dir(out_dir)
    json_io.write(out_dir / HOLDOUT_EVAL_JSON, result.to_dict())
    _logger.info(
        "holdout-eval %s done: sharpe=%.4f total_return=%.4f max_dd=%.4f",
        out_name,
        result.sharpe_ratio,
        result.total_return,
        result.max_drawdown,
    )
    return result, out_dir


__all__ = [
    "HoldoutEvalResult",
    "SourceKind",
    "resolve_source",
    "run_holdout_eval",
]
