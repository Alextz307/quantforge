"""Train one leaf model standalone on a single window.

Complements the strategy-level walk-forward orchestrator (``Experiment``)
with a one-shot trainer sized for pretrained-leaf injection: fetch data,
slice to ``[train_start, train_end]``, optionally apply a feature
pipeline, fit the model, and return ``(model, data_hash, manifest)`` so
a caller (the ``experiment train-model`` CLI) can persist the artifact
bundle via :func:`src.orchestration.model_artifact.save_model_artifact`.

Currently supported models — the two composites whose owning strategies
accept pretrained-leaf injection:

* ``hybrid_return`` — ``HybridReturnModel`` trained on log returns
* ``hybrid_volatility`` — ``HybridVolatilityModel`` trained on annualized
  Garman-Klass realized volatility

Any other ``cfg.model.name`` raises :class:`NotImplementedError` with a
pointer to the supported set. Extend the dispatch dict below when a new
strategy gains pretrained-leaf support.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from src.core.constants import DEFAULT_REALIZED_VOL_WINDOW
from src.core.registry import (
    classifier_registry,
    data_source_registry,
    feature_registry,
    model_registry,
)
from src.core.seeding import seed_all
from src.core.types import ModelKind
from src.core.utils import annualized_garman_klass, compute_log_returns
from src.data.fingerprint import fingerprint_bars
from src.orchestration.git_info import read_git_sha
from src.orchestration.model_artifact import (
    ModelArtifactManifest,
    build_model_artifact_manifest,
)

if TYPE_CHECKING:
    from src.core.config import StandaloneModelConfig
    from src.models.interface import IClassifier, IPredictor

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StandaloneTrainingResult:
    """Bundle returned by :func:`train_model_standalone`.

    ``model`` is the fitted leaf (already has ``training_metadata`` set
    via the model's own ``fit()``); ``manifest`` is the
    ``ModelArtifactManifest`` ready to write alongside the model; the
    caller persists both via ``save_model_artifact``.
    """

    model: IPredictor | IClassifier
    manifest: ModelArtifactManifest


def _slice_training_window(
    bars: pd.DataFrame,
    *,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp | None,
) -> pd.DataFrame:
    if train_start is None and train_end is None:
        return bars
    lo = pd.Timestamp(train_start) if train_start is not None else bars.index[0]
    hi = pd.Timestamp(train_end) if train_end is not None else bars.index[-1]
    sliced = bars.loc[lo:hi]
    if len(sliced) == 0:
        raise ValueError(
            f"training window [{lo}, {hi}] selected zero bars from fetched "
            f"range [{bars.index[0]}, {bars.index[-1]}]; widen the window or "
            f"refetch with a longer range."
        )
    return sliced


_TargetComputer = Callable[[pd.DataFrame, "StandaloneModelConfig"], tuple[pd.DataFrame, pd.Series]]


def _target_for_hybrid_return(
    bars: pd.DataFrame, _cfg: StandaloneModelConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Log returns of close, dropped to their valid support."""
    log_returns = compute_log_returns(bars["close"]).dropna()
    return bars.loc[log_returns.index], log_returns


def _target_for_hybrid_volatility(
    bars: pd.DataFrame, cfg: StandaloneModelConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Annualized Garman-Klass target — same recipe the owning strategy uses."""
    window_raw = cfg.model.params.get("realized_vol_window", DEFAULT_REALIZED_VOL_WINDOW)
    if not isinstance(window_raw, int):
        raise ValueError(
            f"hybrid_volatility model.params.realized_vol_window must be an int, "
            f"got {type(window_raw).__name__}={window_raw!r}"
        )
    target = annualized_garman_klass(bars, window=window_raw, interval=cfg.data.interval).dropna()
    return bars.loc[target.index], target


_TARGET_DISPATCH: dict[str, _TargetComputer] = {
    "hybrid_return": _target_for_hybrid_return,
    "hybrid_volatility": _target_for_hybrid_volatility,
}


def train_model_standalone(cfg: StandaloneModelConfig) -> StandaloneTrainingResult:
    """Fetch data, slice, apply features, fit ``cfg.model``, return the bundle.

    Every step mirrors the corresponding step in
    :meth:`Experiment.run` so two artifacts trained on the same window
    + seed land with the same ``data_hash`` regardless of whether they
    came from the experiment runner or the standalone trainer.

    Order is deliberate:

    1. Seed everything — reproducibility precondition.
    2. Fetch raw OHLCV — IO, may hit network.
    3. Slice to ``[train_start, train_end]`` — cheap filter.
    4. Fingerprint the sliced raw bars — the hash represents "what bar
       stream the model trained on", BEFORE features, so two trainers
       that apply different feature pipelines to the same bars still
       produce the same hash (and a consumer can distinguish via the
       feature config).
    5. Apply features (if configured) — produces the DataFrame the
       model's ``fit()`` actually consumes.
    6. Compute target + align bars — per-model dispatch.
    7. Fit model — populates ``training_metadata`` atomically inside
       the model's own ``fit()``.
    8. Build manifest + return. ``save_model_artifact`` writes it.
    """
    seed_all(cfg.seed)

    # 2: fetch — single-ticker only, same contract as Experiment.run
    if len(cfg.data.tickers) != 1:
        raise NotImplementedError(
            f"standalone training accepts exactly one ticker, got {cfg.data.tickers}; "
            f"the pretrained-leaf seam doesn't currently compose multi-ticker bundles."
        )
    data_source = data_source_registry.create_from_config(cfg.data.source)
    bars = data_source.fetch(
        cfg.data.tickers[0],
        cfg.data.start,
        cfg.data.end,
        cfg.data.interval,
    )

    # 3: slice
    sliced = _slice_training_window(
        bars,
        train_start=pd.Timestamp(cfg.train_start) if cfg.train_start is not None else None,
        train_end=pd.Timestamp(cfg.train_end) if cfg.train_end is not None else None,
    )

    # 4: hash the raw sliced bars (pre-features)
    data_hash = fingerprint_bars(sliced)

    # 5: features
    if cfg.features is not None:
        pipeline = feature_registry.create_from_config(cfg.features)
        feature_frame = pipeline.fit_transform(sliced)
    else:
        feature_frame = sliced

    # 6: target
    target_fn = _TARGET_DISPATCH.get(cfg.model.name)
    if target_fn is None:
        raise NotImplementedError(
            f"standalone training for model '{cfg.model.name}' is not supported; "
            f"supported in this release: {sorted(_TARGET_DISPATCH)}. "
            f"Add a target computer to _TARGET_DISPATCH to extend."
        )
    aligned, target = target_fn(feature_frame, cfg)

    # 7: fit — inject cfg.data.interval (source of truth) into model params
    # so the model's training_metadata matches downstream expectations
    # without forcing users to spell it twice in YAML.
    params = dict(cfg.model.params)
    params["interval"] = cfg.data.interval
    model_cls: type[IPredictor] | type[IClassifier]
    if cfg.model_kind == ModelKind.PREDICTOR:
        model_cls = model_registry.get(cfg.model.name)
    else:
        model_cls = classifier_registry.get(cfg.model.name)
    model: IPredictor | IClassifier = model_cls(**params)
    model.fit(aligned, target)
    _logger.info(
        "standalone training complete: model=%s kind=%s bars=%d data_hash=%s...",
        cfg.model.name,
        cfg.model_kind.value,
        len(aligned),
        data_hash[:8],
    )

    # 8: manifest
    manifest = build_model_artifact_manifest(
        name=cfg.name,
        model_name=cfg.model.name,
        model_kind=cfg.model_kind,
        git_sha=read_git_sha(),
        seed=cfg.seed,
        data_hash=data_hash,
    )
    return StandaloneTrainingResult(model=model, manifest=manifest)
