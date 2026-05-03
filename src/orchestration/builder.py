"""Config → wired :class:`Experiment` factory.

Resolves every :class:`ComponentConfig` against its global registry and
instantiates the concrete validator, engine, and slippage scenario. Kept
deliberately thin: composite-wiring logic (strategies that own their own
leaf models or feature pipelines) is handled inside each strategy's own
constructor, not reinvented here.

Pretrained-leaf handling: when ``cfg.pretrained_leaves`` is non-empty,
each artifact directory is loaded here (via ``load_model_artifact``)
before the strategy ctor runs. The loaded leaves flow into the strategy
via its ``pretrained_leaves`` ctor kwarg; the strategy's own
``validate_pretrained_leaf`` catches interval / features / lookback
mismatches before the first fold.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from src.core.config import ComponentConfig, ExperimentConfig
from src.core.registry import (
    data_source_registry,
    feature_registry,
    strategy_registry,
)
from src.core.temporal import WalkForwardValidator
from src.engine.cpp_engine import CppBacktestEngine
from src.engine.scenarios import SLIPPAGE_SCENARIOS
from src.features.interface import IFeaturePipeline
from src.orchestration.experiment import Experiment
from src.orchestration.manifest import PretrainedLeafRecord
from src.orchestration.model_artifact import load_model_artifact
from src.strategies.interface import IStrategy


def _make_feature_pipeline_factory(
    features_cfg: ComponentConfig,
) -> Callable[[], IFeaturePipeline]:
    """Capture ``features_cfg`` in a closure so callers get a fresh instance per call.

    Split out so the closure binds ``features_cfg`` once, not a loop variable —
    avoids the late-binding trap if this ever ends up inside a loop.
    """
    return lambda: feature_registry.create_from_config(features_cfg)


def _load_pretrained_leaves(
    cfg: ExperimentConfig,
) -> tuple[dict[str, object], tuple[PretrainedLeafRecord, ...]]:
    """Load each artifact in ``cfg.pretrained_leaves`` + build manifest records.

    Returns ``(loaded_models, records)``:

    * ``loaded_models`` goes to the strategy ctor — keyed by leaf key,
      each value is the in-memory model instance. The strategy ctor runs
      ``validate_pretrained_leaf`` on each.
    * ``records`` goes to the experiment manifest for provenance —
      ``(key, artifact_path, artifact_data_hash, leaf_train_end)``. The
      ``train_end`` comes from the leaf's own ``training_metadata``, not
      the artifact manifest (which doesn't duplicate it); ``data_hash``
      comes from the artifact manifest (the hash of the training bars).

    Failure at the builder boundary fails ``experiment run`` fast rather
    than surfacing mid-fold.
    """
    loaded: dict[str, object] = {}
    records: list[PretrainedLeafRecord] = []
    for key, path in cfg.pretrained_leaves.items():
        model, artifact_manifest = load_model_artifact(path)
        loaded[key] = model
        meta = getattr(model, "training_metadata", None)
        if meta is None:
            raise ValueError(
                f"pretrained_leaves['{key}']: loaded model from {path} has "
                f"no training_metadata; artifact may be corrupt or was saved "
                f"before fit() completed."
            )
        records.append(
            PretrainedLeafRecord(
                key=key,
                path=str(path),
                data_hash=artifact_manifest.data_hash,
                train_start=meta.train_start,
                train_end=meta.train_end,
            )
        )
    return loaded, tuple(records)


def _validate_strategy_data_shape(cfg: ExperimentConfig) -> None:
    """Cross-check ticker count + shape flags against the strategy class.

    Three valid shapes, all mutually exclusive:

    * **Pairs**: exactly two tickers (one per leg); no feature pipeline.
    * **Multi-feature**: N≥1 tickers; ``cfg.strategy.params['primary_ticker']``
      MUST be in ``cfg.data.tickers``; no feature pipeline (the strategy
      reads the wide frame directly).
    * **Single-asset**: exactly one ticker.

    Any mismatch is rejected here, before any data fetch.
    """
    n_tickers = len(cfg.data.tickers)
    strategy_cls = strategy_registry.get(cfg.strategy.name)
    if not issubclass(strategy_cls, IStrategy):
        raise TypeError(
            f"strategy '{cfg.strategy.name}' resolves to {strategy_cls!r}, "
            f"which does not subclass IStrategy."
        )
    if strategy_cls.is_pairs_strategy and strategy_cls.is_multi_feature_strategy:
        raise TypeError(
            f"strategy '{cfg.strategy.name}' ({strategy_cls.__name__}) sets both "
            f"is_pairs_strategy=True and is_multi_feature_strategy=True; the two "
            f"capability flags are mutually exclusive (different dispatch paths, "
            f"different wide-frame conventions). Fix by clearing one ClassVar."
        )
    if strategy_cls.is_pairs_strategy:
        if n_tickers != 2:
            raise ValueError(
                f"strategy '{cfg.strategy.name}' requires exactly 2 "
                f"tickers (one per leg); got {n_tickers}: {cfg.data.tickers}. "
                f"Fix by listing two tickers under data.tickers, or by "
                f"choosing a single-asset strategy."
            )
        if cfg.features is not None:
            raise ValueError(
                f"strategy '{cfg.strategy.name}' does not consume an "
                f"engineered feature pipeline (it operates directly on the "
                f"two close columns); got features={cfg.features.name!r}. "
                f"Fix by removing the 'features:' block from the config."
            )
    elif strategy_cls.is_multi_feature_strategy:
        if n_tickers < 1:
            raise ValueError(
                f"strategy '{cfg.strategy.name}' is multi-feature and requires "
                f"at least 1 ticker (the primary); got 0. Fix by listing the "
                f"primary plus any feature tickers under data.tickers."
            )
        primary_raw = cfg.strategy.params.get("primary_ticker")
        if not isinstance(primary_raw, str) or not primary_raw:
            raise ValueError(
                f"strategy '{cfg.strategy.name}' is multi-feature; "
                f"strategy.params must contain a non-empty string 'primary_ticker' "
                f"naming the asset to trade. Got primary_ticker={primary_raw!r}. "
                f"Fix by adding 'primary_ticker: <TICKER>' to strategy.params."
            )
        if primary_raw not in cfg.data.tickers:
            raise ValueError(
                f"strategy '{cfg.strategy.name}' declares primary_ticker="
                f"{primary_raw!r}, but data.tickers={list(cfg.data.tickers)} "
                f"does not contain it. Fix by adding {primary_raw!r} to "
                f"data.tickers (the primary must be fetched alongside the "
                f"feature tickers) or by choosing a different primary_ticker."
            )
        if cfg.features is not None:
            raise ValueError(
                f"strategy '{cfg.strategy.name}' is multi-feature and reads "
                f"the wide multi-ticker frame directly; got features="
                f"{cfg.features.name!r}. Fix by removing the 'features:' block "
                f"from the config — multi-feature strategies engineer their "
                f"own cross-asset features inline."
            )
    elif n_tickers != 1:
        raise ValueError(
            f"strategy '{cfg.strategy.name}' is single-asset; expected 1 "
            f"ticker, got {n_tickers}: {cfg.data.tickers}. Fix by trimming "
            f"data.tickers, or by switching to a pairs strategy for two legs."
        )


def build_experiment(cfg: ExperimentConfig) -> Experiment:
    """Instantiate every component referenced by ``cfg`` and bundle into an :class:`Experiment`."""
    _validate_strategy_data_shape(cfg)
    data_source = data_source_registry.create_from_config(cfg.data.source)

    if cfg.pretrained_leaves:
        strategy_cls = strategy_registry.get(cfg.strategy.name)
        if "pretrained_leaves" not in inspect.signature(strategy_cls).parameters:
            raise ValueError(
                f"strategy '{cfg.strategy.name}' ({strategy_cls.__name__}) does not "
                f"accept a 'pretrained_leaves' ctor kwarg; the config carries "
                f"{sorted(cfg.pretrained_leaves)!r} which can't be injected. Fix by "
                f"adding the kwarg to the strategy ctor or removing pretrained_leaves "
                f"from the config."
            )
        loaded_leaves, leaf_records = _load_pretrained_leaves(cfg)
        # IStrategy doesn't declare ``pretrained_leaves`` on the abstract
        # ctor — it's a per-subclass convention (see the ``_leaf_keys``
        # ClassVar on each concrete strategy). The signature check above
        # replaces what a protocol-level declaration would enforce
        # statically; the config-layer validator already rejected unknown
        # keys, and the strategy ctor validates the map shape.
        strategy = strategy_cls(  # type: ignore[call-arg]
            **cfg.strategy.params, pretrained_leaves=loaded_leaves
        )
    else:
        strategy = strategy_registry.create_from_config(cfg.strategy)
        leaf_records = ()

    feature_pipeline_factory: Callable[[], IFeaturePipeline] | None = (
        _make_feature_pipeline_factory(cfg.features) if cfg.features is not None else None
    )
    validator = WalkForwardValidator(
        n_splits=cfg.validation.n_splits,
        test_size=cfg.validation.test_size,
        gap=cfg.validation.gap,
        expanding=cfg.validation.expanding,
        snap_to_day=cfg.validation.snap_to_day,
    )
    engine = CppBacktestEngine()
    slippage = SLIPPAGE_SCENARIOS[cfg.slippage.scenario]

    return Experiment(
        config=cfg,
        data_source=data_source,
        strategy=strategy,
        validator=validator,
        engine=engine,
        slippage=slippage,
        feature_pipeline_factory=feature_pipeline_factory,
        pretrained_leaf_records=leaf_records,
    )
