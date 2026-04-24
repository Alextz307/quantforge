"""Validator + helpers for the pretrained-leaf injection seam.

Strategies that own ML leaves (``ReturnForecastStrategy`` owns
``HybridReturnModel``, ``VolatilityTargetingStrategy`` owns
``HybridVolatilityModel``) can be constructed with
``pretrained_leaves={"<leaf_key>": <loaded_leaf>}`` to freeze the leaf
across folds: the strategy's own state re-fits fresh per fold, but the
leaf's weights, scaler, and ``training_metadata`` stay pinned.

This module owns the invariant checks that run BEFORE the strategy
accepts the injected leaf. Failing here raises ``ValueError`` with an
actionable remediation — silent mismatches (wrong interval, wrong
features, wrong lookback) would produce nonsensical predictions for
hundreds of bars before any downstream assertion trips.

Temporal invariants (leaf.train_end < fold.train_start strict, leaf.
train_end < fold.test_start always) live in
:func:`src.engine.walk_forward._validate_deep_metadata` — they're
enforced per fold, not at injection time, and use the
``TrackedMetadata.is_pretrained`` flag to distinguish a pinned leaf from
the strategy's fresh per-fold fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.core.leaf_keys import describe_supported_leaf_keys
from src.core.types import Interval


def normalize_pretrained_leaves(
    pretrained_leaves: Mapping[str, object] | None,
    supported_keys: frozenset[str],
    strategy_cls_name: str,
) -> dict[str, object]:
    """Validate map keys against ``supported_keys`` and return a fresh ``dict``.

    Used by every strategy ctor that accepts ``pretrained_leaves``. The
    return type is ``dict[str, object]`` so the caller can store it on
    the instance without aliasing the user-supplied Mapping (defensive
    copy — user shouldn't be able to mutate strategy state by mutating
    the mapping they passed in).

    Raises ``ValueError`` listing unknown keys + the strategy's supported
    keys. For non-ML strategies (``_leaf_keys == frozenset()``), any
    non-empty map raises — the kwarg exists for API uniformity, not to
    be used.
    """
    if not pretrained_leaves:
        return {}
    unknown = set(pretrained_leaves) - supported_keys
    if unknown:
        raise ValueError(
            f"{strategy_cls_name} does not own pretrained leaf(s) "
            f"{sorted(unknown)!r}; "
            f"{describe_supported_leaf_keys(supported_keys, strategy_cls_name)}."
        )
    return dict(pretrained_leaves)


def _resolve_leaf_lookback(leaf: object) -> int | None:
    """Return the LSTM lookback for a leaf that has one, else ``None``.

    Walks the two attribute paths currently in use:

    * ``LSTMPredictor`` — ``leaf._lookback`` directly.
    * ``HybridReturnModel`` / ``HybridVolatilityModel`` — ``leaf._lstm._lookback``.

    Tolerant ``getattr`` chain rather than an isinstance check so a future
    hybrid composite with the same shape doesn't need to be added here. A
    leaf without any recognisable lookback (ARMA, GARCH) returns ``None``
    — the caller skips the lookback check in that case.
    """
    direct = getattr(leaf, "_lookback", None)
    if isinstance(direct, int):
        return direct
    inner = getattr(leaf, "_lstm", None)
    if inner is not None:
        inner_lookback = getattr(inner, "_lookback", None)
        if isinstance(inner_lookback, int):
            return inner_lookback
    return None


def validate_pretrained_leaf(
    leaf: object,
    *,
    interval: Interval,
    feature_columns: Sequence[str],
    lstm_lookback: int | None = None,
) -> None:
    """Reject an injected leaf that mismatches the strategy's training contract.

    Each check corresponds to a silent-misprediction class:

    * ``training_metadata is None`` — leaf never completed ``fit()``. Using
      it as frozen would produce ``.predict()`` calls against uninitialised
      state. Deep metadata check would also trip downstream, but failing
      at injection is more diagnostic.
    * **Interval mismatch** — leaf was trained on e.g. daily bars but the
      strategy runs hourly. The LSTM window semantics are in bars, not
      time, so the leaf would produce predictions for the wrong horizon.
    * **Feature-columns mismatch** — the leaf's scaler was fit on one
      column ordering; calling ``.predict`` with a different ordering
      would feed shuffled columns into the scaler and produce arbitrary
      garbage. Tuple comparison catches both missing columns and reordered
      columns.
    * **Lookback mismatch** — the strategy's ``required_warmup_bars``
      advertises ``lstm_lookback`` worth of history; a leaf trained on a
      different lookback would receive an under- or over-sized window at
      inference time and produce wrong predictions with no error. Only
      checked when the caller passes ``lstm_lookback`` — for non-LSTM
      leaves (GARCH, ARMA) there's no lookback contract to verify.

    Intentionally NOT checked here:

    * Temporal no-overlap (leaf.train_end < fold.train_start /
      fold.test_start). Those are per-fold invariants — the walk-forward
      orchestrator enforces them via ``_validate_deep_metadata`` using
      the ``TrackedMetadata.is_pretrained`` flag. Checking at injection
      time would reject legitimate cases where the user hasn't yet
      decided which folds to run.
    """
    meta = getattr(leaf, "training_metadata", None)
    if meta is None:
        raise ValueError(
            f"pretrained leaf {type(leaf).__name__} has no training_metadata; "
            f"fix by fitting the leaf via `experiment train-model` before "
            f"injection (the artifact load path populates training_metadata "
            f"only if the saved model completed fit())."
        )

    if meta.interval != interval:
        raise ValueError(
            f"pretrained leaf interval mismatch: "
            f"leaf={meta.interval.value}, strategy={interval.value}; "
            f"fix by retraining the leaf on {interval.value} bars or by "
            f"switching the strategy config's interval to {meta.interval.value}."
        )

    expected = tuple(feature_columns)
    if tuple(meta.feature_columns) != expected:
        raise ValueError(
            f"pretrained leaf feature_columns mismatch: "
            f"leaf={list(meta.feature_columns)}, strategy expects={list(expected)}; "
            f"fix by retraining the leaf with the same feature pipeline (or by "
            f"aligning the strategy config's feature_columns with the leaf)."
        )

    if lstm_lookback is not None:
        leaf_lookback = _resolve_leaf_lookback(leaf)
        if leaf_lookback is not None and leaf_lookback != lstm_lookback:
            raise ValueError(
                f"pretrained leaf lstm_lookback mismatch: "
                f"leaf={leaf_lookback}, strategy={lstm_lookback}; "
                f"fix by aligning the strategy config's lstm_lookback with the "
                f"leaf's trained value (generate_signals() feeds the leaf a "
                f"{lstm_lookback}-bar window but the LSTM was trained on "
                f"{leaf_lookback}-bar windows — silent misalignment otherwise)."
            )
