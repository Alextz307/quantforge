"""Map fold test windows to regime labels for per-regime aggregation.

Each :class:`FoldRecord` is assigned to one regime — the one that covers
at least ``majority_threshold`` of bars in its test window. Folds without
a dominant regime are bucketed under :data:`MIXED_REGIME_LABEL` rather
than dropped, so the regime report can surface them as a separate row
("strategy behaved on regime-straddling folds: ...").

Why majority-by-bar-count, not first-bar
----------------------------------------
A fold's test window can easily span a regime boundary — e.g., a
6-month walk-forward window in 2020 starts in low-vol pre-COVID and
ends in high-vol crash. Picking by the first bar would silently mis-tag
the fold as low-vol; picking by majority over all bars in the window
respects the regime that dominated. The 60% threshold was chosen as
the smallest majority where the fold-mean Sharpe is still defensibly
attributable to that regime; tighter (>70%) thresholds shrink the
sample size of any single regime to the point of triviality on a
3-fold walk-forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from src.orchestration.types import MIXED_REGIME_LABEL, UNCLASSIFIED_LABEL, FoldRecord

if TYPE_CHECKING:
    from src.orchestration.regime import IRegimeDetector

_DEFAULT_MAJORITY_THRESHOLD = 0.6


@dataclass(frozen=True)
class SplitResult:
    """Outcome of :func:`split_folds_by_regime`.

    ``per_regime`` maps each regime label to the fold records whose test
    windows landed primarily in that regime. ``mixed`` collects the folds
    whose dominant-regime share fell below ``majority_threshold`` —
    surfaced separately so a reader can see how often the strategy lived
    on regime boundaries (a separate signal from how it behaved within a
    given regime).

    The label set in ``per_regime`` is exactly the set of labels the
    detector produced over the relevant fold windows; regimes the
    detector emitted but no fold landed in DO NOT appear here. Empty-
    regime rows are fabricated by the reporter at render time so the
    heatmap always covers every label in ``detector.tag(bars).unique()``.
    """

    per_regime: dict[str, tuple[FoldRecord, ...]]
    mixed: tuple[FoldRecord, ...]
    majority_threshold: float


def split_folds_by_regime(
    folds: tuple[FoldRecord, ...],
    detector: IRegimeDetector,
    bars: pd.DataFrame,
    *,
    majority_threshold: float = _DEFAULT_MAJORITY_THRESHOLD,
) -> SplitResult:
    """Assign each fold to a regime by majority-bar-count over its test window.

    ``bars`` must cover every fold's test window — partial coverage is a
    contract violation (the detector can't tag bars it doesn't see) and
    raises :class:`ValueError` so a misaligned ``bars`` argument can't
    silently skew the assignment.

    The unclassified-label produced by warmup-needing detectors (e.g.
    trend's first ``window`` bars) is excluded from the bar count so a
    fold that opens in the warmup region but spends most of its test
    window in a real regime still gets the right label.
    """
    if not (0.5 < majority_threshold <= 1.0):
        raise ValueError(
            f"majority_threshold must be in (0.5, 1.0], got {majority_threshold}; "
            f"fix by passing a fraction strictly above 0.5 (an exact tie below "
            f"0.5 wouldn't define a 'dominant' regime)."
        )

    tagged = detector.tag(bars)
    per_regime: dict[str, list[FoldRecord]] = {}
    mixed: list[FoldRecord] = []

    for fold in folds:
        window_tags = tagged.loc[(tagged.index >= fold.test_start) & (tagged.index < fold.test_end)]
        if len(window_tags) == 0:
            raise ValueError(
                f"fold {fold.fold_index} test window [{fold.test_start}, "
                f"{fold.test_end}) contains zero bars in the supplied 'bars' frame; "
                f"fix by passing the bars DataFrame the experiment was run on (or a "
                f"superset of it)."
            )
        # Drop unclassified bars BEFORE majority math so trend / vol warmup
        # at the head of the first fold doesn't dilute every regime share.
        classified = window_tags[window_tags != UNCLASSIFIED_LABEL]
        if len(classified) == 0:
            mixed.append(fold)
            continue

        counts = classified.value_counts()
        dominant_label = str(counts.index[0])
        dominant_share = float(counts.iloc[0]) / float(len(classified))
        if dominant_share >= majority_threshold:
            per_regime.setdefault(dominant_label, []).append(fold)
        else:
            mixed.append(fold)

    return SplitResult(
        per_regime={k: tuple(v) for k, v in per_regime.items()},
        mixed=tuple(mixed),
        majority_threshold=majority_threshold,
    )


__all__ = ["MIXED_REGIME_LABEL", "SplitResult", "split_folds_by_regime"]
