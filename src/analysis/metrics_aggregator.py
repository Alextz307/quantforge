"""Fold-level → run-level metric aggregation.

:class:`AggregateStats` collapses a tuple of :class:`FoldRecord` objects into
a frozen dataclass holding per-metric mean / std / 95% CI plus a couple of
run-wide scalars (worst drawdown, total trades). :meth:`AggregateStats.to_dict`
returns a flat ``dict[str, object]`` — the shape the objective adapters and
``metrics.json`` readers rely on.

CI semantics
------------
Per-metric 95% CIs come from an IID percentile bootstrap over the FOLD
samples (not intra-fold returns): folds are disjoint walk-forward windows,
so IID resampling of fold means is the right null. Autocorrelation-aware
bootstrap on raw return series lives in :mod:`src.analysis.significance`.

RNG is seeded from a fixed internal seed so ``metrics.json`` is a
deterministic function of fold values — two ``experiment run``
invocations on the same config produce bit-identical CI bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from src.analysis.significance import percentile_ci

if TYPE_CHECKING:
    from src.orchestration.types import FoldRecord

_FloatArray = npt.NDArray[np.float64]

# IID bootstrap draws for per-metric CIs. 2000 is sufficient for 2-decimal
# stability of the 2.5 / 97.5 percentiles on the fold-count range we
# realistically see (3-20 folds).
_BOOTSTRAP_N_RESAMPLES = 2000

# Hardcoded 95% CI matches the ``*_ci95_*`` field names on AggregateStats —
# changing this would mean renaming a public field. Held as a constant so
# the percentile_ci call below has a labeled argument instead of 0.95 magic.
_BOOTSTRAP_CONFIDENCE = 0.95

# Fixed seed so ``metrics.json`` is reproducible across invocations. Users
# who want a different seed can call ``aggregate_folds(folds, rng=...)``
# directly and emit to_dict() themselves.
_DEFAULT_RNG_SEED = 42


@dataclass(frozen=True)
class AggregateStats:
    """Summary of a walk-forward run's fold-level metrics.

    All ``*_mean`` / ``*_std`` / ``*_ci95_*`` fields are ``float('nan')``
    when ``n_folds == 0`` — the zero-fold path is a degenerate case (empty
    dev slice after holdout reservation) that callers should surface as an
    error, not aggregate over.
    """

    n_folds: int
    sharpe_mean: float
    sharpe_std: float
    sharpe_ci95_low: float
    sharpe_ci95_high: float
    sortino_mean: float
    sortino_std: float
    sortino_ci95_low: float
    sortino_ci95_high: float
    calmar_mean: float
    calmar_std: float
    calmar_ci95_low: float
    calmar_ci95_high: float
    max_drawdown_worst: float
    max_drawdown_mean: float
    total_return_mean: float
    total_return_std: float
    win_rate_mean: float
    trade_count_total: int

    def to_dict(self) -> dict[str, object]:
        """Flatten to the dict shape consumed by objectives + ``metrics.json``.

        Empty-fold path short-circuits to ``{"n_folds": 0}`` — matches the
        pre-refactor behavior that the HPO objectives rely on for clear
        error messages ("missing ``sharpe_mean``: most likely zero folds").
        """
        if self.n_folds == 0:
            return {"n_folds": 0}
        return {
            "n_folds": self.n_folds,
            "sharpe_mean": self.sharpe_mean,
            "sharpe_std": self.sharpe_std,
            "sharpe_ci95_low": self.sharpe_ci95_low,
            "sharpe_ci95_high": self.sharpe_ci95_high,
            "sortino_mean": self.sortino_mean,
            "sortino_std": self.sortino_std,
            "sortino_ci95_low": self.sortino_ci95_low,
            "sortino_ci95_high": self.sortino_ci95_high,
            "calmar_mean": self.calmar_mean,
            "calmar_std": self.calmar_std,
            "calmar_ci95_low": self.calmar_ci95_low,
            "calmar_ci95_high": self.calmar_ci95_high,
            "max_drawdown_worst": self.max_drawdown_worst,
            "max_drawdown_mean": self.max_drawdown_mean,
            "total_return_mean": self.total_return_mean,
            "total_return_std": self.total_return_std,
            "win_rate_mean": self.win_rate_mean,
            "trade_count_total": self.trade_count_total,
        }

    @classmethod
    def empty(cls) -> AggregateStats:
        """Zero-fold sentinel: every numeric stat ``NaN``, trade-count 0.

        Discriminate via ``n_folds == 0`` before reading other fields —
        ``to_dict()`` short-circuits the dict view to ``{"n_folds": 0}``,
        but in-process callers that read attributes directly will see NaN
        and should treat that as "no aggregate available."
        """
        nan = float("nan")
        return cls(
            n_folds=0,
            sharpe_mean=nan,
            sharpe_std=nan,
            sharpe_ci95_low=nan,
            sharpe_ci95_high=nan,
            sortino_mean=nan,
            sortino_std=nan,
            sortino_ci95_low=nan,
            sortino_ci95_high=nan,
            calmar_mean=nan,
            calmar_std=nan,
            calmar_ci95_low=nan,
            calmar_ci95_high=nan,
            max_drawdown_worst=nan,
            max_drawdown_mean=nan,
            total_return_mean=nan,
            total_return_std=nan,
            win_rate_mean=nan,
            trade_count_total=0,
        )


def aggregate_folds(
    folds: tuple[FoldRecord, ...],
    *,
    rng: np.random.Generator | None = None,
) -> AggregateStats:
    """Collapse ``folds`` into per-metric mean / std / 95% CI.

    ``rng`` seeds the IID bootstrap resampler. Defaults to a fixed internal
    seed so ``metrics.json`` is a deterministic function of fold values —
    pass a user-seeded generator only if you need to vary CI bounds
    deliberately (e.g., ensembles of experiment reports).

    Single-fold case: std is ``0.0`` and the CI collapses to the point
    estimate — matches how :func:`numpy.std` treats a one-element array
    with ``ddof=1``. We substitute ``0.0`` for std there rather than
    propagating ``NaN`` so ``AggregateStats`` stays comparable via ``==``
    in tests.
    """
    if not folds:
        return AggregateStats.empty()

    rng = rng if rng is not None else np.random.default_rng(_DEFAULT_RNG_SEED)

    sharpe = np.array([f.sharpe_ratio for f in folds], dtype=np.float64)
    sortino = np.array([f.sortino_ratio for f in folds], dtype=np.float64)
    calmar = np.array([f.calmar_ratio for f in folds], dtype=np.float64)
    drawdown = np.array([f.max_drawdown for f in folds], dtype=np.float64)
    total_return = np.array([f.total_return for f in folds], dtype=np.float64)
    win_rate = np.array([f.win_rate for f in folds], dtype=np.float64)
    trade_count = np.array([f.trade_count for f in folds], dtype=np.int64)

    sharpe_mean, sharpe_std, sharpe_lo, sharpe_hi = _mean_std_ci(sharpe, rng)
    sortino_mean, sortino_std, sortino_lo, sortino_hi = _mean_std_ci(sortino, rng)
    calmar_mean, calmar_std, calmar_lo, calmar_hi = _mean_std_ci(calmar, rng)
    total_return_mean = float(np.mean(total_return))
    total_return_std = float(np.std(total_return, ddof=1)) if len(total_return) > 1 else 0.0

    return AggregateStats(
        n_folds=len(folds),
        sharpe_mean=sharpe_mean,
        sharpe_std=sharpe_std,
        sharpe_ci95_low=sharpe_lo,
        sharpe_ci95_high=sharpe_hi,
        sortino_mean=sortino_mean,
        sortino_std=sortino_std,
        sortino_ci95_low=sortino_lo,
        sortino_ci95_high=sortino_hi,
        calmar_mean=calmar_mean,
        calmar_std=calmar_std,
        calmar_ci95_low=calmar_lo,
        calmar_ci95_high=calmar_hi,
        max_drawdown_worst=float(np.min(drawdown)),
        max_drawdown_mean=float(np.mean(drawdown)),
        total_return_mean=total_return_mean,
        total_return_std=total_return_std,
        win_rate_mean=float(np.mean(win_rate)),
        trade_count_total=int(np.sum(trade_count)),
    )


def _mean_std_ci(
    values: _FloatArray,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Return ``(mean, std_ddof1, ci95_low, ci95_high)`` via IID percentile bootstrap.

    NaN inputs (zero-vol folds produce NaN Sharpe/Sortino) propagate
    through ``np.mean`` / ``np.std`` as ``NaN`` — preserved on purpose so
    the aggregate surfaces the degenerate fold instead of hiding it.

    ``n == 1`` short-circuits: std is ``0.0`` and the CI collapses to the
    point — bootstrapping a one-element sample just re-draws the same
    value every time and is wasted work.
    """
    n = len(values)
    point = float(np.mean(values))
    if n == 1:
        return point, 0.0, point, point
    std = float(np.std(values, ddof=1))
    # Any NaN in the sample propagates through bootstrap means — skip the
    # work and report NaN bounds consistently with the point estimate.
    if not math.isfinite(point):
        return point, std, float("nan"), float("nan")
    idx = rng.integers(0, n, size=(_BOOTSTRAP_N_RESAMPLES, n))
    resample_means = values[idx].mean(axis=1)
    lo, hi = percentile_ci(resample_means, _BOOTSTRAP_CONFIDENCE)
    return point, std, lo, hi
