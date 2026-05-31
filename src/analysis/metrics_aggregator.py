"""
Fold-level -> run-level metric aggregation.

:class:`AggregateStats` collapses a tuple of :class:`FoldRecord` objects into
a frozen dataclass holding per-metric mean / std / 95% CI plus a couple of
run-wide scalars (worst drawdown, total trades). :meth:`AggregateStats.to_dict`
returns a flat ``dict[str, object]`` - the shape the objective adapters and
``metrics.json`` readers rely on.

CI semantics
------------
Per-metric 95% CIs come from an IID percentile bootstrap over the FOLD
samples (not intra-fold returns): folds are disjoint walk-forward windows,
so IID resampling of fold means is the right null. Autocorrelation-aware
bootstrap on raw return series lives in :mod:`src.analysis.significance`.

RNG is seeded from a fixed internal seed so ``metrics.json`` is a
deterministic function of fold values - two ``experiment run``
invocations on the same config produce bit-identical CI bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from src.analysis.significance import compute_pooled_sharpe, percentile_ci
from src.core import json_io

if TYPE_CHECKING:
    from src.orchestration.types import FoldRecord

_FloatArray = npt.NDArray[np.float64]

_BOOTSTRAP_N_RESAMPLES = 2000

_BOOTSTRAP_CONFIDENCE = 0.95

_DEFAULT_RNG_SEED = 42


@dataclass(frozen=True)
class AggregateStats:
    """
    Summary of a walk-forward run's fold-level metrics.

    All ``*_mean`` / ``*_std`` / ``*_ci95_*`` and the pooled-Sharpe fields
    (``sharpe_pooled``, ``psr_pooled``, ``pooled_skew``,
    ``pooled_kurtosis``) are ``float('nan')`` when ``n_folds == 0`` (and
    ``n_oos_bars`` is ``0``) - the zero-fold path is a degenerate case
    (empty dev slice after holdout reservation) that callers should
    surface as an error, not aggregate over.
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
    sharpe_pooled: float
    psr_pooled: float
    n_oos_bars: int
    pooled_skew: float
    pooled_kurtosis: float

    def to_dict(self) -> dict[str, object]:
        """
        Flatten to the dict shape consumed by objectives + ``metrics.json``.

        Empty-fold path short-circuits to ``{"n_folds": 0}`` - matches the
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
            "sharpe_pooled": self.sharpe_pooled,
            "psr_pooled": self.psr_pooled,
            "n_oos_bars": self.n_oos_bars,
            "pooled_skew": self.pooled_skew,
            "pooled_kurtosis": self.pooled_kurtosis,
        }

    @classmethod
    def empty(cls) -> AggregateStats:
        """
        Zero-fold sentinel: every numeric stat ``NaN``, trade-count 0.

        Discriminate via ``n_folds == 0`` before reading other fields -
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
            sharpe_pooled=nan,
            psr_pooled=nan,
            n_oos_bars=0,
            pooled_skew=nan,
            pooled_kurtosis=nan,
        )

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> AggregateStats:
        """
        Reconstruct from the :meth:`to_dict` view (e.g. a run's metrics.json).

        Mirrors the zero-fold short-circuit: a dict with ``n_folds == 0``
        rebuilds the :meth:`empty` sentinel without requiring the other keys.
        """

        if json_io.get_int(d, "n_folds") == 0:
            return cls.empty()
        return cls(
            n_folds=json_io.get_int(d, "n_folds"),
            sharpe_mean=json_io.get_float(d, "sharpe_mean"),
            sharpe_std=json_io.get_float(d, "sharpe_std"),
            sharpe_ci95_low=json_io.get_float(d, "sharpe_ci95_low"),
            sharpe_ci95_high=json_io.get_float(d, "sharpe_ci95_high"),
            sortino_mean=json_io.get_float(d, "sortino_mean"),
            sortino_std=json_io.get_float(d, "sortino_std"),
            sortino_ci95_low=json_io.get_float(d, "sortino_ci95_low"),
            sortino_ci95_high=json_io.get_float(d, "sortino_ci95_high"),
            calmar_mean=json_io.get_float(d, "calmar_mean"),
            calmar_std=json_io.get_float(d, "calmar_std"),
            calmar_ci95_low=json_io.get_float(d, "calmar_ci95_low"),
            calmar_ci95_high=json_io.get_float(d, "calmar_ci95_high"),
            max_drawdown_worst=json_io.get_float(d, "max_drawdown_worst"),
            max_drawdown_mean=json_io.get_float(d, "max_drawdown_mean"),
            total_return_mean=json_io.get_float(d, "total_return_mean"),
            total_return_std=json_io.get_float(d, "total_return_std"),
            win_rate_mean=json_io.get_float(d, "win_rate_mean"),
            trade_count_total=json_io.get_int(d, "trade_count_total"),
            sharpe_pooled=json_io.get_float(d, "sharpe_pooled"),
            psr_pooled=json_io.get_float(d, "psr_pooled"),
            n_oos_bars=json_io.get_int(d, "n_oos_bars"),
            pooled_skew=json_io.get_float(d, "pooled_skew"),
            pooled_kurtosis=json_io.get_float(d, "pooled_kurtosis"),
        )


def aggregate_folds(
    folds: tuple[FoldRecord, ...],
    *,
    annualization_factor: int,
    risk_free_rate: float = 0.0,
    rng: np.random.Generator | None = None,
) -> AggregateStats:
    """
    Collapse ``folds`` into per-metric mean / std / 95% CI plus pooled Sharpe.

    Two views of Sharpe come out of this. The ``sharpe_mean`` / ``sharpe_std``
    pair is the equal-weighted mean and dispersion across folds - a
    *stability* read (does the edge hold across regimes?). ``sharpe_pooled``
    is the Sharpe of the stitched out-of-sample return stream (every fold's
    within-fold returns concatenated, seams dropped) - the *realised*
    end-to-end track record, observation weighted. ``annualization_factor``
    (bars per year for the data's interval) and ``risk_free_rate`` (the same
    rate the per-fold Sharpes subtract) put the pooled Sharpe on the same
    annualised scale as the per-fold Sharpes.

    ``rng`` seeds the IID bootstrap resampler. Defaults to a fixed internal
    seed so ``metrics.json`` is a deterministic function of fold values -
    pass a user-seeded generator only if you need to vary CI bounds
    deliberately (e.g., ensembles of experiment reports).

    Single-fold case: std is ``0.0`` and the CI collapses to the point
    estimate - matches how :func:`numpy.std` treats a one-element array
    with ``ddof=1``. We substitute ``0.0`` for std there rather than
    propagating ``NaN`` so ``AggregateStats`` stays comparable via ``==``
    in tests.
    """

    if not folds:
        return AggregateStats.empty()

    rng = rng if rng is not None else np.random.default_rng(_DEFAULT_RNG_SEED)
    pooled = compute_pooled_sharpe(
        _pooled_oos_returns(folds),
        annualization_factor=annualization_factor,
        risk_free_rate=risk_free_rate,
    )

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
        sharpe_pooled=pooled.sharpe,
        psr_pooled=pooled.psr,
        n_oos_bars=pooled.n_obs,
        pooled_skew=pooled.skew,
        pooled_kurtosis=pooled.kurtosis,
    )


def _pooled_oos_returns(folds: tuple[FoldRecord, ...]) -> _FloatArray:
    """
    Concatenate each fold's within-fold OOS returns, dropping the fold seams.

    Each fold's equity curve is re-based to its own starting capital, so a
    return spanning the boundary between two folds is not a tradeable
    return. We take simple returns inside each fold
    (``e[1:] / e[:-1] - 1``) and concatenate them, which reconstructs the
    stitched OOS return stream the pooled Sharpe is computed on. Folds with
    fewer than two equity points contribute nothing.

    A non-positive prior equity point (a blow-up to zero / negative equity)
    yields ``0.0`` rather than ``inf`` / ``nan``, mirroring the C++
    ``MetricsCalculator::equity_to_returns`` guard so the pooled return
    stream is derived identically to the per-fold one.
    """

    segments: list[_FloatArray] = []
    for fold in folds:
        equity = np.asarray(fold.equity_curve, dtype=np.float64)
        if equity.size < 2:
            continue
        prev = equity[:-1]
        positive = prev > 0.0
        segments.append(np.where(positive, equity[1:] / np.where(positive, prev, 1.0) - 1.0, 0.0))
    if not segments:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(segments)


def _mean_std_ci(
    values: _FloatArray,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """
    Return ``(mean, std_ddof1, ci95_low, ci95_high)`` via IID percentile bootstrap.

    NaN inputs (zero-vol folds produce NaN Sharpe/Sortino) propagate
    through ``np.mean`` / ``np.std`` as ``NaN`` - preserved on purpose so
    the aggregate surfaces the degenerate fold instead of hiding it.

    ``n == 1`` short-circuits: std is ``0.0`` and the CI collapses to the
    point - bootstrapping a one-element sample just re-draws the same
    value every time and is wasted work.
    """

    n = len(values)
    point = float(np.mean(values))
    if n == 1:
        return point, 0.0, point, point
    std = float(np.std(values, ddof=1))
    if not math.isfinite(point):
        return point, std, float("nan"), float("nan")
    idx = rng.integers(0, n, size=(_BOOTSTRAP_N_RESAMPLES, n))
    resample_means = values[idx].mean(axis=1)
    lo, hi = percentile_ci(resample_means, _BOOTSTRAP_CONFIDENCE)
    return point, std, lo, hi
