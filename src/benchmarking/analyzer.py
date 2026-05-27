"""Statistical summaries, regression detection, and scaling-law fits.

All analysis runs on ``BenchmarkRun`` objects loaded from the store — no
subprocess access, no filesystem writes. This module is cpu-bound pandas /
numpy math and is pure in the functional sense: given the same inputs it
produces the same outputs.

Regression detection uses a paired two-gate rule: both the effect size
(|pct_delta|) and the significance (|z|) must exceed their thresholds.
This avoids flagging a 0.2% slowdown as a regression just because the
variance is microscopic, and avoids flagging a single noisy 30% outlier
as a regression when it's within baseline noise.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import numpy.typing as npt

from src.benchmarking.types import (
    SCALING_O_N,
    SCALING_O_N_LOG_N,
    SCALING_O_N_SQUARED,
    SCALING_UNCLEAR,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkStats,
    RegressionReport,
    ScalingAnalysis,
)

DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_PCT_THRESHOLD = 5.0
SCALING_SLOPE_LINEAR = 1.0
SCALING_SLOPE_LINEARITHMIC = 1.15
SCALING_SLOPE_QUADRATIC = 2.0
SCALING_SLOPE_TOLERANCE = 0.15
SCALING_MIN_R_SQUARED = 0.9


class BenchmarkAnalyzer:
    def summarize(self, results: list[BenchmarkResult]) -> list[BenchmarkStats]:
        """Group results by name and compute per-group summary stats.

        Within a single run, Google Benchmark typically emits one row per
        benchmark; repeated rows appear when the user passes
        ``--benchmark_repetitions``. The summarizer handles both.
        """

        grouped: dict[str, list[float]] = defaultdict(list)
        for r in results:
            grouped[r.name].append(r.real_time_ns)
        return [_summarize_group(name, times) for name, times in sorted(grouped.items())]

    def detect_regressions(
        self,
        current: BenchmarkRun,
        baseline: BenchmarkRun,
        *,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        pct_threshold: float = DEFAULT_PCT_THRESHOLD,
    ) -> list[RegressionReport]:
        """Per-benchmark paired comparison between two runs.

        Benchmarks present in ``current`` but missing from ``baseline`` (or
        vice versa) are silently skipped — the caller decides whether the
        universe mismatch matters.
        """

        current_by_name = _results_by_name(current.results)
        baseline_by_name = _results_by_name(baseline.results)
        shared = sorted(set(current_by_name) & set(baseline_by_name))
        return [
            _compare_pair(
                name,
                current_by_name[name],
                baseline_by_name[name],
                z_threshold,
                pct_threshold,
            )
            for name in shared
        ]

    def analyze_scaling(self, run: BenchmarkRun, family: str) -> ScalingAnalysis:
        """Fit ``log(time) ~ slope * log(n) + intercept`` over one family.

        ``family`` is the prefix before the ``/`` in a Google Benchmark name
        (e.g., ``BM_RSI`` for ``BM_RSI/10000``). The size is read from
        ``result.params['n']`` which the runner parses from the name.
        """

        members = [r for r in run.results if r.family == family]
        if len(members) < 2:
            raise ValueError(
                f"scaling analysis needs >= 2 points for family {family!r}, "
                f"got {len(members)}; fix by running the benchmark across "
                f"at least 2 input sizes (Google Benchmark range / Args)."
            )
        sized = [(r.params.get("n", 0), r.real_time_ns) for r in members]
        sized.sort()
        sizes = [s for s, _ in sized]
        times = [t for _, t in sized]
        if any(s <= 0 for s in sizes):
            raise ValueError(
                f"all sizes must be > 0 for family {family!r}, got {sizes}; "
                f"fix by ensuring each benchmark name encodes a positive 'n' "
                f"(e.g. BM_RSI/10000)."
            )
        slope, intercept, r_squared = _log_log_fit(
            np.asarray(sizes, dtype=np.int64), np.asarray(times, dtype=np.float64)
        )
        return ScalingAnalysis(
            family=family,
            sizes=tuple(sizes),
            times_ns=tuple(times),
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            classification=_classify_slope(slope, r_squared),
        )


def _summarize_group(name: str, times_ns: list[float]) -> BenchmarkStats:
    arr = np.asarray(times_ns, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    p5 = float(np.percentile(arr, 5.0))
    p95 = float(np.percentile(arr, 95.0))
    half = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
    return BenchmarkStats(
        name=name,
        mean_ns=mean,
        std_ns=std,
        p5_ns=p5,
        p95_ns=p95,
        ci95_low=mean - half,
        ci95_high=mean + half,
        n_samples=n,
    )


def _compare_pair(
    name: str,
    current: list[BenchmarkResult],
    baseline: list[BenchmarkResult],
    z_threshold: float,
    pct_threshold: float,
) -> RegressionReport:
    cur_mean, cur_var, cur_n = _mean_var_n([r.real_time_ns for r in current])
    base_mean, base_var, base_n = _mean_var_n([r.real_time_ns for r in baseline])
    pct = 100.0 * (cur_mean - base_mean) / base_mean
    pooled = math.sqrt((cur_var / cur_n) + (base_var / base_n))

    if pooled > 0:
        z = (cur_mean - base_mean) / pooled
    elif pct == 0.0:
        z = 0.0
    else:
        z = math.copysign(math.inf, pct)

    is_regression = pct >= pct_threshold and abs(z) >= z_threshold
    is_improvement = pct <= -pct_threshold and abs(z) >= z_threshold
    return RegressionReport(
        name=name,
        baseline_mean_ns=base_mean,
        current_mean_ns=cur_mean,
        pct_delta=pct,
        z_score=z,
        is_regression=is_regression,
        is_improvement=is_improvement,
    )


def _mean_var_n(samples: list[float]) -> tuple[float, float, int]:
    """Return (mean, variance, n). Fast path avoids a numpy array allocation
    for the common one-run-per-bench case where n == 1."""

    n = len(samples)
    if n == 0:
        raise ValueError("cannot compare empty sample list")
    if n == 1:
        return samples[0], 0.0, 1
    arr = np.asarray(samples, dtype=np.float64)
    return float(arr.mean()), float(arr.var(ddof=1)), n


def _log_log_fit(
    sizes: npt.NDArray[np.int64 | np.float64], times: npt.NDArray[np.float64]
) -> tuple[float, float, float]:
    """Returns (slope, intercept, r_squared) of ``log(times) ~ slope*log(sizes) + intercept``."""

    xs = np.log(sizes.astype(np.float64))
    ys = np.log(times.astype(np.float64))
    slope, intercept = np.polyfit(xs, ys, 1)
    predicted = slope * xs + intercept
    ss_res = float(np.sum((ys - predicted) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), r_squared


def _classify_slope(slope: float, r_squared: float) -> str:
    if r_squared < SCALING_MIN_R_SQUARED:
        return SCALING_UNCLEAR
    if abs(slope - SCALING_SLOPE_LINEAR) <= SCALING_SLOPE_TOLERANCE:
        return SCALING_O_N
    if abs(slope - SCALING_SLOPE_LINEARITHMIC) <= SCALING_SLOPE_TOLERANCE:
        return SCALING_O_N_LOG_N
    if abs(slope - SCALING_SLOPE_QUADRATIC) <= SCALING_SLOPE_TOLERANCE:
        return SCALING_O_N_SQUARED
    return SCALING_UNCLEAR


def _results_by_name(results: tuple[BenchmarkResult, ...]) -> dict[str, list[BenchmarkResult]]:
    out: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        out[r.name].append(r)
    return out
