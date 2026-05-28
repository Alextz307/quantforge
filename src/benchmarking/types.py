"""
Dataclasses for benchmarking runs, summary stats, and comparison reports.

On-disk format is JSONL: one ``BenchmarkRun`` per file under
``benchmark_results/runs/<timestamp>_<sha7>.jsonl`` (or
``benchmark_results/baselines/<name>.jsonl`` for anchored baselines). The
outer container is a single JSON object that round-trips through
``BenchmarkRun.to_dict`` / ``from_dict``; the ``.jsonl`` extension is an
affordance for future expansion (multiple runs per file) but today each
file contains exactly one object.

All dataclasses are ``frozen=True`` so a ``BenchmarkRun`` in hand cannot be
mutated after construction — the store is append-only by convention. Lists
become ``tuple`` to make ``frozen=True`` effective (a frozen dataclass with a
``list`` field still allows in-place mutation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NoReturn, Self

from src.core import json_io

SCALING_O_N = "O(n)"
SCALING_O_N_LOG_N = "O(n log n)"
SCALING_O_N_SQUARED = "O(n^2)"
SCALING_UNCLEAR = "unclear"


@dataclass(frozen=True)
class HardwareInfo:
    """
    Host machine signature for a benchmark run.

    ``git_sha`` + ``git_dirty`` anchor the measurement to a specific code
    state; baselines committed to the repo must document the SHA they were
    captured on so later cross-machine comparisons are interpretable.
    """

    cpu_brand: str
    cpu_count: int
    ram_gb: float
    os_name: str
    os_version: str
    python_version: str
    git_sha: str
    git_dirty: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "cpu_brand": self.cpu_brand,
            "cpu_count": self.cpu_count,
            "ram_gb": self.ram_gb,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "python_version": self.python_version,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        return cls(
            cpu_brand=json_io.get_str(d, "cpu_brand"),
            cpu_count=json_io.get_int(d, "cpu_count"),
            ram_gb=json_io.get_float(d, "ram_gb"),
            os_name=json_io.get_str(d, "os_name"),
            os_version=json_io.get_str(d, "os_version"),
            python_version=json_io.get_str(d, "python_version"),
            git_sha=json_io.get_str(d, "git_sha"),
            git_dirty=json_io.get_bool(d, "git_dirty"),
        )


@dataclass(frozen=True)
class BenchmarkResult:
    """
    One measurement emitted by Google Benchmark (or the hybrid runner).

    ``params`` pulls the size/variant arguments out of the benchmark name
    (e.g., ``BM_RSI/10000`` -> ``{"n": 10000}``) so scaling analysis does not
    have to re-parse strings. ``custom_counters`` carries the cycle / IPC /
    instructions counters added by the cross-platform cycle counter.
    """

    name: str
    family: str
    iterations: int
    real_time_ns: float
    cpu_time_ns: float
    items_per_second: float
    custom_counters: dict[str, float] = field(default_factory=dict)
    params: dict[str, int] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "iterations": self.iterations,
            "real_time_ns": self.real_time_ns,
            "cpu_time_ns": self.cpu_time_ns,
            "items_per_second": self.items_per_second,
            "custom_counters": dict(self.custom_counters),
            "params": dict(self.params),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        counters_raw = d.get("custom_counters", {})
        if not isinstance(counters_raw, dict):
            raise ValueError("custom_counters must be a dict")
        counters: dict[str, float] = {}
        for k, v in counters_raw.items():
            if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"custom_counters[{k!r}] must map str -> number")
            counters[k] = float(v)

        params_raw = d.get("params", {})
        if not isinstance(params_raw, dict):
            raise ValueError("params must be a dict")
        params: dict[str, int] = {}
        for k, v in params_raw.items():
            if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"params[{k!r}] must map str -> int")
            params[k] = v

        return cls(
            name=json_io.get_str(d, "name"),
            family=json_io.get_str(d, "family"),
            iterations=json_io.get_int(d, "iterations"),
            real_time_ns=json_io.get_float(d, "real_time_ns"),
            cpu_time_ns=json_io.get_float(d, "cpu_time_ns"),
            items_per_second=json_io.get_float(d, "items_per_second"),
            custom_counters=counters,
            params=params,
            tags=tuple(json_io.get_str_list(d, "tags")) if "tags" in d else (),
        )


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    timestamp: str
    tags: tuple[str, ...]
    results: tuple[BenchmarkResult, ...]
    hardware: HardwareInfo

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "tags": list(self.tags),
            "results": [r.to_dict() for r in self.results],
            "hardware": self.hardware.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        results_raw = d.get("results", [])
        if not isinstance(results_raw, list):
            raise ValueError("results must be a list")
        results = tuple(
            BenchmarkResult.from_dict(r) if isinstance(r, dict) else _reject("result")
            for r in results_raw
        )
        hw_raw = d.get("hardware")
        if not isinstance(hw_raw, dict):
            raise ValueError("hardware must be a dict")
        return cls(
            run_id=json_io.get_str(d, "run_id"),
            timestamp=json_io.get_str(d, "timestamp"),
            tags=tuple(json_io.get_str_list(d, "tags")),
            results=results,
            hardware=HardwareInfo.from_dict(hw_raw),
        )


@dataclass(frozen=True)
class BenchmarkStats:
    """
    Aggregated statistics when a benchmark is run repeatedly.
    """

    name: str
    mean_ns: float
    std_ns: float
    p5_ns: float
    p95_ns: float
    ci95_low: float
    ci95_high: float
    n_samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mean_ns": self.mean_ns,
            "std_ns": self.std_ns,
            "p5_ns": self.p5_ns,
            "p95_ns": self.p95_ns,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        return cls(
            name=json_io.get_str(d, "name"),
            mean_ns=json_io.get_float(d, "mean_ns"),
            std_ns=json_io.get_float(d, "std_ns"),
            p5_ns=json_io.get_float(d, "p5_ns"),
            p95_ns=json_io.get_float(d, "p95_ns"),
            ci95_low=json_io.get_float(d, "ci95_low"),
            ci95_high=json_io.get_float(d, "ci95_high"),
            n_samples=json_io.get_int(d, "n_samples"),
        )


@dataclass(frozen=True)
class RegressionReport:
    """
    Per-benchmark before/after delta with a significance flag.

    ``is_regression`` uses a two-gate rule (|z| >= threshold_z AND |pct| >=
    threshold_pct) to avoid flagging statistically-significant microscopic
    changes as regressions.
    """

    name: str
    baseline_mean_ns: float
    current_mean_ns: float
    pct_delta: float
    z_score: float
    is_regression: bool
    is_improvement: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "baseline_mean_ns": self.baseline_mean_ns,
            "current_mean_ns": self.current_mean_ns,
            "pct_delta": self.pct_delta,
            "z_score": self.z_score,
            "is_regression": self.is_regression,
            "is_improvement": self.is_improvement,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        return cls(
            name=json_io.get_str(d, "name"),
            baseline_mean_ns=json_io.get_float(d, "baseline_mean_ns"),
            current_mean_ns=json_io.get_float(d, "current_mean_ns"),
            pct_delta=json_io.get_float(d, "pct_delta"),
            z_score=json_io.get_float(d, "z_score"),
            is_regression=json_io.get_bool(d, "is_regression"),
            is_improvement=json_io.get_bool(d, "is_improvement"),
        )


@dataclass(frozen=True)
class ScalingAnalysis:
    """
    Log-log polyfit across sizes for one benchmark family.
    """

    family: str
    sizes: tuple[int, ...]
    times_ns: tuple[float, ...]
    slope: float
    intercept: float
    r_squared: float
    classification: str

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "sizes": list(self.sizes),
            "times_ns": list(self.times_ns),
            "slope": self.slope,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "classification": self.classification,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        return cls(
            family=json_io.get_str(d, "family"),
            sizes=tuple(json_io.get_int_list(d, "sizes")),
            times_ns=tuple(json_io.get_float_list(d, "times_ns")),
            slope=json_io.get_float(d, "slope"),
            intercept=json_io.get_float(d, "intercept"),
            r_squared=json_io.get_float(d, "r_squared"),
            classification=json_io.get_str(d, "classification"),
        )


@dataclass(frozen=True)
class ComparisonReport:
    """
    Full regression sweep across two runs.
    """

    baseline_run_id: str
    current_run_id: str
    reports: tuple[RegressionReport, ...]

    @property
    def regressions(self) -> tuple[RegressionReport, ...]:
        return tuple(r for r in self.reports if r.is_regression)

    @property
    def improvements(self) -> tuple[RegressionReport, ...]:
        return tuple(r for r in self.reports if r.is_improvement)

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "reports": [r.to_dict() for r in self.reports],
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        reports_raw = d.get("reports", [])
        if not isinstance(reports_raw, list):
            raise ValueError("reports must be a list")
        reports = tuple(
            RegressionReport.from_dict(r) if isinstance(r, dict) else _reject("report")
            for r in reports_raw
        )

        return cls(
            baseline_run_id=json_io.get_str(d, "baseline_run_id"),
            current_run_id=json_io.get_str(d, "current_run_id"),
            reports=reports,
        )


def _reject(what: str) -> NoReturn:
    raise ValueError(f"{what} entries must be JSON objects")
