"""Orchestrator that drives Google Benchmark and Python hybrid benches.

Google Benchmark is the source of truth for micro-bench timing; the C++
path shells out to ``quant_bench --benchmark_format=json`` and parses the
emitted JSON into ``BenchmarkResult`` objects. The Python path runs
end-to-end hybrid measurements via ``time.perf_counter_ns``; hybrid
benches are zero-arg callables registered by caller code so the runner
itself stays free of ML/library imports.

Hardware provenance is captured once per run (``HardwareInfo``): CPU
brand via platform-appropriate ``sysctl`` / ``/proc/cpuinfo``, RAM via
``psutil``, git SHA + dirty flag via ``git`` subprocess. The SHA anchors
baselines to a specific code state.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import psutil

from src.benchmarking.types import BenchmarkResult, BenchmarkRun, HardwareInfo
from src.core import json_io

GBENCH_SIZE_PATTERN = re.compile(r"/(\d+)(?:/|$)")
HYBRID_FAMILY = "PyHybrid"
HYBRID_DEFAULT_ITEMS = 1
DEFAULT_QUANT_BENCH_PATH = Path("cpp/build/benchmarks/quant_bench")
DEFAULT_CPP_TIMEOUT_S = 600.0

HybridCallable = Callable[[], int]  # returns items processed (for items/sec)
HybridBench = tuple[str, HybridCallable]


class BenchmarkRunner:
    def __init__(
        self,
        *,
        quant_bench: Path = DEFAULT_QUANT_BENCH_PATH,
        hybrid_benches: tuple[HybridBench, ...] = (),
        repo_root: Path | None = None,
    ) -> None:
        self._quant_bench = Path(quant_bench)
        self._hybrid_benches = hybrid_benches
        self._repo_root = Path(repo_root) if repo_root is not None else Path.cwd()

    def run(
        self,
        *,
        tags: tuple[str, ...] = (),
        cpp_filter: str | None = None,
        min_time_s: float = 0.1,
        cpp_timeout_s: float = DEFAULT_CPP_TIMEOUT_S,
    ) -> BenchmarkRun:
        hw = collect_hardware_info(self._repo_root)
        cpp_results = self._run_cpp(
            filter_regex=cpp_filter, min_time_s=min_time_s, timeout_s=cpp_timeout_s
        )
        py_results = self._run_hybrid()
        timestamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        sha7 = hw.git_sha[:7] if hw.git_sha else "nosha"
        safe_ts = timestamp.replace(":", "-")
        run_id = f"{safe_ts}_{sha7}"
        return BenchmarkRun(
            run_id=run_id,
            timestamp=timestamp,
            tags=tags,
            results=tuple(cpp_results + py_results),
            hardware=hw,
        )

    def _run_cpp(
        self, *, filter_regex: str | None, min_time_s: float, timeout_s: float
    ) -> list[BenchmarkResult]:
        if not self._quant_bench.exists():
            raise FileNotFoundError(
                f"quant_bench binary not found at {self._quant_bench}; "
                f"build with `cd cpp/build && cmake --build . --target quant_bench`"
            )
        cmd: list[str] = [
            str(self._quant_bench),
            "--benchmark_format=json",
            f"--benchmark_min_time={min_time_s}s",
        ]
        if filter_regex is not None:
            cmd.append(f"--benchmark_filter={filter_regex}")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout_s)
        return parse_gbench_json(proc.stdout)

    def _run_hybrid(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for name, fn in self._hybrid_benches:
            start = time.perf_counter_ns()
            items = fn()
            elapsed = time.perf_counter_ns() - start
            elapsed_f = float(elapsed)
            items_per_s = (items / elapsed_f) * 1e9 if elapsed_f > 0 else 0.0
            results.append(
                BenchmarkResult(
                    name=name,
                    family=HYBRID_FAMILY,
                    iterations=1,
                    real_time_ns=elapsed_f,
                    cpu_time_ns=elapsed_f,
                    items_per_second=items_per_s,
                    custom_counters={},
                    params={"n": items if items > 0 else HYBRID_DEFAULT_ITEMS},
                    tags=("hybrid",),
                )
            )
        return results


def parse_gbench_json(stdout: str) -> list[BenchmarkResult]:
    """Parse Google Benchmark ``--benchmark_format=json`` output.

    The top-level object has a ``benchmarks`` array; each entry is one
    measurement. ``run_type == "aggregate"`` rows (min/max/mean when
    ``--benchmark_repetitions`` is used) are skipped — summary aggregation
    is the analyzer's job so we keep the raw iterations here.
    """
    parsed = json.loads(stdout)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object from quant_bench")
    raw_list = parsed.get("benchmarks", [])
    if not isinstance(raw_list, list):
        raise ValueError("quant_bench output missing 'benchmarks' array")
    out: list[BenchmarkResult] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        if raw.get("run_type") == "aggregate":
            continue
        out.append(_gbench_entry_to_result(raw))
    return out


_GBENCH_STANDARD_FIELDS = frozenset(
    {
        "name",
        "family_index",
        "per_family_instance_index",
        "run_name",
        "run_type",
        "repetitions",
        "repetition_index",
        "threads",
        "iterations",
        "real_time",
        "cpu_time",
        "time_unit",
        "items_per_second",
        "bytes_per_second",
        "label",
        "error_message",
        "error_occurred",
    }
)


def _gbench_entry_to_result(raw: dict[str, object]) -> BenchmarkResult:
    name = json_io.get_str(raw, "name")
    family = name.split("/", 1)[0]
    size_match = GBENCH_SIZE_PATTERN.search(name)
    params: dict[str, int] = {"n": int(size_match.group(1))} if size_match else {}
    # Google Benchmark emits user counters as top-level numeric keys alongside
    # the standard fields. We filter by exclusion to avoid knowing the full
    # standard-field set, which drifts across versions.
    counters: dict[str, float] = {}
    for k, v in raw.items():
        if k in _GBENCH_STANDARD_FIELDS:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        counters[k] = float(v)

    time_unit = json_io.get_str(raw, "time_unit") if "time_unit" in raw else "ns"
    real_time = _to_ns(json_io.get_float(raw, "real_time"), time_unit)
    cpu_time = _to_ns(json_io.get_float(raw, "cpu_time"), time_unit)
    items_per_second = (
        json_io.get_float(raw, "items_per_second") if "items_per_second" in raw else 0.0
    )

    return BenchmarkResult(
        name=name,
        family=family,
        iterations=json_io.get_int(raw, "iterations"),
        real_time_ns=real_time,
        cpu_time_ns=cpu_time,
        items_per_second=items_per_second,
        custom_counters=counters,
        params=params,
        tags=(),
    )


_UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


def _to_ns(value: float, unit: str) -> float:
    multiplier = _UNIT_TO_NS.get(unit)
    if multiplier is None:
        raise ValueError(f"unknown time unit {unit!r}; expected one of {sorted(_UNIT_TO_NS)}")
    return value * multiplier


def collect_hardware_info(repo_root: Path) -> HardwareInfo:
    """Collect CPU brand, RAM, OS, Python version, and git state."""
    vm = psutil.virtual_memory()
    sha, dirty = _git_state(repo_root)
    return HardwareInfo(
        cpu_brand=_cpu_brand(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=vm.total / 1e9,
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        git_sha=sha,
        git_dirty=dirty,
    )


def _cpu_brand() -> str:
    system = platform.system()
    if system == "Darwin":
        sysctl = shutil.which("sysctl")
        if sysctl is None:
            return platform.processor() or "unknown"
        proc = subprocess.run(
            [sysctl, "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
        return platform.processor() or "unknown"
    if system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    _, _, value = line.partition(":")
                    return value.strip()
        return platform.processor() or "unknown"
    return platform.processor() or "unknown"


def _git_state(repo_root: Path) -> tuple[str, bool]:
    git = shutil.which("git")
    if git is None:
        return "", False
    sha_proc = subprocess.run(
        [git, "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if sha_proc.returncode != 0:
        return "", False
    sha = sha_proc.stdout.strip()
    # Exclude untracked files so a fresh baseline JSONL written by this run
    # doesn't self-flag the tree as dirty.
    dirty_proc = subprocess.run(
        [git, "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = dirty_proc.returncode == 0 and bool(dirty_proc.stdout.strip())
    return sha, dirty
