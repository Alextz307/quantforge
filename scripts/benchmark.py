"""Command-line driver for the benchmarking suite.

Subcommands:

* ``run``              Execute the benchmark suite, store the run, optionally save as baseline.
* ``compare``          Compare two runs (by baseline name) and print the regression report.
* ``latex``            Emit LaTeX tables for a stored run.
* ``history``          List stored runs filtered by since/tag/commit.
* ``show-baseline``    Pretty-print a stored baseline.

All commands operate on files under ``benchmark_results/`` relative to the
current working directory. Invoke from the repo root (``make bench`` does
this automatically).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from src.benchmarking.analyzer import BenchmarkAnalyzer
from src.benchmarking.comparator import BenchmarkComparator
from src.benchmarking.reporter import BenchmarkReporter
from src.benchmarking.runner import BenchmarkRunner
from src.benchmarking.store import BenchmarkStore
from src.benchmarking.types import BenchmarkRun

DEFAULT_STORE_ROOT = Path("benchmark_results")
REPORTS_SUBDIR = "reports"


@click.group()
def cli() -> None:
    """Quant-engine benchmark orchestrator."""


@cli.command("run")
@click.option("--tag", "tags", multiple=True, help="Attach a tag to this run (repeatable).")
@click.option(
    "--filter",
    "cpp_filter",
    default=None,
    help="Regex passed to quant_bench --benchmark_filter.",
)
@click.option("--min-time", default=0.1, type=float, help="Per-benchmark min time in seconds.")
@click.option(
    "--save-baseline",
    "baseline_name",
    default=None,
    help="Save the run as a named baseline under benchmark_results/baselines/.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Allow --save-baseline to overwrite an existing baseline file.",
)
@click.option(
    "--report/--no-report",
    default=True,
    help="Generate a full report directory after the run.",
)
@click.option(
    "--baseline",
    "compare_baseline",
    default=None,
    help="Name of a baseline to compare against in the generated report.",
)
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    help="Override the benchmark_results directory.",
)
def run_cmd(
    tags: tuple[str, ...],
    cpp_filter: str | None,
    min_time: float,
    baseline_name: str | None,
    overwrite: bool,
    report: bool,
    compare_baseline: str | None,
    store_root: str,
) -> None:
    store = BenchmarkStore(Path(store_root))
    runner = BenchmarkRunner()
    click.echo(f"running quant_bench (min_time={min_time}s, filter={cpp_filter!r}) ...")
    try:
        run = runner.run(tags=tags, cpp_filter=cpp_filter, min_time_s=min_time)
    except subprocess.CalledProcessError as e:
        raise click.ClickException(
            f"quant_bench failed (exit={e.returncode}): {e.stderr.strip() or e.stdout.strip()}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise click.ClickException(
            f"quant_bench exceeded timeout ({e.timeout}s); consider a tighter --filter"
        ) from e
    path = store.save_run(run)
    click.echo(f"saved run to {path}")

    if baseline_name is not None:
        baseline_path = store.save_baseline(run, baseline_name, overwrite=overwrite)
        click.echo(f"saved baseline {baseline_name!r} to {baseline_path}")

    if report:
        reporter = BenchmarkReporter()
        out_dir = Path(store_root) / REPORTS_SUBDIR / run.run_id
        baseline_run: BenchmarkRun | None = None
        if compare_baseline is not None:
            baseline_run = store.load_baseline(compare_baseline)
        reporter.generate_full_report(run, out_dir, baseline=baseline_run)
        click.echo(f"wrote report to {out_dir}")


@cli.command("compare")
@click.argument("baseline_name")
@click.argument("current_name")
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    help="Override the benchmark_results directory.",
)
def compare_cmd(baseline_name: str, current_name: str, store_root: str) -> None:
    store = BenchmarkStore(Path(store_root))
    baseline = store.load_baseline(baseline_name)
    current = store.load_baseline(current_name)
    cmp = BenchmarkComparator().compare(baseline, current)
    click.echo(f"{cmp.current_run_id} vs {cmp.baseline_run_id}:")
    for r in cmp.reports:
        flag = "REG" if r.is_regression else "IMP" if r.is_improvement else "  "
        click.echo(
            f"  [{flag}] {r.name:40s} "
            f"{r.baseline_mean_ns:12.1f} -> {r.current_mean_ns:12.1f} "
            f"({r.pct_delta:+6.2f}%, z={r.z_score:+.2f})"
        )
    click.echo(
        f"summary: {len(cmp.regressions)} regression(s), {len(cmp.improvements)} improvement(s)"
    )


@cli.command("latex")
@click.argument("run_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out-dir",
    default=None,
    help="Directory for the report. Defaults to <store-root>/reports/<run_id>.",
)
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    help="Override the benchmark_results directory.",
)
def latex_cmd(run_path: Path, out_dir: str | None, store_root: str) -> None:
    root = Path(store_root)
    run = BenchmarkStore(root).load_run(run_path)
    target = Path(out_dir) if out_dir is not None else root / REPORTS_SUBDIR / run.run_id
    BenchmarkReporter().generate_full_report(run, target)
    click.echo(f"wrote LaTeX tables + plots to {target}")


@cli.command("history")
@click.option("--since", default=None, help="ISO-timestamp lower bound.")
@click.option("--tag", "tags", multiple=True, help="Require these tags (repeatable).")
@click.option("--commit", default=None, help="Match runs whose git_sha starts with this prefix.")
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    help="Override the benchmark_results directory.",
)
def history_cmd(
    since: str | None, tags: tuple[str, ...], commit: str | None, store_root: str
) -> None:
    store = BenchmarkStore(Path(store_root))
    runs = store.load_runs(since=since, tags=tags if tags else None, commit=commit)
    if not runs:
        click.echo("no runs match the given filters")
        return
    for run in runs:
        click.echo(
            f"{run.run_id:40s} {run.timestamp:30s} "
            f"tags={list(run.tags)} sha={run.hardware.git_sha[:7]} "
            f"dirty={run.hardware.git_dirty}"
        )


@cli.command("show-baseline")
@click.argument("name")
@click.option(
    "--store-root",
    default=str(DEFAULT_STORE_ROOT),
    help="Override the benchmark_results directory.",
)
def show_baseline_cmd(name: str, store_root: str) -> None:
    store = BenchmarkStore(Path(store_root))
    run = store.load_baseline(name)
    analyzer = BenchmarkAnalyzer()
    stats = analyzer.summarize(list(run.results))
    click.echo(
        f"baseline {name!r}: run_id={run.run_id} ts={run.timestamp} "
        f"sha={run.hardware.git_sha[:7]} dirty={run.hardware.git_dirty}"
    )
    click.echo(f"  hardware: {run.hardware.cpu_brand} ({run.hardware.cpu_count} cores)")
    for s in stats:
        click.echo(f"  {s.name:40s} mean={s.mean_ns:12.1f} ns  n={s.n_samples}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
