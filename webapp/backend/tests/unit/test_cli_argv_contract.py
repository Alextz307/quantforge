"""
Drift guard: the webapp's CLI argv builders must agree with the live Click CLI.

``process_manager.build_*_command`` hand-construct the argv the webapp spawns
(``python -m scripts.experiment <verb> --flag ...``). Those flag strings are a
contract with the Click options declared in ``scripts/experiment.py`` /
``scripts/study.py``, and the two have no shared source - a renamed or removed
flag would otherwise surface only as a failed subprocess at runtime. This test
reads the flags off the live Click command objects, so it cannot drift from the
real CLI: every flag a builder emits must be a declared option, and every
required option must be emitted. A separate completeness check asserts every
builder in the module is exercised, so a newly-added builder cannot ship
unguarded.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import pytest

from scripts.experiment import cli
from src.orchestration.comparison import SignificanceTest
from webapp.backend.app.infrastructure import process_manager
from webapp.backend.app.infrastructure.process_manager import (
    build_compare_command,
    build_holdout_command,
    build_importance_command,
    build_run_command,
    build_study_command,
    build_tune_command,
)

_CLI_PREFIX = (sys.executable, "-m", "scripts.experiment")
_DUMMY_PATH = Path("/tmp/quantforge-argv-contract")
_DUMMY_JOB = "job-id"


@dataclass(frozen=True)
class _Case:
    command_path: tuple[str, ...]
    argv: tuple[str, ...]
    builder: str


# id -> _Case(command path under the cli group, argv the webapp would spawn,
# the name of the builder that produced it). Each builder is invoked with
# arguments that emit every flag *that builder* can emit (optional flags on,
# both holdout source branches) so the subset check exercises the conditional
# flags too. The `builder` field feeds the completeness check below.
_CASES: dict[str, _Case] = {
    "run": _Case(
        ("run",),
        build_run_command(
            config_path=_DUMMY_PATH,
            job_id=_DUMMY_JOB,
            store_root=_DUMMY_PATH,
            feature_importance=True,
        ),
        build_run_command.__name__,
    ),
    "importance": _Case(
        ("importance",),
        build_importance_command(run_dir=_DUMMY_PATH, store_root=_DUMMY_PATH, job_id=_DUMMY_JOB),
        build_importance_command.__name__,
    ),
    "tune": _Case(
        ("tune",),
        build_tune_command(
            experiment_config_path=_DUMMY_PATH,
            hpo_config_path=_DUMMY_PATH,
            store_root=_DUMMY_PATH,
        ),
        build_tune_command.__name__,
    ),
    "compare": _Case(
        ("compare",),
        build_compare_command(
            config_paths=(_DUMMY_PATH,),
            reuse_run_dirs=(_DUMMY_PATH,),
            out_name="comparison",
            significance_test=SignificanceTest.BOOTSTRAP,
            n_jobs=1,
            write_report=True,
            publish_label="label",
            store_root=_DUMMY_PATH,
        ),
        build_compare_command.__name__,
    ),
    "holdout-eval-run": _Case(
        ("holdout-eval",),
        build_holdout_command(
            source_kind="run",
            source_path=_DUMMY_PATH,
            out_name="holdout",
            write_report=True,
            publish_label="label",
            store_root=_DUMMY_PATH,
        ),
        build_holdout_command.__name__,
    ),
    "holdout-eval-hpo": _Case(
        ("holdout-eval",),
        build_holdout_command(
            source_kind="hpo",
            source_path=_DUMMY_PATH,
            out_name=None,
            write_report=False,
            publish_label=None,
            store_root=_DUMMY_PATH,
        ),
        build_holdout_command.__name__,
    ),
    "study-run": _Case(
        ("study", "run"),
        build_study_command(
            spec_path=_DUMMY_PATH,
            force_rerun=True,
            only_legs=("strategy-x-universe",),
            skip_compares=True,
            skip_holdout_eval=True,
            store_root=_DUMMY_PATH,
        ),
        build_study_command.__name__,
    ),
}


def _module_builder_names() -> set[str]:
    return {
        name
        for name, obj in inspect.getmembers(process_manager, inspect.isfunction)
        if name.startswith("build_")
        and name.endswith("_command")
        and obj.__module__ == process_manager.__name__
    }


def _resolve_command(command_path: tuple[str, ...]) -> click.Command:
    node: click.Command = cli
    for name in command_path:
        assert isinstance(node, click.Group), f"{name!r}'s parent is not a Click group"
        child = node.commands.get(name)
        assert child is not None, f"the CLI has no command for {' '.join(command_path)!r}"
        node = child
    return node


def _declared_flags(command: click.Command) -> set[str]:
    return {
        opt
        for param in command.params
        for opt in (*param.opts, *param.secondary_opts)
        if opt.startswith("-")
    }


def _required_option_flags(command: click.Command) -> set[str]:
    required: set[str] = set()
    for param in command.params:
        if isinstance(param, click.Option) and param.required:
            long_opts = [opt for opt in param.opts if opt.startswith("--")]
            required.add(long_opts[0] if long_opts else param.opts[0])
    return required


def _flag_region(argv: tuple[str, ...], command_path: tuple[str, ...]) -> tuple[str, ...]:
    assert argv[: len(_CLI_PREFIX)] == _CLI_PREFIX
    head = len(_CLI_PREFIX)
    assert argv[head : head + len(command_path)] == command_path
    return argv[head + len(command_path) :]


def _emitted_flags(flag_region: tuple[str, ...]) -> set[str]:
    # CLI options are all long-form, so a token is a flag only when it leads
    # with '--'. Filtering on the double dash keeps a dash-leading option
    # *value* (e.g. a negative int) from being mistaken for an emitted flag.
    return {token for token in flag_region if token.startswith("--")}


def test_every_builder_command_has_a_case() -> None:
    covered = {case.builder for case in _CASES.values()}
    missing = _module_builder_names() - covered
    assert not missing, (
        f"process_manager defines argv builders with no contract case: {sorted(missing)} - "
        "add a _CASES entry so the new builder's flags are checked against the live CLI"
    )


@pytest.mark.parametrize("case", list(_CASES.values()), ids=list(_CASES.keys()))
def test_builder_emits_only_declared_flags(case: _Case) -> None:
    command = _resolve_command(case.command_path)
    emitted = _emitted_flags(_flag_region(case.argv, case.command_path))
    unknown = emitted - _declared_flags(command)
    assert not unknown, f"{' '.join(case.command_path)} emits undeclared flags: {sorted(unknown)}"


@pytest.mark.parametrize("case", list(_CASES.values()), ids=list(_CASES.keys()))
def test_builder_supplies_every_required_option(case: _Case) -> None:
    command = _resolve_command(case.command_path)
    emitted = _emitted_flags(_flag_region(case.argv, case.command_path))
    missing = _required_option_flags(command) - emitted
    assert not missing, f"{' '.join(case.command_path)} omits required options: {sorted(missing)}"
