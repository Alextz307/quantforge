"""Empirical-study orchestrator: drive (strategy x universe) sweeps end-to-end.

For each (strategy, universe) leg in a :class:`StudySpec`, compose the
experiment config (deep-merge universe profile onto strategy YAML), run
tune -> run -> holdout-eval, then per-universe cross-strategy compare.
Resumable via :class:`StudyState` (one ``study_state.json`` under
``<study_dir>/``); per-leg failures are isolated and don't abort the sweep.

The orchestrator's per-leg outputs reuse the standard artifact
directories under the study root:

* ``<study_dir>/hpo/<leg_id>/``           tune output (best_config.yaml, trials_artifacts)
* ``<study_dir>/runs/<run_experiment_id>/`` run materialised from best_config.yaml
* ``<study_dir>/holdout_evals/<leg_id>/``   honest OOS (if validation has a holdout)
* ``<study_dir>/comparisons/<universe>/``   cross-strategy compare per universe

The auto-generated ``run_experiment_id`` is opaque (timestamp + sha +
random); we record it on each :class:`LegState` so cross-strategy
compare can resolve the run dir without re-walking ``runs/``.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml

from src.core.config import (
    ExperimentConfig,
    StudySpec,
    load_experiment_config,
    load_study_spec,
    load_universe_profile,
)
from src.core.hpo_config import HPOConfig
from src.core.logging import get_logger
from src.core.persistence import (
    COMPARISONS_SUBDIR,
    HPO_SUBDIR,
)
from src.optimization.checkpointing import BEST_CONFIG_YAML_NAME
from src.orchestration.builder import build_experiment
from src.orchestration.comparison import SignificanceTest, run_comparison
from src.orchestration.experiment import RunOptions
from src.orchestration.holdout_eval import resolve_source, run_holdout_eval
from src.orchestration.run_loader import (
    load_experiment_result,
    resolve_run_dir,
)
from src.orchestration.study_state import (
    LEG_STEP_HOLDOUT_EVAL,
    LEG_STEP_RUN,
    LEG_STEP_TUNE,
    LegState,
    StudyState,
    compute_spec_hash,
    read_study_state,
    write_study_state,
)
from src.visualization.comparison_reporter import ComparisonReporter
from src.visualization.holdout_eval_reporter import HoldoutEvalReporter
from src.visualization.hpo_reporter import HPOReporter

_logger = get_logger(__name__)


STUDY_STATE_FILENAME = "study_state.json"
SPEC_SNAPSHOT_FILENAME = "spec.yaml"


@dataclass(frozen=True)
class StudyLegRun:
    """One (strategy, universe) pair expanded from a :class:`StudySpec`.

    The composed :class:`ExperimentConfig` is built lazily by
    :func:`compose_leg_config` rather than stored here so leg-state
    serialisation stays cheap (paths only).
    """

    leg_id: str
    strategy: str
    universe: str
    strategy_config_path: Path
    hpo_config_path: Path
    universe_profile_path: Path


@dataclass(frozen=True)
class StudyRunResult:
    """End-of-sweep summary returned by :func:`run_study`."""

    study_dir: Path
    state: StudyState
    n_legs_completed: int
    n_legs_failed: int
    n_legs_skipped: int
    n_compares_done: int


def make_leg_id(strategy: str, universe: str) -> str:
    """Canonical leg identifier — also the directory-name suffix on artifacts."""

    return f"{strategy}__{universe}"


def expand_spec_into_legs(spec: StudySpec, *, repo_root: Path) -> list[StudyLegRun]:
    """Cross-product the spec into per-(strategy x universe) legs.

    Universe profile paths are resolved as
    ``repo_root / "config/universes" / f"{name}.yaml"`` so spec YAMLs
    can list bare names. Strategy-config and hpo-config paths from the
    spec are resolved against ``repo_root`` if relative.
    """

    universes_dir = repo_root / "config" / "universes"
    out: list[StudyLegRun] = []
    for leg in spec.legs:
        strategy_path = _resolve_under(repo_root, leg.strategy_config)
        hpo_path = _resolve_under(repo_root, leg.hpo_config)
        for universe in leg.universes:
            profile_path = universes_dir / f"{universe}.yaml"
            out.append(
                StudyLegRun(
                    leg_id=make_leg_id(leg.strategy, universe),
                    strategy=leg.strategy,
                    universe=universe,
                    strategy_config_path=strategy_path,
                    hpo_config_path=hpo_path,
                    universe_profile_path=profile_path,
                )
            )
    return out


def compose_leg_config(leg: StudyLegRun) -> ExperimentConfig:
    """Deep-merge universe profile onto strategy YAML.

    The strategy YAML provides ``strategy``, ``features``, ``slippage``,
    ``risk_free_rate`` and any base ``validation`` defaults. The universe
    profile's ``data`` block wins entirely; its ``validation`` keys
    override per-key (so the universe pins ``holdout_pct`` while leaving
    the strategy YAML's ``n_splits``/``test_size``/``gap`` intact unless
    the universe overrides them).
    """

    base = _read_yaml(leg.strategy_config_path)
    profile = load_universe_profile(leg.universe_profile_path)
    base["name"] = leg.leg_id
    base["data"] = profile.data.model_dump(mode="json")
    raw_validation = base.get("validation") or {}
    if not isinstance(raw_validation, dict):
        raise ValueError(
            f"strategy YAML {leg.strategy_config_path} 'validation' block "
            f"must be a mapping, got {type(raw_validation).__name__}"
        )
    merged_validation: dict[str, object] = dict(raw_validation)
    merged_validation.update(profile.validation.model_dump(exclude_unset=True))
    base["validation"] = merged_validation

    return ExperimentConfig.model_validate(base)


def compose_hpo_config(leg: StudyLegRun) -> HPOConfig:
    """Load the HPO YAML and override ``study_name`` to the leg id.

    Without this rewrite, every universe sharing one HPO YAML (e.g. all
    12 AdaptiveBollinger universes) would write into the same Optuna
    SQLite study and contaminate one another's trials.
    """

    raw = _read_yaml(leg.hpo_config_path)
    raw["study_name"] = leg.leg_id
    return HPOConfig.model_validate(raw)


def run_leg(
    leg: StudyLegRun,
    *,
    study_dir: Path,
    skip_holdout_eval: bool,
    prior_state: LegState,
) -> LegState:
    """Execute one leg's tune -> run -> holdout pipeline.

    Returns the updated :class:`LegState`. Steps already in
    ``prior_state.steps_completed`` are skipped (mid-leg resume). Any
    raised exception is captured, logged, and recorded as an ``error``
    on the returned state — the caller decides whether to stop or
    continue with the next leg.
    """

    from src.optimization.tuner import StrategyTuner

    started_at = prior_state.started_at if prior_state.started_at is not None else datetime.now(UTC)
    state = replace(
        prior_state,
        started_at=started_at,
        completed_at=None,
        is_complete=False,
        error=None,
    )

    try:
        cfg = compose_leg_config(leg)
        hpo_cfg = compose_hpo_config(leg)

        if LEG_STEP_TUNE not in state.steps_completed:
            _logger.info("leg %s: starting tune (study_name=%s)", leg.leg_id, hpo_cfg.study_name)
            tuner = StrategyTuner(
                experiment_cfg=cfg,
                hpo_cfg=hpo_cfg,
                store_root=study_dir,
            )
            study = tuner.run(progress=False)
            HPOReporter().generate_full_report(study, tuner.study_dir)
            state = state.with_step_completed(LEG_STEP_TUNE)
            _logger.info("leg %s: tune done", leg.leg_id)

        run_experiment_id = state.run_experiment_id
        best_config_path = study_dir / HPO_SUBDIR / leg.leg_id / BEST_CONFIG_YAML_NAME
        if LEG_STEP_RUN not in state.steps_completed:
            if not best_config_path.is_file():
                raise FileNotFoundError(
                    f"leg {leg.leg_id}: tune did not produce {best_config_path} — "
                    f"the study likely had zero completed trials. Bump n_trials "
                    f"or fix the failing trial signature."
                )
            best_cfg = load_experiment_config(best_config_path)
            experiment = build_experiment(best_cfg)
            _logger.info("leg %s: starting run (best config)", leg.leg_id)
            result = experiment.run(
                RunOptions(
                    store_root=study_dir,
                    write_report=True,
                    publish_label=f"study:{leg.leg_id}",
                )
            )
            run_experiment_id = result.experiment_id
            state = replace(
                state.with_step_completed(LEG_STEP_RUN),
                run_experiment_id=run_experiment_id,
            )
            _logger.info("leg %s: run done (experiment_id=%s)", leg.leg_id, run_experiment_id)

        if run_experiment_id is None:
            raise RuntimeError(f"leg {leg.leg_id}: run step did not record an experiment_id")

        if LEG_STEP_HOLDOUT_EVAL not in state.steps_completed:
            has_holdout = (
                cfg.validation.holdout_pct > 0.0 or cfg.validation.holdout_start is not None
            )
            if has_holdout and not skip_holdout_eval:
                run_dir = resolve_run_dir(study_dir, run_experiment_id)
                source = resolve_source(run_dir=run_dir, hpo_dir=None)
                _logger.info("leg %s: starting holdout-eval", leg.leg_id)
                result_h, out_dir_h = run_holdout_eval(
                    source=source,
                    out_name=leg.leg_id,
                    store_root=study_dir,
                )
                HoldoutEvalReporter().generate_full_report(
                    result_h, out_dir_h, publish_label=f"study:holdout:{leg.leg_id}"
                )
                _logger.info("leg %s: holdout-eval done", leg.leg_id)
            state = state.with_step_completed(LEG_STEP_HOLDOUT_EVAL)

        return replace(state, completed_at=datetime.now(UTC), is_complete=True, error=None)

    except Exception as exc:  # noqa: BLE001 — sweep continues on per-leg failure
        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        _logger.error("leg %s: failed (%s)", leg.leg_id, tb)
        return replace(state, completed_at=datetime.now(UTC), is_complete=False, error=tb)


def resolve_study_dir(spec: StudySpec, store_root: Path) -> Path:
    """Compose the study's output directory.

    ``spec.output_dir`` is relative-to-store-root (e.g. ``studies/main``)
    unless it's already absolute. Centralising the rule here keeps the
    orchestrator and tests in lockstep.
    """

    return spec.output_dir if spec.output_dir.is_absolute() else store_root / spec.output_dir


def run_study(
    spec_path: Path,
    *,
    store_root: Path,
    force_rerun: bool = False,
    only_legs: Sequence[str] | None = None,
    skip_compares: bool = False,
    skip_holdout_eval: bool = False,
    repo_root: Path | None = None,
) -> StudyRunResult:
    """Top-level: expand legs, run each, then per-universe cross-strategy compares.

    Resume rule: a leg with ``is_complete=True`` in the loaded
    ``study_state.json`` is skipped unless ``force_rerun=True``. The
    state's ``spec_hash`` must match the current spec — a mismatch
    raises rather than silently running against a mutated spec.
    """

    repo = repo_root if repo_root is not None else Path.cwd()
    spec = load_study_spec(spec_path)
    study_dir = resolve_study_dir(spec, store_root)
    study_dir.mkdir(parents=True, exist_ok=True)

    spec_hash = compute_spec_hash(spec_path)
    state_path = study_dir / STUDY_STATE_FILENAME
    legs = expand_spec_into_legs(spec, repo_root=repo)

    try:
        existing = read_study_state(state_path)
    except FileNotFoundError:
        existing = None

    if existing is not None:
        if existing.spec_hash != spec_hash:
            raise ValueError(
                f"existing study state at {state_path} was written against a "
                f"different spec (hash {existing.spec_hash[:12]}...) than the "
                f"current spec ({spec_hash[:12]}...); refusing to resume. Move "
                f"or delete the existing state to start fresh."
            )
        state = existing
    else:
        state = StudyState(
            spec_name=spec.name,
            spec_hash=spec_hash,
            started_at=datetime.now(UTC),
            legs=tuple(LegState.initial(leg.leg_id, leg.strategy, leg.universe) for leg in legs),
            cross_strategy_compares_done=(),
        )
        (study_dir / SPEC_SNAPSHOT_FILENAME).write_text(
            spec_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        write_study_state(state_path, state)

    only = set(only_legs) if only_legs else None

    n_completed = 0
    n_failed = 0
    n_skipped = 0
    for leg in legs:
        if only is not None and leg.leg_id not in only:
            n_skipped += 1
            continue
        prior = state.get_leg(leg.leg_id)
        if prior.is_complete and not force_rerun:
            _logger.info("leg %s: already complete — skipping", leg.leg_id)
            n_skipped += 1
            continue
        if force_rerun:
            prior = LegState.initial(leg.leg_id, leg.strategy, leg.universe)
        updated = run_leg(
            leg,
            study_dir=study_dir,
            skip_holdout_eval=skip_holdout_eval,
            prior_state=prior,
        )
        state = state.with_leg(updated)
        write_study_state(state_path, state)
        if updated.is_complete:
            n_completed += 1
        else:
            n_failed += 1

    n_compares = 0
    if not skip_compares:
        state, n_compares = _run_per_universe_compares(
            legs,
            state=state,
            study_dir=study_dir,
            force_rerun=force_rerun,
        )
        write_study_state(state_path, state)

    _logger.info(
        "study '%s' complete: %d legs done, %d failed, %d skipped, %d compares",
        spec.name,
        n_completed,
        n_failed,
        n_skipped,
        n_compares,
    )
    return StudyRunResult(
        study_dir=study_dir,
        state=state,
        n_legs_completed=n_completed,
        n_legs_failed=n_failed,
        n_legs_skipped=n_skipped,
        n_compares_done=n_compares,
    )


def _run_per_universe_compares(
    legs: Sequence[StudyLegRun],
    *,
    state: StudyState,
    study_dir: Path,
    force_rerun: bool,
) -> tuple[StudyState, int]:
    """Group completed runs by universe; run cross-strategy compare per universe.

    Universes covered by a single strategy (e.g. ivv_voo_daily_5y, the
    only pairs universe) are silently skipped — pairwise ranking against
    one strategy is undefined.
    """

    by_universe: dict[str, list[StudyLegRun]] = {}
    for leg in legs:
        leg_state = state.get_leg(leg.leg_id)
        if not leg_state.is_complete or leg_state.run_experiment_id is None:
            continue
        by_universe.setdefault(leg.universe, []).append(leg)

    n_done = 0
    for universe, universe_legs in sorted(by_universe.items()):
        if len(universe_legs) < 2:
            _logger.info(
                "compare for universe %s: only %d strategy — skipping",
                universe,
                len(universe_legs),
            )
            continue
        if universe in state.cross_strategy_compares_done and not force_rerun:
            _logger.info("compare for universe %s: already done — skipping", universe)
            continue

        run_dirs = [
            resolve_run_dir(study_dir, cast(str, state.get_leg(leg.leg_id).run_experiment_id))
            for leg in universe_legs
        ]
        try:
            results = [load_experiment_result(d) for d in run_dirs]
            _logger.info(
                "compare for universe %s: %d strategies",
                universe,
                len(universe_legs),
            )
            report, folds_by_strategy = run_comparison(
                out_name=universe,
                store_root=study_dir,
                n_jobs=1,
                significance_test=SignificanceTest.BOOTSTRAP,
                reused_results=results,
            )
            cmp_dir = study_dir / COMPARISONS_SUBDIR / universe
            ComparisonReporter().generate_full_report(
                report,
                cmp_dir,
                folds_by_strategy=folds_by_strategy,
                publish_label=f"study:cmp:{universe}",
            )
            state = state.with_compare_done(universe)
            n_done += 1
        except Exception as exc:  # noqa: BLE001 — continue with remaining universes
            tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            _logger.error("compare for universe %s: failed (%s)", universe, tb)
    return state, n_done


def _read_yaml(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"YAML at {path} must be a mapping at the top level, got {type(raw).__name__}"
        )
    return raw


def _resolve_under(repo_root: Path, p: Path) -> Path:
    return p if p.is_absolute() else repo_root / p


logging.getLogger(__name__).addHandler(logging.NullHandler())
