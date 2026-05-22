"""Optuna-backed joint HPO driver over one :class:`ExperimentConfig`.

Each trial materialises a fresh :class:`ExperimentConfig` by merging the
sampler's draw into ``experiment_cfg.strategy.params``, runs a full
walk-forward :class:`Experiment` under the study's trial-artefacts
subdirectory, and returns the configured objective value.

Directory layout
----------------
::

    <store_root>/hpo/<study_name>/
        optuna_study.db           # SQLite — enables cross-process resume
        experiment_config.yaml    # frozen copy of the base config
        hpo_config.yaml           # frozen copy of the HPO config
        best_config.yaml          # refreshed per new-best trial
        trials.jsonl              # append-only per-trial record
        trials_artifacts/         # store_root for per-trial Experiment.run()
            runs/<experiment_id>/
                ...
        plots/ tables/            # produced by :func:`generate_hpo_report`

Resume semantics
----------------
Re-running with the same ``study_name`` + ``store_root`` loads the
existing SQLite study and runs ``n_trials`` MORE trials (Optuna's own
semantics — the count is "additional trials", not "target total"). The
base ``experiment_config.yaml`` is written once on first run; if the
user passes a different config under the same study name the SQLite
trials are still valid-as-executed but the objective may have changed
shape, so we check equality by content hash before appending.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import optuna
import yaml

from src.analysis.metrics_aggregator import aggregate_folds
from src.core.config import (
    ExperimentConfig,
    load_experiment_config,
    write_frozen_yaml,
)
from src.core.hpo_config import HPOConfig
from src.core.logging import get_logger
from src.core.persistence import HPO_SUBDIR
from src.models._garch_cache import GarchGridCache, garch_cache_context
from src.optimization.checkpointing import (
    BEST_CONFIG_YAML_NAME,
    TRIAL_ARTIFACTS_SUBDIR,
    TRIALS_JSONL_NAME,
    TrialCallback,
)
from src.optimization.objectives import IObjective, build_objective
from src.optimization.pruners import build_pruner
from src.optimization.samplers import build_sampler
from src.optimization.sampling import sample_trial_params
from src.orchestration.builder import build_experiment
from src.orchestration.experiment import RunOptions

if TYPE_CHECKING:
    from src.orchestration.types import ExperimentResult

_logger = get_logger(__name__)

_DEFAULT_STORE_ROOT = Path("experiment_results")
STUDY_DB_FILENAME = "optuna_study.db"
EXPERIMENT_CONFIG_YAML = "experiment_config.yaml"
HPO_CONFIG_YAML = "hpo_config.yaml"
# Re-export for downstream consumers that want to read the checkpoint files.
BEST_CONFIG_YAML = BEST_CONFIG_YAML_NAME
TRIALS_JSONL = TRIALS_JSONL_NAME
# Optuna stores the full ``experiment_id`` each trial ran under so users
# can cross-reference trial.params with the full artefact directory.
USER_ATTR_EXPERIMENT_ID = "experiment_id"
_TRIAL_NAME_SUFFIX = "_trial"


@dataclass(frozen=True)
class StrategyTuner:
    """Drive one Optuna study against one :class:`ExperimentConfig`.

    Construction is deliberately minimal — the interesting work happens
    in :meth:`run`. Frozen because the tuner is pure configuration: once
    built it's a read-only handle to a study, and accidental ctor-arg
    reassignment mid-run (e.g. mutating ``store_root`` after ``study_dir``
    has been memoised implicitly by logs / artefacts on disk) would be a
    bug we'd rather not be able to write.
    """

    experiment_cfg: ExperimentConfig
    hpo_cfg: HPOConfig
    store_root: Path | None = None

    @property
    def study_dir(self) -> Path:
        store = self.store_root if self.store_root is not None else _DEFAULT_STORE_ROOT
        return store / HPO_SUBDIR / self.hpo_cfg.study_name

    @cached_property
    def _study_logger(self) -> logging.LoggerAdapter:  # type: ignore[type-arg]
        """Per-study context-bound logger; built once, reused across trials."""
        return get_logger(__name__, study=self.hpo_cfg.study_name)

    @property
    def storage_url(self) -> str:
        """SQLite URL Optuna stores the study under.

        Absolute path so the URL is invariant to the working directory
        the tuner is invoked from — matters for resume from a different
        shell or CI worker.
        """
        return storage_url_for(self.study_dir)

    def run(self, *, progress: bool = False) -> optuna.Study:
        """Run the study end-to-end, returning the completed study.

        Creates ``study_dir`` if missing, persists configs on first run,
        builds sampler/pruner from the HPO config, and drives Optuna's
        optimize loop with a :class:`TrialCallback` that refreshes
        ``best_config.yaml`` after every completed trial. When
        ``progress=True``, Optuna's built-in ``show_progress_bar`` renders a
        per-trial bar to TTY (silently disabled in non-interactive runs).
        """
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self._persist_configs()

        study = optuna.create_study(
            study_name=self.hpo_cfg.study_name,
            storage=self.storage_url,
            direction="maximize",
            sampler=build_sampler(self.hpo_cfg.sampler, self.hpo_cfg.seed),
            pruner=build_pruner(self.hpo_cfg.pruner),
            load_if_exists=True,
        )

        objective = build_objective(self.hpo_cfg.objective)
        callback = TrialCallback(
            experiment_cfg=self.experiment_cfg,
            study_dir=self.study_dir,
        )

        _logger.info(
            "study '%s' starting: n_trials=%d n_jobs=%d sampler=%s pruner=%s objective=%s",
            self.hpo_cfg.study_name,
            self.hpo_cfg.n_trials,
            self.hpo_cfg.n_jobs,
            self.hpo_cfg.sampler.value,
            self.hpo_cfg.pruner.value,
            self.hpo_cfg.objective.value,
        )

        # Bind a study-scoped GARCH grid cache so two trials whose
        # ``(p_max, q_max)`` overlap on the same fold reuse one another's
        # AIC tables instead of re-fitting every cell. Strategies that
        # don't construct GARCH simply never read the ContextVar.
        with garch_cache_context(GarchGridCache()):
            study.optimize(
                lambda trial: self._objective(trial, objective),
                n_trials=self.hpo_cfg.n_trials,
                n_jobs=self.hpo_cfg.n_jobs,
                timeout=self.hpo_cfg.timeout_s,
                callbacks=[callback],
                show_progress_bar=progress,
            )
        return study

    def _objective(self, trial: optuna.Trial, objective: IObjective) -> float:
        start = time.monotonic()
        sampled = sample_trial_params(self.experiment_cfg, trial)
        trial_cfg = _materialize_trial_config(self.experiment_cfg, sampled)
        experiment = build_experiment(trial_cfg)
        result: ExperimentResult = experiment.run(
            RunOptions(
                store_root=self.study_dir / TRIAL_ARTIFACTS_SUBDIR,
                write_report=False,
            )
        )
        trial.set_user_attr(USER_ATTR_EXPERIMENT_ID, result.experiment_id)
        metrics = aggregate_folds(result.folds).to_dict()
        value = objective(metrics)
        elapsed = time.monotonic() - start
        # ``best_trial`` is direction-aware (Optuna picks max for "maximize"
        # and min for "minimize"); using it instead of a hand-rolled compare
        # keeps the log honest if the study direction ever changes. It raises
        # ValueError before any trial completes — only the very first trial
        # hits that branch and there's nothing better than itself yet.
        try:
            best = trial.study.best_trial
            best_value = best.value if best.value is not None else value
            best_number = best.number
        except ValueError:
            best_value = value
            best_number = trial.number
        self._study_logger.info(
            "trial %d/%d experiment_id=%s value=%.6f best=%.6f (trial %d) elapsed=%.1fs",
            trial.number + 1,
            self.hpo_cfg.n_trials,
            result.experiment_id,
            value,
            best_value,
            best_number,
            elapsed,
        )
        return value

    def _persist_configs(self) -> None:
        """Write frozen ``experiment_config.yaml`` + ``hpo_config.yaml`` once.

        On resume the configs already exist; we verify the base
        experiment config hash matches the incoming one so a user who
        quietly edits a YAML and re-runs under the same study name gets
        a pointed error instead of studying under two different
        objectives.

        Both files are written together — a crash that leaves only one
        on disk is treated as a half-initialised study dir and the next
        run re-writes whichever is missing, so the "both frozen yamls
        present" invariant is always restored on a clean rerun.
        """
        exp_path = self.study_dir / EXPERIMENT_CONFIG_YAML
        hpo_path = self.study_dir / HPO_CONFIG_YAML
        if exp_path.exists():
            stored = load_experiment_config(exp_path)
            if _config_content_hash(stored) != _config_content_hash(self.experiment_cfg):
                raise ValueError(
                    f"study '{self.hpo_cfg.study_name}' was created under a "
                    f"different experiment config; the SQLite trials would "
                    f"belong to a different objective. Fix by choosing a new "
                    f"study_name, or by reverting the config at {exp_path} "
                    f"to its original content."
                )
            if hpo_path.exists():
                return
            # Partial init: experiment_config.yaml survived but hpo_config.yaml
            # didn't. Rewrite the missing sibling so downstream readers see a
            # complete study dir.
            write_frozen_yaml(hpo_path, self.hpo_cfg)
            return
        write_frozen_yaml(exp_path, self.experiment_cfg)
        write_frozen_yaml(hpo_path, self.hpo_cfg)


def storage_url_for(study_dir: Path) -> str:
    """SQLite URL Optuna stores the study DB under ``study_dir``.

    Uses ``Path.as_posix()`` so the URL is well-formed on Windows too —
    SQLAlchemy expects forward slashes regardless of platform.
    """
    db_path = (study_dir / STUDY_DB_FILENAME).resolve()
    return f"sqlite:///{db_path.as_posix()}"


def _materialize_trial_config(
    base: ExperimentConfig, sampled: dict[str, object]
) -> ExperimentConfig:
    """Merge sampled ctor kwargs into ``base.strategy.params`` and re-validate.

    Revalidation is required — registry lookups and strategy-name
    validation live on the ``ExperimentConfig`` validators and we want a
    trial's config to pass the same gates as any user-authored YAML.
    """
    payload = base.model_dump(mode="json")
    strategy_payload = dict(payload["strategy"])
    current_params = dict(strategy_payload.get("params", {}))
    merged_params: dict[str, object] = {**current_params, **sampled}
    strategy_payload["params"] = merged_params
    payload["strategy"] = strategy_payload
    payload["name"] = f"{base.name}{_TRIAL_NAME_SUFFIX}"
    return ExperimentConfig.model_validate(payload)


def _config_content_hash(cfg: ExperimentConfig) -> str:
    """Stable SHA over a model's JSON dump — content equality, not object identity."""
    payload = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
