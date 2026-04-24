"""Per-trial callback writing ``trials.jsonl`` + ``best_config.yaml``.

Optuna calls the registered callback after every trial finishes (complete,
pruned, or failed). We:

1. Append a line to ``trials.jsonl`` — append-only so a ``tail -f`` on a
   running study shows progress; JSON object per line so downstream
   analysis can stream-parse.
2. Refresh ``best_config.yaml`` if the trial is COMPLETE and is the new
   study best. The YAML is a fully-materialised :class:`ExperimentConfig`
   — a user can drop it straight into ``experiment run --config`` for a
   final re-train on the full dev region, or feed it into a holdout-eval
   pipeline.

Checkpointing per trial (rather than every N) is intentional: a YAML
write is <1 ms, the cost of re-checkpointing on a non-best trial is
a no-op file existence check, and the guarantee "study_dir always
holds the best config so far" is worth more than the tiny disk churn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import optuna

from src.core.config import write_frozen_yaml
from src.optimization.sampling import sample_trial_params

if TYPE_CHECKING:
    from src.core.config import ExperimentConfig

_logger = logging.getLogger(__name__)

# Files written under the study directory.  Both are referenced by the
# tuner (to find the latest best config / to tail the log) so they live
# here alongside the code that writes them.
BEST_CONFIG_YAML_NAME = "best_config.yaml"
TRIALS_JSONL_NAME = "trials.jsonl"


@dataclass(frozen=True)
class TrialCallback:
    """Optuna ``callback`` wrapper — frozen so it's safe across worker threads."""

    experiment_cfg: ExperimentConfig
    study_dir: Path

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self._append_trial_record(trial)
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        try:
            best = study.best_trial
        except ValueError:
            return  # no completed trials yet — nothing to checkpoint
        if best.number != trial.number:
            return
        self._refresh_best_config(best)

    def _append_trial_record(self, trial: optuna.trial.FrozenTrial) -> None:
        record: dict[str, object] = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
            "datetime_start": (
                trial.datetime_start.isoformat() if trial.datetime_start is not None else None
            ),
            "datetime_complete": (
                trial.datetime_complete.isoformat() if trial.datetime_complete is not None else None
            ),
        }
        path = self.study_dir / TRIALS_JSONL_NAME
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")

    def _refresh_best_config(self, best: optuna.trial.FrozenTrial) -> None:
        """Write ``best_config.yaml`` = base config + best trial's sampled kwargs.

        Uses :class:`optuna.trial.FixedTrial` to replay the strategy's
        ``suggest_params`` with the stored Optuna-namespaced params —
        that's the only way to go from the Optuna-internal name space
        (``"retf_arma_p_max"``) back to ctor-kwarg space
        (``"arma_p_max"``). The filter inside ``sample_trial_params``
        also reruns so pinned-leaf keys stay out of the best config,
        identical to how the HPO sampler produced them.
        """
        fixed = optuna.trial.FixedTrial(best.params)
        resolved = sample_trial_params(self.experiment_cfg, fixed)
        materialized = _merge_params(self.experiment_cfg, resolved)
        write_frozen_yaml(self.study_dir / BEST_CONFIG_YAML_NAME, materialized)
        _logger.info(
            "best_config.yaml refreshed: trial=%d value=%s",
            best.number,
            best.value,
        )


def _merge_params(base: ExperimentConfig, sampled: dict[str, object]) -> ExperimentConfig:
    """Return a fresh ``ExperimentConfig`` with ``sampled`` merged into strategy.params.

    Separate from :func:`src.optimization.tuner._materialize_trial_config`
    because the ``best_config.yaml`` output keeps the ORIGINAL name (a
    user who loads the file should see their original config name, not
    ``"<name>_trial"``).
    """
    payload = base.model_dump(mode="json")
    strategy_payload = dict(payload["strategy"])
    current_params = dict(strategy_payload.get("params", {}))
    strategy_payload["params"] = {**current_params, **sampled}
    payload["strategy"] = strategy_payload
    return base.model_validate(payload)
