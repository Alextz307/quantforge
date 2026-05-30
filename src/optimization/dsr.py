"""
Compute and persist the deflated Sharpe ratio for a completed HPO study.

A small post-tune enrichment step. Reads the completed Optuna study,
pulls the per-trial Sharpes, and writes ``dsr.json`` into the same
``<hpo_dir>`` the tuner already produced (next to ``best_config.yaml``
and ``optuna_study.db``).

Why here, not inside the tuner
------------------------------
The tuner's job is to drive Optuna; deflation is a downstream
interpretation of the search outcome. Keeping it separate means a
future change to the DSR estimator (e.g. swapping the trial-moment
heuristic for the strict BLP form once per-trial returns are stored)
doesn't touch the optimisation loop.
"""

from __future__ import annotations

from pathlib import Path

import optuna

from src.analysis.significance import DeflatedSharpe, deflated_sharpe_ratio
from src.core import json_io
from src.core.logging import get_logger
from src.core.persistence import DSR_JSON_FILENAME

_logger = get_logger(__name__)


def compute_and_write_dsr(
    study: optuna.Study,
    hpo_dir: Path,
    *,
    n_dev_bars: int,
) -> tuple[DeflatedSharpe | None, Path]:
    """
    Compute the deflated Sharpe for ``study`` and persist it to disk.

    Returns ``(DeflatedSharpe | None, path)``. ``None`` indicates the
    study had zero completed trials with non-None objective values - the
    deflation is undefined and the JSON is not written. Callers should
    treat that case as "DSR unavailable" rather than a failure.

    ``n_dev_bars`` is the length of the dev region the study tuned on
    (typically ``len(bars_full.loc[bars_full.index < boundary])``); it
    enters BLP eq.(9) as the sample length. The caller passes this in
    so DSR computation stays free of any data-fetch dependency.

    The path is always returned (even on the None case) so the caller
    can log a stable location regardless of the outcome.
    """

    out_path = hpo_dir / DSR_JSON_FILENAME

    trial_sharpes = [
        t.value
        for t in study.trials
        if t.state is optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not trial_sharpes:
        _logger.warning(
            "study '%s' has no completed trials with non-None values - skipping DSR",
            study.study_name,
        )
        return None, out_path

    if n_dev_bars < 2:
        _logger.warning(
            "study '%s' dev region has %d bars (<2) - skipping DSR",
            study.study_name,
            n_dev_bars,
        )
        return None, out_path

    dsr = deflated_sharpe_ratio(trial_sharpes, sample_length=n_dev_bars)
    json_io.write(out_path, dsr.to_dict())
    _logger.info(
        "study '%s' DSR: observed=%.4f deflated=%.4f n_trials=%d expected_max=%.4f",
        study.study_name,
        dsr.observed_sharpe,
        dsr.deflated_sharpe,
        dsr.n_trials,
        dsr.expected_max_sharpe,
    )
    return dsr, out_path


__all__ = ["compute_and_write_dsr"]
