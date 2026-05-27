"""Trial-parameter composition for :func:`sample_trial_params`.

Each strategy already owns its ``suggest_params`` — the sampler delegates
to the registered ``strategy_cls`` and returns a fresh dict per call.
"""

from __future__ import annotations

import optuna
import pytest

from src.core.config import ExperimentConfig
from src.core.registry import strategy_registry
from src.optimization.sampling import sample_trial_params

_SPY_DATA = {
    "source": "csv",
    "tickers": ["SPY"],
    "start": "2020-01-01",
    "end": "2024-01-01",
    "interval": "daily",
}
_FEATURE_COLUMNS = ["return_5d", "rsi_14", "vol_20"]


def _fresh_trial() -> optuna.Trial:
    """One-shot Optuna Trial that accepts arbitrary ``suggest_*`` calls."""

    study = optuna.create_study(direction="maximize")
    return study.ask()


def _build_cfg(
    strategy_name: str,
    *,
    strategy_params: dict[str, object] | None = None,
) -> ExperimentConfig:
    payload: dict[str, object] = {
        "name": f"test_{strategy_name}",
        "seed": 42,
        "data": _SPY_DATA,
        "strategy": {
            "name": strategy_name,
            "params": strategy_params if strategy_params is not None else {},
        },
    }
    return ExperimentConfig.model_validate(payload)


class TestSampleTrialParams:
    @pytest.mark.parametrize(
        "strategy_name",
        ["AdaptiveBollinger", "PairsTrading", "MomentumGatekeeper"],
    )
    def test_passthrough_matches_strategy_suggest(self, strategy_name: str) -> None:
        cfg = _build_cfg(strategy_name, strategy_params={"feature_columns": _FEATURE_COLUMNS})
        trial_a = _fresh_trial()
        expected = strategy_registry.get(strategy_name).suggest_params(trial_a)

        trial_b = _fresh_trial()
        actual = sample_trial_params(cfg, trial_b)

        assert set(actual) == set(expected)

    def test_return_forecast_search_space(self) -> None:
        cfg = _build_cfg("ReturnForecast", strategy_params={"feature_columns": _FEATURE_COLUMNS})
        actual = sample_trial_params(cfg, _fresh_trial())
        assert "position_scale" in actual
        assert "max_leverage" in actual
        assert "arma_p_max" in actual
        assert "lstm_hidden_dim" in actual
        assert "lstm_lookback" in actual

    def test_volatility_targeting_search_space(self) -> None:
        cfg = _build_cfg(
            "VolatilityTargeting", strategy_params={"feature_columns": _FEATURE_COLUMNS}
        )
        actual = sample_trial_params(cfg, _fresh_trial())
        assert "target_vol" in actual
        assert "lstm_hidden_dim" in actual
        assert "realized_vol_window" in actual


class TestSampleTrialParamsReturnsFreshDict:
    def test_each_call_returns_new_dict(self) -> None:
        cfg = _build_cfg("AdaptiveBollinger", strategy_params={})
        first = sample_trial_params(cfg, _fresh_trial())
        second = sample_trial_params(cfg, _fresh_trial())
        assert first is not second
        first["window"] = -999  # mutating caller copy must not affect next call
        reread = sample_trial_params(cfg, _fresh_trial())
        assert reread["window"] != -999
