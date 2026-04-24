"""Trial-parameter composition + pretrained-leaf filter.

Each strategy already owns its ``suggest_params`` — the sampler's only
value-add is dropping ctor kwargs that a pinned pretrained leaf owns.
Tests pair every injection-capable strategy with its corresponding
leaf-key entry in :data:`_LEAF_KEY_OWNED_PARAMS` and assert the intersection
of ``suggest_params`` keys with that owned set is removed (and nothing
outside that intersection is touched).
"""

from __future__ import annotations

from pathlib import Path

import optuna
import pytest

from src.core.config import _LEAF_KEY_OWNED_PARAMS, ExperimentConfig
from src.core.registry import strategy_registry
from src.optimization.sampling import sample_trial_params

_SPY_DATA = {
    "source": "csv",
    "tickers": ["SPY"],
    "start": "2020-01-01",
    "end": "2024-01-01",
    "interval": "daily",
}
_FEATURE_COLUMNS = ["sma_20", "rsi_14", "volume_z"]


def _fresh_trial() -> optuna.Trial:
    """One-shot Optuna Trial that accepts arbitrary ``suggest_*`` calls."""
    study = optuna.create_study(direction="maximize")
    return study.ask()


def _build_cfg(
    strategy_name: str,
    *,
    strategy_params: dict[str, object] | None = None,
    pretrained_leaves: dict[str, Path] | None = None,
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
    if pretrained_leaves is not None:
        payload["pretrained_leaves"] = {k: str(v) for k, v in pretrained_leaves.items()}
    return ExperimentConfig.model_validate(payload)


class TestSampleTrialParamsNoPinning:
    """With no pretrained_leaves, the filter is a no-op passthrough."""

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

    def test_return_forecast_passthrough(self) -> None:
        cfg = _build_cfg("ReturnForecast", strategy_params={"feature_columns": _FEATURE_COLUMNS})
        actual = sample_trial_params(cfg, _fresh_trial())
        # Full ReturnForecast search space — includes all arma_*, lstm_* knobs.
        assert "position_scale" in actual
        assert "max_leverage" in actual
        assert "arma_p_max" in actual
        assert "lstm_hidden_dim" in actual
        assert "lstm_lookback" in actual

    def test_volatility_targeting_passthrough(self) -> None:
        cfg = _build_cfg(
            "VolatilityTargeting", strategy_params={"feature_columns": _FEATURE_COLUMNS}
        )
        actual = sample_trial_params(cfg, _fresh_trial())
        assert "target_vol" in actual
        assert "lstm_hidden_dim" in actual
        assert "realized_vol_window" in actual


class TestSampleTrialParamsPinnedLeafFilter:
    """With a pinned leaf, owned ctor kwargs are dropped from the sampled dict."""

    def _pinned_cfg(
        self,
        strategy_name: str,
        leaf_key: str,
        tmp_path: Path,
    ) -> ExperimentConfig:
        leaf_dir = tmp_path / leaf_key
        leaf_dir.mkdir()
        return _build_cfg(
            strategy_name,
            strategy_params={"feature_columns": _FEATURE_COLUMNS},
            pretrained_leaves={leaf_key: leaf_dir},
        )

    def test_return_forecast_return_model_pinned(self, tmp_path: Path) -> None:
        cfg = self._pinned_cfg("ReturnForecast", "return_model", tmp_path)
        owned = set(_LEAF_KEY_OWNED_PARAMS["ReturnForecast"]["return_model"])

        # Raw suggest for comparison — fresh trial, so Optuna state doesn't leak.
        raw = strategy_registry.get("ReturnForecast").suggest_params(_fresh_trial())
        filtered = sample_trial_params(cfg, _fresh_trial())

        # Every owned key present in raw is dropped from filtered.
        expected_dropped = owned & set(raw)
        assert expected_dropped, "fixture must exercise the filter — owned ∩ raw nonempty"
        assert not (expected_dropped & set(filtered))

        # No non-owned key was touched.
        assert set(filtered) == set(raw) - owned

        # Sanity: strategy-level knobs that are NOT leaf-owned survive.
        assert "position_scale" in filtered
        assert "max_leverage" in filtered
        # ``lstm_lookback`` is deliberately excluded from owned params
        # (strategy still needs it to build batches for the frozen leaf).
        assert "lstm_lookback" in filtered

    def test_volatility_targeting_vol_model_pinned(self, tmp_path: Path) -> None:
        cfg = self._pinned_cfg("VolatilityTargeting", "vol_model", tmp_path)
        owned = set(_LEAF_KEY_OWNED_PARAMS["VolatilityTargeting"]["vol_model"])

        raw = strategy_registry.get("VolatilityTargeting").suggest_params(_fresh_trial())
        filtered = sample_trial_params(cfg, _fresh_trial())

        expected_dropped = owned & set(raw)
        assert expected_dropped
        assert not (expected_dropped & set(filtered))
        assert set(filtered) == set(raw) - owned

        assert "target_vol" in filtered
        assert "realized_vol_window" in filtered
        assert "trend_window" in filtered
        assert "lstm_lookback" in filtered

    def test_momentum_gatekeeper_classifier_pinned(self, tmp_path: Path) -> None:
        cfg = self._pinned_cfg("MomentumGatekeeper", "directional_classifier", tmp_path)
        owned = set(_LEAF_KEY_OWNED_PARAMS["MomentumGatekeeper"]["directional_classifier"])

        raw = strategy_registry.get("MomentumGatekeeper").suggest_params(_fresh_trial())
        filtered = sample_trial_params(cfg, _fresh_trial())

        expected_dropped = owned & set(raw)
        assert expected_dropped
        assert not (expected_dropped & set(filtered))
        assert set(filtered) == set(raw) - owned

        # MACD / MA / RSI gates are strategy-level — must survive.
        assert "macd_fast" in filtered
        assert "macd_slow" in filtered
        assert "ma_window" in filtered
        assert "rsi_period" in filtered


class TestSampleTrialParamsReturnsFreshDict:
    def test_each_call_returns_new_dict(self) -> None:
        cfg = _build_cfg("AdaptiveBollinger", strategy_params={})
        first = sample_trial_params(cfg, _fresh_trial())
        second = sample_trial_params(cfg, _fresh_trial())
        assert first is not second
        first["window"] = -999  # mutating caller copy must not affect next call
        reread = sample_trial_params(cfg, _fresh_trial())
        assert reread["window"] != -999
