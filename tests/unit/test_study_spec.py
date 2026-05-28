"""
Tests for :class:`StudySpec` and the committed ``main_study.yaml``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import (
    StudyLeg,
    StudySpec,
    load_study_spec,
    load_universe_profile,
)
from tests.conftest import REPO_ROOT

MAIN_STUDY_PATH = REPO_ROOT / "config" / "study" / "main_study.yaml"
UNIVERSE_DIR = REPO_ROOT / "config" / "universes"


@pytest.fixture(scope="module")
def main_study() -> StudySpec:
    return load_study_spec(MAIN_STUDY_PATH)


def _minimal_leg_dict(strategy: str = "AdaptiveBollinger") -> dict[str, object]:
    return {
        "strategy": strategy,
        "strategy_config": "config/strategies/adaptive_bollinger.yaml",
        "hpo_config": "config/hpo/adaptive_bollinger.yaml",
        "universes": ["spy_daily_5y"],
    }


class TestStudySpecSchema:
    def test_valid_spec_loads(self) -> None:
        spec = StudySpec.model_validate(
            {
                "name": "tiny",
                "output_dir": "out",
                "legs": [_minimal_leg_dict()],
            }
        )
        assert spec.name == "tiny"
        assert spec.seed == 42
        assert len(spec.legs) == 1

    def test_empty_universe_list_rejected(self) -> None:
        leg = _minimal_leg_dict()
        leg["universes"] = []
        with pytest.raises(ValidationError):
            StudyLeg.model_validate(leg)

    def test_duplicate_universe_in_leg_rejected(self) -> None:
        leg = _minimal_leg_dict()
        leg["universes"] = ["spy_daily_5y", "spy_daily_5y"]
        with pytest.raises(ValidationError, match="duplicate universe"):
            StudyLeg.model_validate(leg)

    def test_duplicate_strategy_across_legs_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate strategy"):
            StudySpec.model_validate(
                {
                    "name": "tiny",
                    "output_dir": "out",
                    "legs": [
                        _minimal_leg_dict("AdaptiveBollinger"),
                        _minimal_leg_dict("AdaptiveBollinger"),
                    ],
                }
            )

    def test_empty_legs_list_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StudySpec.model_validate({"name": "tiny", "output_dir": "out", "legs": []})

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StudySpec.model_validate(
                {
                    "name": "tiny",
                    "output_dir": "out",
                    "legs": [_minimal_leg_dict()],
                    "extra_field": "boom",
                }
            )


class TestMainStudyYAML:
    def test_main_study_yaml_loads(self, main_study: StudySpec) -> None:
        assert main_study.name == "main_study"

    def test_every_referenced_strategy_config_exists(self, main_study: StudySpec) -> None:
        for leg in main_study.legs:
            path = REPO_ROOT / leg.strategy_config
            assert path.is_file(), f"missing strategy config: {path}"

    def test_every_referenced_hpo_config_exists(self, main_study: StudySpec) -> None:
        for leg in main_study.legs:
            path = REPO_ROOT / leg.hpo_config
            assert path.is_file(), f"missing hpo config: {path}"

    def test_every_referenced_universe_resolves_and_parses(self, main_study: StudySpec) -> None:
        for leg in main_study.legs:
            for name in leg.universes:
                path = UNIVERSE_DIR / f"{name}.yaml"
                assert path.is_file(), f"missing universe: {path}"
                load_universe_profile(path)

    def test_pairs_strategy_only_runs_on_pairs_universe(self, main_study: StudySpec) -> None:
        pairs_legs = [leg for leg in main_study.legs if leg.strategy == "PairsTrading"]
        assert len(pairs_legs) == 1
        for name in pairs_legs[0].universes:
            profile = load_universe_profile(UNIVERSE_DIR / f"{name}.yaml")
            assert len(profile.data.tickers) == 2, (
                f"PairsTrading leg references non-pair universe '{name}' "
                f"with tickers {profile.data.tickers}"
            )

    def test_single_asset_strategies_run_on_single_ticker_universes(
        self, main_study: StudySpec
    ) -> None:
        for leg in main_study.legs:
            if leg.strategy == "PairsTrading":
                continue
            for name in leg.universes:
                profile = load_universe_profile(UNIVERSE_DIR / f"{name}.yaml")
                assert len(profile.data.tickers) == 1, (
                    f"single-asset strategy '{leg.strategy}' references multi-ticker "
                    f"universe '{name}' with tickers {profile.data.tickers}"
                )
