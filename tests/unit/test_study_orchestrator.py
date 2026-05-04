"""Unit tests for the empirical-study orchestrator.

Covers leg expansion, universe-profile composition, pretrained-leaf
path rewriting, HPO study_name override, study-state round-trip + atomic
write, resume logic (spec_hash mismatch), and train-leaves leaf-pair
discovery. The actual tune/run/compare pipeline is exercised end-to-end
in ``tests/integration/test_study_smoke.py`` (gated, ~60s).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.core.config import StudySpec, load_study_spec
from src.orchestration.study import (
    LEAF_KEY_TO_TEMPLATE_YAML,
    SPEC_SNAPSHOT_FILENAME,
    STUDY_STATE_FILENAME,
    _collect_pretrained_leaf_pairs,
    _compose_standalone_model_config,
    compose_hpo_config,
    compose_leg_config,
    expand_spec_into_legs,
    expected_pretrained_leaf_path,
    make_leg_id,
    resolve_study_dir,
)
from src.orchestration.study_state import (
    LEG_STEP_RUN,
    LEG_STEP_TUNE,
    LegState,
    StudyState,
    compute_spec_hash,
    read_study_state,
    write_study_state,
)
from tests.conftest import REPO_ROOT

MAIN_STUDY_PATH = REPO_ROOT / "config" / "study" / "main_study.yaml"

# Spec-derived counts (kept beside the spec path so a spec edit updates one
# place). main_study composition: AdaptiveBollinger(11) + PairsTrading(1) +
# MomentumGatekeeper(9) + VolatilityTargeting(9) + ReturnForecast(9) = 39.
EXPECTED_MAIN_STUDY_LEG_COUNT = 39
# 9 universes * 3 ML-bearing strategies (Momentum, ReturnForecast,
# VolatilityTargeting) = 27 (universe, leaf_key) pairs.
EXPECTED_MAIN_STUDY_PRETRAINED_PAIR_COUNT = 27


@pytest.fixture(scope="module")
def main_spec() -> StudySpec:
    return load_study_spec(MAIN_STUDY_PATH)


def _write_minimal_spec(tmp_path: Path, output_dir: str = "studies/test") -> Path:
    """Produce a tiny 2-strategy x 2-universe StudySpec on disk."""
    payload: dict[str, Any] = {
        "name": "test_study",
        "output_dir": output_dir,
        "legs": [
            {
                "strategy": "AdaptiveBollinger",
                "strategy_config": "config/strategies/adaptive_bollinger.yaml",
                "hpo_config": "config/hpo/adaptive_bollinger.yaml",
                "universes": ["spy_daily_5y", "qqq_daily_5y"],
            },
            {
                "strategy": "PairsTrading",
                "strategy_config": "config/strategies/pairs_trading.yaml",
                "hpo_config": "config/hpo/pairs_trading.yaml",
                "universes": ["ivv_voo_daily_5y"],
            },
        ],
    }
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(payload, default_flow_style=False))
    return path


class TestLegIdFormat:
    def test_make_leg_id_concatenates_with_double_underscore(self) -> None:
        assert make_leg_id("AdaptiveBollinger", "spy_daily_5y") == "AdaptiveBollinger__spy_daily_5y"


class TestExpandSpec:
    def test_cross_product_size(self, tmp_path: Path) -> None:
        spec = load_study_spec(_write_minimal_spec(tmp_path))
        legs = expand_spec_into_legs(spec, repo_root=REPO_ROOT)
        # 2 (AB universes) + 1 (Pairs universe) = 3 legs
        assert len(legs) == 3

    def test_leg_ids_unique_and_well_formed(self, main_spec: StudySpec) -> None:
        legs = expand_spec_into_legs(main_spec, repo_root=REPO_ROOT)
        leg_ids = [leg.leg_id for leg in legs]
        assert len(leg_ids) == len(set(leg_ids))
        for leg in legs:
            assert leg.leg_id == f"{leg.strategy}__{leg.universe}"

    def test_main_study_yaml_yields_expected_leg_count(self, main_spec: StudySpec) -> None:
        legs = expand_spec_into_legs(main_spec, repo_root=REPO_ROOT)
        assert len(legs) == EXPECTED_MAIN_STUDY_LEG_COUNT

    def test_universe_profile_paths_resolve(self, main_spec: StudySpec) -> None:
        legs = expand_spec_into_legs(main_spec, repo_root=REPO_ROOT)
        for leg in legs:
            assert leg.universe_profile_path.is_file(), (
                f"expand_spec_into_legs returned a non-existent universe path: "
                f"{leg.universe_profile_path}"
            )


class TestExpectedPretrainedLeafPath:
    def test_path_format(self) -> None:
        path = expected_pretrained_leaf_path(
            store_root=Path("/tmp/store"),
            universe="spy_daily_5y",
            leaf_key="vol_model",
        )
        assert path == Path("/tmp/store/models/spy_daily_5y_vol_model")


class TestComposeLegConfig:
    def test_non_pretrained_leg_composes(self, tmp_path: Path) -> None:
        spec = load_study_spec(_write_minimal_spec(tmp_path))
        legs = expand_spec_into_legs(spec, repo_root=REPO_ROOT)
        ab_qqq = next(leg for leg in legs if leg.universe == "qqq_daily_5y")
        cfg = compose_leg_config(ab_qqq, store_root=tmp_path)
        assert cfg.name == "AdaptiveBollinger__qqq_daily_5y"
        assert cfg.data.tickers == ["QQQ"]
        assert cfg.validation.holdout_pct > 0.0

    def test_pretrained_leg_rewrites_paths(self, tmp_path: Path, main_spec: StudySpec) -> None:
        legs = expand_spec_into_legs(main_spec, repo_root=REPO_ROOT)
        vt_leg = next(
            leg
            for leg in legs
            if leg.strategy == "VolatilityTargeting" and leg.universe == "spy_daily_5y"
        )
        store_root = tmp_path / "store"
        # The path validator on ExperimentConfig requires the artifact dir
        # to exist — the orchestrator's normal flow runs train-leaves first.
        artifact = expected_pretrained_leaf_path(store_root, "spy_daily_5y", "vol_model")
        artifact.mkdir(parents=True)
        cfg = compose_leg_config(vt_leg, store_root=store_root)
        assert "vol_model" in cfg.pretrained_leaves
        assert cfg.pretrained_leaves["vol_model"] == artifact

    def test_universe_holdout_pct_overrides_strategy_default(
        self, main_spec: StudySpec, tmp_path: Path
    ) -> None:
        legs = expand_spec_into_legs(main_spec, repo_root=REPO_ROOT)
        ab_2008 = next(
            leg
            for leg in legs
            if leg.strategy == "AdaptiveBollinger" and leg.universe == "spy_daily_2008"
        )
        cfg = compose_leg_config(ab_2008, store_root=tmp_path)
        # spy_daily_2008 pins holdout_pct: 0.0 to make the GFC the eval set.
        assert cfg.validation.holdout_pct == 0.0


class TestComposeHpoConfig:
    def test_study_name_overridden_to_leg_id(self, tmp_path: Path) -> None:
        spec = load_study_spec(_write_minimal_spec(tmp_path))
        legs = expand_spec_into_legs(spec, repo_root=REPO_ROOT)
        leg = legs[0]
        hpo = compose_hpo_config(leg)
        assert hpo.study_name == leg.leg_id


class TestCollectPretrainedLeafPairs:
    def test_only_ml_bearing_strategies_contribute(self, main_spec: StudySpec) -> None:
        legs = expand_spec_into_legs(main_spec, repo_root=REPO_ROOT)
        pairs = _collect_pretrained_leaf_pairs(legs)
        assert len(pairs) == EXPECTED_MAIN_STUDY_PRETRAINED_PAIR_COUNT
        # Non-ML strategies (AB, Pairs) contribute none.
        leaf_keys = {leaf_key for _, leaf_key in pairs}
        assert leaf_keys == set(LEAF_KEY_TO_TEMPLATE_YAML.keys())


class TestComposeStandaloneModelConfig:
    def test_overrides_name_and_data(self, tmp_path: Path) -> None:
        cfg = _compose_standalone_model_config(
            template_path=REPO_ROOT / "config" / "models" / "spy_hybrid_volatility.yaml",
            universe_profile_path=REPO_ROOT / "config" / "universes" / "qqq_daily_5y.yaml",
            artifact_name="qqq_daily_5y_vol_model",
        )
        assert cfg.name == "qqq_daily_5y_vol_model"
        assert cfg.data.tickers == ["QQQ"]
        # Template defaults preserved (model name + kind).
        assert cfg.model.name == "hybrid_volatility"


class TestResolveStudyDir:
    def test_relative_output_dir_resolves_under_store_root(
        self, main_spec: StudySpec, tmp_path: Path
    ) -> None:
        store = tmp_path / "store"
        # main_study output_dir is "studies/main" (relative)
        assert resolve_study_dir(main_spec, store) == store / "studies" / "main"

    def test_absolute_output_dir_kept_as_is(self, tmp_path: Path) -> None:
        absolute = tmp_path / "abs_studies" / "x"
        payload = {
            "name": "abs_test",
            "output_dir": str(absolute),
            "legs": [
                {
                    "strategy": "AdaptiveBollinger",
                    "strategy_config": "config/strategies/adaptive_bollinger.yaml",
                    "hpo_config": "config/hpo/adaptive_bollinger.yaml",
                    "universes": ["spy_daily_5y"],
                }
            ],
        }
        spec_path = tmp_path / "abs_spec.yaml"
        spec_path.write_text(yaml.safe_dump(payload, default_flow_style=False))
        spec = load_study_spec(spec_path)
        assert resolve_study_dir(spec, tmp_path / "ignored") == absolute


class TestLegStateRoundTrip:
    def test_initial_then_with_step_completed(self) -> None:
        s = LegState.initial("X__y", "X", "y")
        s = s.with_step_completed(LEG_STEP_TUNE).with_step_completed(LEG_STEP_RUN)
        assert s.steps_completed == (LEG_STEP_TUNE, LEG_STEP_RUN)
        # Idempotent on re-add.
        assert s.with_step_completed(LEG_STEP_TUNE).steps_completed == s.steps_completed

    def test_unknown_step_in_persisted_json_raises(self) -> None:
        bad_dict = LegState.initial("X__y", "X", "y").to_dict()
        bad_dict["steps_completed"] = ["not_a_real_step"]
        with pytest.raises(ValueError, match="not_a_real_step"):
            LegState.from_dict(bad_dict)

    def test_dict_round_trip(self) -> None:
        original = LegState.initial("X__y", "X", "y").with_step_completed(LEG_STEP_TUNE)
        recovered = LegState.from_dict(original.to_dict())
        assert recovered == original


class TestStudyStateRoundTrip:
    def _fresh_state(self) -> StudyState:
        return StudyState(
            spec_name="t",
            spec_hash="abc123",
            started_at=datetime(2026, 5, 3, 12, 0, tzinfo=UTC),
            legs=(
                LegState.initial("S__a", "S", "a"),
                LegState.initial("S__b", "S", "b"),
            ),
            cross_strategy_compares_done=(),
        )

    def test_dict_round_trip(self) -> None:
        original = self._fresh_state()
        recovered = StudyState.from_dict(original.to_dict())
        assert recovered == original

    def test_with_leg_replaces_match(self) -> None:
        state = self._fresh_state()
        updated = state.legs[0].with_step_completed(LEG_STEP_TUNE)
        new_state = state.with_leg(updated)
        assert new_state.get_leg("S__a").steps_completed == (LEG_STEP_TUNE,)
        # Other leg untouched.
        assert new_state.get_leg("S__b").steps_completed == ()

    def test_with_leg_unknown_id_raises(self) -> None:
        state = self._fresh_state()
        rogue = LegState.initial("does_not_exist", "X", "y")
        with pytest.raises(KeyError, match="does_not_exist"):
            state.with_leg(rogue)

    def test_with_compare_done_idempotent(self) -> None:
        state = self._fresh_state()
        once = state.with_compare_done("a")
        twice = once.with_compare_done("a")
        assert once == twice
        assert once.cross_strategy_compares_done == ("a",)


class TestComputeSpecHash:
    def test_deterministic_on_same_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "x.yaml"
        path.write_text("a: 1\n")
        assert compute_spec_hash(path) == compute_spec_hash(path)

    def test_changes_when_bytes_change(self, tmp_path: Path) -> None:
        path = tmp_path / "x.yaml"
        path.write_text("a: 1\n")
        first = compute_spec_hash(path)
        path.write_text("a: 2\n")
        assert compute_spec_hash(path) != first


class TestWriteReadStudyState:
    def test_round_trip_via_disk(self, tmp_path: Path) -> None:
        state = StudyState(
            spec_name="t",
            spec_hash="abc",
            started_at=datetime(2026, 5, 3, tzinfo=UTC),
            legs=(LegState.initial("S__a", "S", "a"),),
            cross_strategy_compares_done=(),
        )
        path = tmp_path / STUDY_STATE_FILENAME
        write_study_state(path, state)
        assert read_study_state(path) == state

    def test_atomic_write_no_tmp_left_behind(self, tmp_path: Path) -> None:
        state = StudyState(
            spec_name="t",
            spec_hash="abc",
            started_at=datetime(2026, 5, 3, tzinfo=UTC),
            legs=(LegState.initial("S__a", "S", "a"),),
            cross_strategy_compares_done=(),
        )
        path = tmp_path / STUDY_STATE_FILENAME
        write_study_state(path, state)
        siblings = sorted(p.name for p in tmp_path.iterdir())
        assert siblings == [STUDY_STATE_FILENAME], (
            "atomic write should leave only the final file, no .tmp"
        )


class TestRunStudySpecHashGuard:
    """The orchestrator must refuse to resume against a mutated spec."""

    def test_mutated_spec_rejected(self, tmp_path: Path) -> None:
        from src.orchestration.study import run_study

        spec_path = _write_minimal_spec(tmp_path, output_dir="studies/test")
        store_root = tmp_path / "store"
        # Pre-write a state file under the resolved study dir using a
        # FAKE hash so we can exercise the guard without running compute.
        spec = load_study_spec(spec_path)
        study_dir = resolve_study_dir(spec, store_root)
        study_dir.mkdir(parents=True)
        ab_legs = tuple(
            LegState.initial(make_leg_id("AdaptiveBollinger", u), "AdaptiveBollinger", u)
            for u in ("spy_daily_5y", "qqq_daily_5y")
        )
        pairs_leg = LegState.initial(
            "PairsTrading__ivv_voo_daily_5y", "PairsTrading", "ivv_voo_daily_5y"
        )
        stale = StudyState(
            spec_name=spec.name,
            spec_hash="0" * 64,
            started_at=datetime(2026, 5, 3, tzinfo=UTC),
            legs=ab_legs + (pairs_leg,),
            cross_strategy_compares_done=(),
        )
        write_study_state(study_dir / STUDY_STATE_FILENAME, stale)
        with pytest.raises(ValueError, match="different spec"):
            run_study(spec_path, store_root=store_root)


class TestRunStudySpecSnapshot:
    """First-run side-effect: the orchestrator copies the spec for provenance."""

    def test_spec_snapshot_written(self, tmp_path: Path) -> None:
        # We exercise just the snapshot side-effect by short-circuiting via
        # only_legs filter that matches no leg — orchestrator initialises
        # state + snapshot, then has nothing to do.
        from src.orchestration.study import run_study

        spec_path = _write_minimal_spec(tmp_path, output_dir="studies/snap")
        store_root = tmp_path / "store"
        result = run_study(
            spec_path,
            store_root=store_root,
            only_legs=["__no_such_leg__"],
            skip_compares=True,
        )
        snapshot = result.study_dir / SPEC_SNAPSHOT_FILENAME
        assert snapshot.is_file()
        assert snapshot.read_text() == spec_path.read_text()
        assert result.n_legs_completed == 0
        assert result.n_legs_failed == 0
        assert result.n_legs_skipped == 3  # all 3 legs filtered out
