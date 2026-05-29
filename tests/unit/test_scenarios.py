"""
Pin the cost-tier values (slippage + commission) and their invariants.

These tiers are the single source of truth shared by the backtest engine
and the deployment signal-evaluation scorecard, so a value drift here
silently changes both. The test fixes the four tiers and the derived
slippage-only view.
"""

from __future__ import annotations

import pytest

from quant_engine import SlippageModel
from src.engine.scenarios import (
    BPS_PER_UNIT,
    COST_SCENARIOS,
    SLIPPAGE_SCENARIOS,
    SlippageScenario,
    commission_bps_for,
    commission_fraction_for,
)

# (slippage_bps, commission_bps) per tier — the contract.
_EXPECTED: dict[SlippageScenario, tuple[float, float]] = {
    SlippageScenario.ZERO: (0.0, 0.0),
    SlippageScenario.LOW: (1.0, 1.0),
    SlippageScenario.NORMAL: (2.0, 2.0),
    SlippageScenario.HIGH: (5.0, 5.0),
}


def test_enum_has_exactly_the_four_tiers() -> None:
    assert set(SlippageScenario) == set(_EXPECTED)
    assert SlippageScenario.NORMAL.value == "normal"  # the default tier


@pytest.mark.parametrize(("scenario", "expected"), list(_EXPECTED.items()))
def test_tier_values(scenario: SlippageScenario, expected: tuple[float, float]) -> None:
    slippage_bps, commission_bps = expected
    cost = COST_SCENARIOS[scenario]

    assert cost.slippage.base_bps == pytest.approx(slippage_bps)
    assert cost.slippage.volume_impact_coeff == pytest.approx(0.0)
    assert commission_bps_for(scenario) == pytest.approx(commission_bps)
    # zero slippage uses the dedicated NoSlippage model; the rest are Fixed
    expected_model = (
        SlippageModel.NoSlippage if slippage_bps == 0.0 else SlippageModel.Fixed
    )
    assert cost.slippage.model == expected_model


@pytest.mark.parametrize(("scenario", "expected"), list(_EXPECTED.items()))
def test_commission_fraction_is_bps_over_unit(
    scenario: SlippageScenario, expected: tuple[float, float]
) -> None:
    _, commission_bps = expected
    assert commission_fraction_for(scenario) == pytest.approx(commission_bps / BPS_PER_UNIT)


def test_slippage_view_is_derived_from_cost_scenarios() -> None:
    assert set(SLIPPAGE_SCENARIOS) == set(COST_SCENARIOS)
    for scenario, cost in COST_SCENARIOS.items():
        assert SLIPPAGE_SCENARIOS[scenario] is cost.slippage
