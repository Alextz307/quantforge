"""
Predefined cost scenarios (slippage + commission) for the backtest engine.

Each tier bundles a per-fill **slippage** profile with a per-turnover
**commission** rate, both in basis points (1 bp = 0.01%):

    tier      slippage   commission
    zero        0 bp        0 bp     friction-free upper bound
    low         1 bp        1 bp
    normal      2 bp        2 bp     default — realistic liquid-ETF retail cost
    high        5 bp        5 bp     adverse / stress

Slippage moves the fill price adversely by ``base_bps``; commission is
charged on ``|Δnotional|`` traded at each rebalance (the engine's
``transaction_fee_rate`` = ``commission_bps / BPS_PER_UNIT``).

Values live in Python (not the C++ engine) so they recalibrate without a
recompile — and so the deployment signal-evaluation scorecard reuses the
exact same tiers as the backtest, from this single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from quant_engine import SlippageConfig, SlippageModel

BPS_PER_UNIT = 10000.0


class SlippageScenario(StrEnum):
    """
    Named cost tiers — keys into ``COST_SCENARIOS`` / ``SLIPPAGE_SCENARIOS``.
    """

    ZERO = "zero"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass(frozen=True)
class CostScenario:
    """
    One cost tier: per-fill slippage plus per-turnover commission.

    ``commission_bps`` is charged on ``|Δnotional|`` at each rebalance;
    ``slippage`` moves the fill price adversely by its ``base_bps``.
    """

    slippage: SlippageConfig
    commission_bps: float


def _slippage(bps: float) -> SlippageConfig:
    model = SlippageModel.NoSlippage if bps == 0.0 else SlippageModel.Fixed
    return SlippageConfig(model=model, base_bps=bps, volume_impact_coeff=0.0)


COST_SCENARIOS: dict[SlippageScenario, CostScenario] = {
    SlippageScenario.ZERO: CostScenario(_slippage(0.0), 0.0),
    SlippageScenario.LOW: CostScenario(_slippage(1.0), 1.0),
    SlippageScenario.NORMAL: CostScenario(_slippage(2.0), 2.0),
    SlippageScenario.HIGH: CostScenario(_slippage(5.0), 5.0),
}

# Slippage-only view. Derived from COST_SCENARIOS so the two cannot drift.
SLIPPAGE_SCENARIOS: dict[SlippageScenario, SlippageConfig] = {
    scenario: cost.slippage for scenario, cost in COST_SCENARIOS.items()
}


def commission_bps_for(scenario: SlippageScenario) -> float:
    """
    Commission in basis points (on ``|Δnotional|``) for a cost tier.
    """

    return COST_SCENARIOS[scenario].commission_bps


def commission_fraction_for(scenario: SlippageScenario) -> float:
    """
    Commission as a notional fraction — the engine's ``transaction_fee_rate``.
    """

    return COST_SCENARIOS[scenario].commission_bps / BPS_PER_UNIT


def total_cost_fraction_for(scenario: SlippageScenario) -> float:
    """
    Round-turn friction (slippage + commission) as a notional fraction.

    The per-unit-turnover cost the signal-evaluation scorecard charges on
    ``|Δleverage|`` at each rebalance — slippage moves the fill and
    commission is paid on the traded notional, bundled into one fraction so
    the scorecard's net figures share the engine's cost tiers exactly.
    """

    cost = COST_SCENARIOS[scenario]
    return (cost.slippage.base_bps + cost.commission_bps) / BPS_PER_UNIT
