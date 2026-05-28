"""
Predefined slippage scenarios for robustness sweeps.

Values are in basis points (1 bp = 0.01%). ``ZERO`` is the friction-free
upper bound; ``NORMAL`` matches typical retail SPY conditions; ``HIGH``
and ``EXTREME`` model adverse / crisis conditions for stress testing.

Recalibrate the bp values without recompiling — they live in Python on
purpose so the C++ engine stays generic.
"""

from __future__ import annotations

from enum import StrEnum

from quant_engine import SlippageConfig, SlippageModel


class SlippageScenario(StrEnum):
    """
    Named lookup keys for ``SLIPPAGE_SCENARIOS``.
    """

    ZERO = "zero"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


SLIPPAGE_SCENARIOS: dict[SlippageScenario, SlippageConfig] = {
    SlippageScenario.ZERO: SlippageConfig(
        model=SlippageModel.NoSlippage,
        base_bps=0.0,
        volume_impact_coeff=0.0,
    ),
    SlippageScenario.NORMAL: SlippageConfig(
        model=SlippageModel.Fixed,
        base_bps=1.0,
        volume_impact_coeff=0.0,
    ),
    SlippageScenario.HIGH: SlippageConfig(
        model=SlippageModel.Fixed,
        base_bps=5.0,
        volume_impact_coeff=0.0,
    ),
    SlippageScenario.EXTREME: SlippageConfig(
        model=SlippageModel.Fixed,
        base_bps=10.0,
        volume_impact_coeff=0.0,
    ),
}
