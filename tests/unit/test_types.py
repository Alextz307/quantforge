"""Tests for domain value types."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.core.types import (
    SLIPPAGE_SCENARIOS,
    BacktestResult,
    BarData,
    Interval,
    PairSignal,
    ScenarioComparisonResult,
    Signal,
    SlippageScenario,
    WalkForwardResult,
)


class TestInterval:
    def test_daily_annualization(self) -> None:
        assert Interval.DAILY.annualization_factor() == 252

    def test_minute_annualization(self) -> None:
        assert Interval.MINUTE.annualization_factor() == 252 * 390

    def test_weekly_annualization(self) -> None:
        assert Interval.WEEKLY.annualization_factor() == 52

    def test_all_intervals_have_factors(self) -> None:
        for interval in Interval:
            assert interval.annualization_factor() > 0

    def test_intraday_consistency(self) -> None:
        """Verify that intraday intervals are consistent with each other."""
        # 1 minute * 390 = 1 day's minutes
        assert Interval.MINUTE.annualization_factor() == Interval.DAILY.annualization_factor() * 390
        # 1 second * 60 = 1 minute
        assert Interval.SECOND.annualization_factor() == Interval.MINUTE.annualization_factor() * 60
        # 5 min = 5 * 1 min
        assert (
            Interval.FIVE_MINUTE.annualization_factor()
            == Interval.MINUTE.annualization_factor() // 5
        )


class TestBarData:
    def test_valid_bar(self) -> None:
        bar = BarData(
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1_000_000.0,
        )
        assert bar.close == 103.0

    def test_rejects_high_less_than_low(self) -> None:
        with pytest.raises(ValidationError, match="high.*must be >= low"):
            BarData(
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=95.0,
                low=99.0,
                close=97.0,
                volume=1000.0,
            )

    def test_rejects_high_less_than_open(self) -> None:
        with pytest.raises(ValidationError, match="high.*must be >= max"):
            BarData(
                timestamp=datetime(2024, 1, 1),
                open=110.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000.0,
            )

    def test_rejects_high_less_than_close(self) -> None:
        with pytest.raises(ValidationError, match="high.*must be >= max"):
            BarData(
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=106.0,
                volume=1000.0,
            )

    def test_rejects_low_greater_than_open(self) -> None:
        with pytest.raises(ValidationError, match="low.*must be <= min"):
            BarData(
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=105.0,
                low=101.0,
                close=103.0,
                volume=1000.0,
            )

    def test_rejects_negative_price(self) -> None:
        with pytest.raises(ValidationError):
            BarData(
                timestamp=datetime(2024, 1, 1),
                open=-1.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000.0,
            )

    def test_rejects_zero_price(self) -> None:
        with pytest.raises(ValidationError):
            BarData(
                timestamp=datetime(2024, 1, 1),
                open=0.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000.0,
            )

    def test_rejects_negative_volume(self) -> None:
        with pytest.raises(ValidationError):
            BarData(
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=-100.0,
            )

    def test_zero_volume_accepted(self) -> None:
        bar = BarData(
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=0.0,
        )
        assert bar.volume == 0.0

    def test_frozen(self) -> None:
        bar = BarData(
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000.0,
        )
        with pytest.raises(ValidationError):
            bar.close = 200.0

    def test_default_interval_is_daily(self) -> None:
        bar = BarData(
            timestamp=datetime(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000.0,
        )
        assert bar.interval == Interval.DAILY


class TestSignal:
    def test_valid_signal(self) -> None:
        sig = Signal(timestamp=datetime(2024, 1, 1), position=0.5)
        assert sig.position == 0.5

    def test_short_position(self) -> None:
        sig = Signal(timestamp=datetime(2024, 1, 1), position=-1.0)
        assert sig.position == -1.0

    def test_max_leverage(self) -> None:
        sig = Signal(timestamp=datetime(2024, 1, 1), position=3.0)
        assert sig.position == 3.0

    def test_rejects_position_below_minus_one(self) -> None:
        with pytest.raises(ValidationError):
            Signal(timestamp=datetime(2024, 1, 1), position=-1.5)

    def test_rejects_position_above_three(self) -> None:
        with pytest.raises(ValidationError):
            Signal(timestamp=datetime(2024, 1, 1), position=3.5)


class TestPairSignal:
    def test_valid_pair_signal(self) -> None:
        ps = PairSignal(
            timestamp=datetime(2024, 1, 1),
            leg_a_position=1.0,
            leg_b_position=-1.0,
            spread_zscore=2.1,
        )
        assert ps.spread_zscore == 2.1

    def test_rejects_excessive_leverage(self) -> None:
        with pytest.raises(ValidationError, match="leverage"):
            PairSignal(
                timestamp=datetime(2024, 1, 1),
                leg_a_position=2.0,
                leg_b_position=-2.0,
                spread_zscore=1.0,
            )

    def test_allows_max_leverage(self) -> None:
        ps = PairSignal(
            timestamp=datetime(2024, 1, 1),
            leg_a_position=1.5,
            leg_b_position=-1.5,
            spread_zscore=1.0,
        )
        assert abs(ps.leg_a_position) + abs(ps.leg_b_position) == 3.0

    def test_rejects_individual_leg_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            PairSignal(
                timestamp=datetime(2024, 1, 1),
                leg_a_position=5.0,
                leg_b_position=0.0,
                spread_zscore=1.0,
            )


class TestBacktestResult:
    def test_creation(self) -> None:
        result = BacktestResult(
            ticker="SPY",
            strategy="VolTarget",
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            total_return=0.25,
            annualized_return=0.12,
            max_drawdown=-0.10,
            win_rate=0.55,
            equity_curve=[100.0, 105.0, 110.0],
            trade_count=50,
        )
        assert result.scenario_label == "normal"
        assert result.trade_count == 50


class TestWalkForwardResult:
    def test_creation(self) -> None:
        fold = BacktestResult(
            ticker="SPY",
            strategy="VolTarget",
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            total_return=0.25,
            annualized_return=0.12,
            max_drawdown=-0.10,
            win_rate=0.55,
            equity_curve=[100.0, 110.0],
            trade_count=50,
        )
        wf = WalkForwardResult(
            ticker="SPY",
            strategy="VolTarget",
            fold_results=[fold],
            mean_sharpe=1.5,
            std_sharpe=0.3,
            mean_return=0.12,
            worst_drawdown=-0.10,
        )
        assert len(wf.fold_results) == 1


class TestSlippageScenarios:
    def test_four_predefined_scenarios(self) -> None:
        assert len(SLIPPAGE_SCENARIOS) == 4

    def test_scenario_labels(self) -> None:
        labels = {s.label for s in SLIPPAGE_SCENARIOS}
        assert labels == {"zero", "normal", "adverse", "extreme"}

    def test_zero_scenario_has_no_friction(self) -> None:
        zero = next(s for s in SLIPPAGE_SCENARIOS if s.label == "zero")
        assert zero.slippage_bps == 0.0
        assert zero.transaction_fee == 0.0

    def test_custom_scenario(self) -> None:
        custom = SlippageScenario(
            label="custom",
            slippage_bps=10.0,
            transaction_fee=0.003,
        )
        assert custom.description == ""


class TestScenarioComparisonResult:
    def test_creation(self) -> None:
        fold = BacktestResult(
            ticker="SPY",
            strategy="VolTarget",
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            total_return=0.25,
            annualized_return=0.12,
            max_drawdown=-0.10,
            win_rate=0.55,
            equity_curve=[100.0],
            trade_count=10,
        )
        wf = WalkForwardResult(
            ticker="SPY",
            strategy="VolTarget",
            fold_results=[fold],
            mean_sharpe=1.5,
            std_sharpe=0.3,
            mean_return=0.12,
            worst_drawdown=-0.10,
        )
        result = ScenarioComparisonResult(
            ticker="SPY",
            strategy="VolTarget",
            scenario_results={"normal": wf},
            alpha_decay_pct=15.0,
        )
        assert result.alpha_decay_pct == 15.0
