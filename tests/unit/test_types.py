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

# Annualization expectations (kept hardcoded — these tests verify the function returns these
# specific numbers, so importing them from the source would be circular)
EXPECTED_DAILY_FACTOR = 252
EXPECTED_MINUTES_PER_TRADING_DAY = 390
EXPECTED_SECONDS_PER_MINUTE = 60
EXPECTED_FIVE_MIN_DIVISOR = 5
EXPECTED_WEEKLY_FACTOR = 52

# Reference OHLCV ladder used across BarData tests (high >= open & close, low <= open & close)
SAMPLE_TIMESTAMP = datetime(2024, 1, 1)
VALID_OPEN = 100.0
VALID_HIGH = 105.0
VALID_LOW = 99.0
VALID_CLOSE = 103.0
VALID_VOLUME = 1_000_000.0
SMALL_VOLUME = 1000.0
INVALID_HIGH_BELOW_LOW = 95.0  # < VALID_LOW
INVALID_OPEN_ABOVE_HIGH = 110.0  # > VALID_HIGH
INVALID_CLOSE_ABOVE_HIGH = 106.0  # > VALID_HIGH
INVALID_LOW_ABOVE_OPEN = 101.0  # > VALID_OPEN
INVALID_NEGATIVE_PRICE = -1.0
INVALID_ZERO_PRICE = 0.0
INVALID_NEGATIVE_VOLUME = -100.0

# Signal test values
SAMPLE_LONG_POSITION = 0.5
SAMPLE_SHORT_POSITION = -1.0
MAX_ALLOWED_POSITION = 3.0
INVALID_POSITION_BELOW_MIN = -1.5
INVALID_POSITION_ABOVE_MAX = 3.5

# PairSignal test values
PAIR_LEG_A_LONG = 1.0
PAIR_LEG_B_SHORT = -1.0
PAIR_SAMPLE_ZSCORE = 2.1
PAIR_NEUTRAL_ZSCORE = 1.0
PAIR_LEG_OVERSIZED = 2.0  # |a| + |b| = 4 > 3 max combined leverage
PAIR_LEG_AT_MAX = 1.5  # |a| + |b| = 3.0 (boundary)
PAIR_INVALID_LEG_VALUE = 5.0  # individual leg out of [-3, 3]
PAIR_MAX_COMBINED_LEVERAGE = 3.0

# BacktestResult sample
SAMPLE_TICKER = "SPY"
SAMPLE_STRATEGY = "VolTarget"
SAMPLE_SHARPE = 1.5
SAMPLE_SORTINO = 2.0
SAMPLE_TOTAL_RETURN = 0.25
SAMPLE_ANNUAL_RETURN = 0.12
SAMPLE_MAX_DRAWDOWN = -0.10
SAMPLE_WIN_RATE = 0.55
SAMPLE_EQUITY_CURVE = [100.0, 105.0, 110.0]
SHORT_EQUITY_CURVE = [100.0, 110.0]
SINGLE_POINT_EQUITY = [100.0]
SAMPLE_TRADE_COUNT = 50
SMALL_TRADE_COUNT = 10
SAMPLE_STD_SHARPE = 0.3
SAMPLE_ALPHA_DECAY_PCT = 15.0

# SlippageScenario expectations
PREDEFINED_SCENARIO_COUNT = 4
PREDEFINED_SCENARIO_LABELS = {"zero", "normal", "adverse", "extreme"}
ZERO_SCENARIO_LABEL = "zero"
CUSTOM_SCENARIO_LABEL = "custom"
CUSTOM_SLIPPAGE_BPS = 10.0
CUSTOM_TRANSACTION_FEE = 0.003


def _valid_bar(**overrides: object) -> BarData:
    """Build a BarData with valid OHLCV defaults, overridable per-test."""
    fields: dict[str, object] = {
        "timestamp": SAMPLE_TIMESTAMP,
        "open": VALID_OPEN,
        "high": VALID_HIGH,
        "low": VALID_LOW,
        "close": VALID_CLOSE,
        "volume": VALID_VOLUME,
    }
    fields.update(overrides)
    return BarData(**fields)  # type: ignore[arg-type]


class TestInterval:
    def test_daily_annualization(self) -> None:
        assert Interval.DAILY.annualization_factor() == EXPECTED_DAILY_FACTOR

    def test_minute_annualization(self) -> None:
        assert (
            Interval.MINUTE.annualization_factor()
            == EXPECTED_DAILY_FACTOR * EXPECTED_MINUTES_PER_TRADING_DAY
        )

    def test_weekly_annualization(self) -> None:
        assert Interval.WEEKLY.annualization_factor() == EXPECTED_WEEKLY_FACTOR

    def test_all_intervals_have_factors(self) -> None:
        for interval in Interval:
            assert interval.annualization_factor() > 0

    def test_intraday_consistency(self) -> None:
        """Verify that intraday intervals are consistent with each other."""
        # 1 minute * EXPECTED_MINUTES_PER_TRADING_DAY = 1 day's minutes
        assert (
            Interval.MINUTE.annualization_factor()
            == Interval.DAILY.annualization_factor() * EXPECTED_MINUTES_PER_TRADING_DAY
        )
        # 1 second * 60 = 1 minute
        assert (
            Interval.SECOND.annualization_factor()
            == Interval.MINUTE.annualization_factor() * EXPECTED_SECONDS_PER_MINUTE
        )
        # 5 min = 5 * 1 min
        assert (
            Interval.FIVE_MINUTE.annualization_factor()
            == Interval.MINUTE.annualization_factor() // EXPECTED_FIVE_MIN_DIVISOR
        )


class TestBarData:
    def test_valid_bar(self) -> None:
        bar = _valid_bar()
        assert bar.close == VALID_CLOSE

    def test_rejects_high_less_than_low(self) -> None:
        with pytest.raises(ValidationError, match="high.*must be >= low"):
            _valid_bar(high=INVALID_HIGH_BELOW_LOW, close=97.0, volume=SMALL_VOLUME)

    def test_rejects_high_less_than_open(self) -> None:
        with pytest.raises(ValidationError, match="high.*must be >= max"):
            _valid_bar(open=INVALID_OPEN_ABOVE_HIGH, volume=SMALL_VOLUME)

    def test_rejects_high_less_than_close(self) -> None:
        with pytest.raises(ValidationError, match="high.*must be >= max"):
            _valid_bar(close=INVALID_CLOSE_ABOVE_HIGH, volume=SMALL_VOLUME)

    def test_rejects_low_greater_than_open(self) -> None:
        with pytest.raises(ValidationError, match="low.*must be <= min"):
            _valid_bar(low=INVALID_LOW_ABOVE_OPEN, volume=SMALL_VOLUME)

    def test_rejects_negative_price(self) -> None:
        with pytest.raises(ValidationError):
            _valid_bar(open=INVALID_NEGATIVE_PRICE, volume=SMALL_VOLUME)

    def test_rejects_zero_price(self) -> None:
        with pytest.raises(ValidationError):
            _valid_bar(open=INVALID_ZERO_PRICE, volume=SMALL_VOLUME)

    def test_rejects_negative_volume(self) -> None:
        with pytest.raises(ValidationError):
            _valid_bar(volume=INVALID_NEGATIVE_VOLUME)

    def test_zero_volume_accepted(self) -> None:
        bar = _valid_bar(volume=0.0)
        assert bar.volume == 0.0

    def test_frozen(self) -> None:
        bar = _valid_bar(volume=SMALL_VOLUME)
        with pytest.raises(ValidationError):
            bar.close = 200.0

    def test_default_interval_is_daily(self) -> None:
        bar = _valid_bar(volume=SMALL_VOLUME)
        assert bar.interval == Interval.DAILY


class TestSignal:
    def test_valid_signal(self) -> None:
        sig = Signal(timestamp=SAMPLE_TIMESTAMP, position=SAMPLE_LONG_POSITION)
        assert sig.position == SAMPLE_LONG_POSITION

    def test_short_position(self) -> None:
        sig = Signal(timestamp=SAMPLE_TIMESTAMP, position=SAMPLE_SHORT_POSITION)
        assert sig.position == SAMPLE_SHORT_POSITION

    def test_max_leverage(self) -> None:
        sig = Signal(timestamp=SAMPLE_TIMESTAMP, position=MAX_ALLOWED_POSITION)
        assert sig.position == MAX_ALLOWED_POSITION

    def test_rejects_position_below_minus_one(self) -> None:
        with pytest.raises(ValidationError):
            Signal(timestamp=SAMPLE_TIMESTAMP, position=INVALID_POSITION_BELOW_MIN)

    def test_rejects_position_above_three(self) -> None:
        with pytest.raises(ValidationError):
            Signal(timestamp=SAMPLE_TIMESTAMP, position=INVALID_POSITION_ABOVE_MAX)


class TestPairSignal:
    def test_valid_pair_signal(self) -> None:
        ps = PairSignal(
            timestamp=SAMPLE_TIMESTAMP,
            leg_a_position=PAIR_LEG_A_LONG,
            leg_b_position=PAIR_LEG_B_SHORT,
            spread_zscore=PAIR_SAMPLE_ZSCORE,
        )
        assert ps.spread_zscore == PAIR_SAMPLE_ZSCORE

    def test_rejects_excessive_leverage(self) -> None:
        with pytest.raises(ValidationError, match="leverage"):
            PairSignal(
                timestamp=SAMPLE_TIMESTAMP,
                leg_a_position=PAIR_LEG_OVERSIZED,
                leg_b_position=-PAIR_LEG_OVERSIZED,
                spread_zscore=PAIR_NEUTRAL_ZSCORE,
            )

    def test_allows_max_leverage(self) -> None:
        ps = PairSignal(
            timestamp=SAMPLE_TIMESTAMP,
            leg_a_position=PAIR_LEG_AT_MAX,
            leg_b_position=-PAIR_LEG_AT_MAX,
            spread_zscore=PAIR_NEUTRAL_ZSCORE,
        )
        assert abs(ps.leg_a_position) + abs(ps.leg_b_position) == PAIR_MAX_COMBINED_LEVERAGE

    def test_rejects_individual_leg_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            PairSignal(
                timestamp=SAMPLE_TIMESTAMP,
                leg_a_position=PAIR_INVALID_LEG_VALUE,
                leg_b_position=0.0,
                spread_zscore=PAIR_NEUTRAL_ZSCORE,
            )


def _sample_backtest_result(
    *, equity_curve: list[float] | None = None, trade_count: int = SAMPLE_TRADE_COUNT
) -> BacktestResult:
    return BacktestResult(
        ticker=SAMPLE_TICKER,
        strategy=SAMPLE_STRATEGY,
        sharpe_ratio=SAMPLE_SHARPE,
        sortino_ratio=SAMPLE_SORTINO,
        total_return=SAMPLE_TOTAL_RETURN,
        annualized_return=SAMPLE_ANNUAL_RETURN,
        max_drawdown=SAMPLE_MAX_DRAWDOWN,
        win_rate=SAMPLE_WIN_RATE,
        equity_curve=equity_curve if equity_curve is not None else SAMPLE_EQUITY_CURVE,
        trade_count=trade_count,
    )


class TestBacktestResult:
    def test_creation(self) -> None:
        result = _sample_backtest_result()
        assert result.scenario_label == "normal"
        assert result.trade_count == SAMPLE_TRADE_COUNT


class TestWalkForwardResult:
    def test_creation(self) -> None:
        fold = _sample_backtest_result(equity_curve=SHORT_EQUITY_CURVE)
        wf = WalkForwardResult(
            ticker=SAMPLE_TICKER,
            strategy=SAMPLE_STRATEGY,
            fold_results=[fold],
            mean_sharpe=SAMPLE_SHARPE,
            std_sharpe=SAMPLE_STD_SHARPE,
            mean_return=SAMPLE_ANNUAL_RETURN,
            worst_drawdown=SAMPLE_MAX_DRAWDOWN,
        )
        assert len(wf.fold_results) == 1


class TestSlippageScenarios:
    def test_four_predefined_scenarios(self) -> None:
        assert len(SLIPPAGE_SCENARIOS) == PREDEFINED_SCENARIO_COUNT

    def test_scenario_labels(self) -> None:
        labels = {s.label for s in SLIPPAGE_SCENARIOS}
        assert labels == PREDEFINED_SCENARIO_LABELS

    def test_zero_scenario_has_no_friction(self) -> None:
        zero = next(s for s in SLIPPAGE_SCENARIOS if s.label == ZERO_SCENARIO_LABEL)
        assert zero.slippage_bps == 0.0
        assert zero.transaction_fee == 0.0

    def test_custom_scenario(self) -> None:
        custom = SlippageScenario(
            label=CUSTOM_SCENARIO_LABEL,
            slippage_bps=CUSTOM_SLIPPAGE_BPS,
            transaction_fee=CUSTOM_TRANSACTION_FEE,
        )
        assert custom.description == ""


class TestScenarioComparisonResult:
    def test_creation(self) -> None:
        fold = _sample_backtest_result(
            equity_curve=SINGLE_POINT_EQUITY, trade_count=SMALL_TRADE_COUNT
        )
        wf = WalkForwardResult(
            ticker=SAMPLE_TICKER,
            strategy=SAMPLE_STRATEGY,
            fold_results=[fold],
            mean_sharpe=SAMPLE_SHARPE,
            std_sharpe=SAMPLE_STD_SHARPE,
            mean_return=SAMPLE_ANNUAL_RETURN,
            worst_drawdown=SAMPLE_MAX_DRAWDOWN,
        )
        result = ScenarioComparisonResult(
            ticker=SAMPLE_TICKER,
            strategy=SAMPLE_STRATEGY,
            scenario_results={"normal": wf},
            alpha_decay_pct=SAMPLE_ALPHA_DECAY_PCT,
        )
        assert result.alpha_decay_pct == SAMPLE_ALPHA_DECAY_PCT
