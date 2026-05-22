"""Domain value types for the quant trading framework.

All models use Pydantic v2 with frozen=True for immutability.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.constants import (
    MAX_LEVERAGE,
    MAX_POSITION,
    MIN_POSITION,
    TRADING_DAYS_PER_YEAR,
    TRADING_WEEKS_PER_YEAR,
    US_TRADING_MINUTES_PER_DAY,
    US_TRADING_SECONDS_PER_DAY,
)


class Interval(StrEnum):
    """Bar timeframe — used for annualization factor computation."""

    SECOND = "second"
    MINUTE = "minute"
    FIVE_MINUTE = "five_minute"
    FIFTEEN_MINUTE = "fifteen_minute"
    HOUR = "hour"
    DAILY = "daily"
    WEEKLY = "weekly"

    def annualization_factor(self) -> int:
        """Return the number of bars per year for this interval."""
        return _ANNUALIZATION_FACTORS[self]


class LossFunction(StrEnum):
    """Loss functions for neural network training."""

    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"


class InformationCriterion(StrEnum):
    """Model selection criteria for statistical models."""

    AIC = "aic"
    BIC = "bic"


class Device(StrEnum):
    """Compute backends for ML model training."""

    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


_ANNUALIZATION_FACTORS: dict[Interval, int] = {
    Interval.SECOND: TRADING_DAYS_PER_YEAR * US_TRADING_SECONDS_PER_DAY,
    Interval.MINUTE: TRADING_DAYS_PER_YEAR * US_TRADING_MINUTES_PER_DAY,
    Interval.FIVE_MINUTE: TRADING_DAYS_PER_YEAR * (US_TRADING_MINUTES_PER_DAY // 5),
    Interval.FIFTEEN_MINUTE: TRADING_DAYS_PER_YEAR * (US_TRADING_MINUTES_PER_DAY // 15),
    Interval.HOUR: TRADING_DAYS_PER_YEAR * 7,
    Interval.DAILY: TRADING_DAYS_PER_YEAR,
    Interval.WEEKLY: TRADING_WEEKS_PER_YEAR,
}


class BarData(BaseModel):
    """Immutable OHLCV bar with validation."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    interval: Interval = Interval.DAILY

    @model_validator(mode="after")
    def validate_hloc_ordering(self) -> Self:
        """Ensure high >= low and high/low bound open/close."""
        if self.high < self.low:
            raise ValueError(
                f"high ({self.high}) must be >= low ({self.low}); fix by checking "
                f"the upstream OHLC source for swapped or corrupted columns."
            )
        oc_max = max(self.open, self.close)
        if self.high < oc_max:
            raise ValueError(
                f"high ({self.high}) must be >= max(open, close) ({oc_max}); fix "
                f"by checking the upstream OHLC source for swapped or corrupted columns."
            )
        oc_min = min(self.open, self.close)
        if self.low > oc_min:
            raise ValueError(
                f"low ({self.low}) must be <= min(open, close) ({oc_min}); fix by "
                f"checking the upstream OHLC source for swapped or corrupted columns."
            )
        return self


class Signal(BaseModel):
    """Strategy output for a single bar."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    position: float = Field(ge=MIN_POSITION, le=MAX_POSITION)


class PairSignal(BaseModel):
    """Pairs trading signal for two legs."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    leg_a_position: float = Field(ge=-3.0, le=3.0)
    leg_b_position: float = Field(ge=-3.0, le=3.0)
    spread_zscore: float

    @model_validator(mode="after")
    def validate_total_leverage(self) -> Self:
        """Ensure combined absolute leverage does not exceed maximum."""
        total_abs = abs(self.leg_a_position) + abs(self.leg_b_position)
        if total_abs > MAX_LEVERAGE:
            raise ValueError(
                f"Total absolute leverage ({total_abs}) exceeds maximum "
                f"({MAX_LEVERAGE}); fix by scaling leg_a_position / leg_b_position "
                f"so |a| + |b| <= {MAX_LEVERAGE}."
            )
        return self
