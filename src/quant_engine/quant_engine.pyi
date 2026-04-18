"""
C++ quantitative engine (backtesting, metrics) — Python bindings
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

import quant_engine

__all__: list[str] = [
    "BacktestEngine",
    "BacktestResult",
    "BollingerBands",
    "BollingerResult",
    "GarchParams",
    "GarmanKlass",
    "MACD",
    "MACDResult",
    "MetricsCalculator",
    "Parkinson",
    "PerformanceMetrics",
    "RSI",
    "SlippageConfig",
    "SlippageModel",
    "garch_filter",
    "hello",
    "run_mean_reversion_state_machine",
    "run_pairs_state_machine",
]

class BacktestEngine:
    def __init__(
        self,
        *,
        initial_capital: float = 10000.0,
        transaction_fee_rate: float = 0.001,
        allow_short: bool = True,
    ) -> None: ...
    def run(
        self,
        timestamps: numpy.typing.NDArray[numpy.int64],
        open: numpy.typing.NDArray[numpy.float64],
        high: numpy.typing.NDArray[numpy.float64],
        low: numpy.typing.NDArray[numpy.float64],
        close: numpy.typing.NDArray[numpy.float64],
        volume: numpy.typing.NDArray[numpy.float64],
        signals: numpy.typing.NDArray[numpy.float64],
        slippage: SlippageConfig,
    ) -> BacktestResult: ...
    def run_scenarios(
        self,
        timestamps: numpy.typing.NDArray[numpy.int64],
        open: numpy.typing.NDArray[numpy.float64],
        high: numpy.typing.NDArray[numpy.float64],
        low: numpy.typing.NDArray[numpy.float64],
        close: numpy.typing.NDArray[numpy.float64],
        volume: numpy.typing.NDArray[numpy.float64],
        signals: numpy.typing.NDArray[numpy.float64],
        scenarios: list[SlippageConfig],
    ) -> list[BacktestResult]: ...

class BacktestResult:
    @property
    def annualized_return(self) -> float: ...
    @property
    def annualized_volatility(self) -> float: ...
    @property
    def equity_curve(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def max_drawdown(self) -> float: ...
    @property
    def scenario_label(self) -> str: ...
    @property
    def sharpe_ratio(self) -> float: ...
    @property
    def sortino_ratio(self) -> float: ...
    @property
    def total_return(self) -> float: ...
    @property
    def trade_count(self) -> int: ...
    @property
    def win_rate(self) -> float: ...

class BollingerBands:
    def __init__(self, period: int = 20, num_std: float = 2.0) -> None: ...
    def compute(
        self, prices: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def compute_all(self, prices: numpy.typing.NDArray[numpy.float64]) -> BollingerResult: ...
    @property
    def name(self) -> str: ...
    @property
    def warmup_period(self) -> int: ...

class BollingerResult:
    @property
    def lower(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def mid(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def upper(self) -> numpy.typing.NDArray[numpy.float64]: ...

class GarchParams:
    def __init__(
        self, *, omega: float, alpha: list[float], beta: list[float], mu: float, backcast: float
    ) -> None: ...
    @property
    def alpha(self) -> list[float]: ...
    @property
    def backcast(self) -> float: ...
    @property
    def beta(self) -> list[float]: ...
    @property
    def mu(self) -> float: ...
    @property
    def omega(self) -> float: ...

class GarmanKlass:
    def __init__(self, window: int = 22) -> None: ...
    def compute(
        self,
        open: numpy.typing.NDArray[numpy.float64],
        high: numpy.typing.NDArray[numpy.float64],
        low: numpy.typing.NDArray[numpy.float64],
        close: numpy.typing.NDArray[numpy.float64],
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def name(self) -> str: ...
    @property
    def warmup_period(self) -> int: ...

class MACD:
    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> None: ...
    def compute(
        self, prices: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def compute_all(self, prices: numpy.typing.NDArray[numpy.float64]) -> MACDResult: ...
    @property
    def name(self) -> str: ...
    @property
    def warmup_period(self) -> int: ...

class MACDResult:
    @property
    def histogram(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def macd_line(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def signal_line(self) -> numpy.typing.NDArray[numpy.float64]: ...

class MetricsCalculator:
    @staticmethod
    def annualized_return(
        equity_curve: numpy.typing.NDArray[numpy.float64], annualization_factor: int
    ) -> float: ...
    @staticmethod
    def annualized_volatility(
        returns: numpy.typing.NDArray[numpy.float64], annualization_factor: int
    ) -> float: ...
    @staticmethod
    def compute(
        equity_curve: numpy.typing.NDArray[numpy.float64],
        annualization_factor: int,
        risk_free_rate: float = 0.0,
    ) -> PerformanceMetrics: ...
    @staticmethod
    def max_drawdown(equity_curve: numpy.typing.NDArray[numpy.float64]) -> float: ...
    @staticmethod
    def sharpe_ratio(
        returns: numpy.typing.NDArray[numpy.float64],
        annualization_factor: int,
        risk_free_rate: float = 0.0,
    ) -> float: ...
    @staticmethod
    def sortino_ratio(
        returns: numpy.typing.NDArray[numpy.float64],
        annualization_factor: int,
        risk_free_rate: float = 0.0,
    ) -> float: ...
    @staticmethod
    def win_rate(returns: numpy.typing.NDArray[numpy.float64]) -> float: ...

class Parkinson:
    def __init__(self, window: int = 22) -> None: ...
    def compute(
        self,
        open: numpy.typing.NDArray[numpy.float64],
        high: numpy.typing.NDArray[numpy.float64],
        low: numpy.typing.NDArray[numpy.float64],
        close: numpy.typing.NDArray[numpy.float64],
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def name(self) -> str: ...
    @property
    def warmup_period(self) -> int: ...

class PerformanceMetrics:
    @property
    def annualized_return(self) -> float: ...
    @property
    def annualized_volatility(self) -> float: ...
    @property
    def calmar_ratio(self) -> float: ...
    @property
    def max_drawdown(self) -> float: ...
    @property
    def sharpe_ratio(self) -> float: ...
    @property
    def sortino_ratio(self) -> float: ...
    @property
    def win_rate(self) -> float: ...

class RSI:
    def __init__(self, period: int = 14) -> None: ...
    def compute(
        self, prices: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def name(self) -> str: ...
    @property
    def warmup_period(self) -> int: ...

class SlippageConfig:
    base_bps: float
    model: SlippageModel
    volume_impact_coeff: float
    def __init__(
        self,
        *,
        model: SlippageModel = quant_engine.SlippageModel.Fixed,
        base_bps: float = 1.0,
        volume_impact_coeff: float = 0.0,
    ) -> None: ...

class SlippageModel:
    """
    Members:

      NoSlippage

      Fixed

      VolumeScaled
    """

    Fixed: typing.ClassVar[SlippageModel]  # value = <SlippageModel.Fixed: 1>
    NoSlippage: typing.ClassVar[SlippageModel]  # value = <SlippageModel.NoSlippage: 0>
    VolumeScaled: typing.ClassVar[SlippageModel]  # value = <SlippageModel.VolumeScaled: 2>
    __members__: typing.ClassVar[dict[str, SlippageModel]]  # noqa: E501
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

def garch_filter(
    scaled_returns: numpy.typing.NDArray[numpy.float64], params: GarchParams
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Run the GARCH(p,q) recursion; returns conditional variances.
    """

def hello() -> str:
    """
    Smoke-test hook confirming the compiled C++ extension is loadable.
    """

def run_mean_reversion_state_machine(
    close: numpy.typing.NDArray[numpy.float64],
    mid: numpy.typing.NDArray[numpy.float64],
    upper: numpy.typing.NDArray[numpy.float64],
    lower: numpy.typing.NDArray[numpy.float64],
    trend_ma: numpy.typing.NDArray[numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Run the AdaptiveBollinger state machine; returns a position series.
    """

def run_pairs_state_machine(
    zscore: numpy.typing.NDArray[numpy.float64],
    entry_zscore: float,
    exit_zscore: float,
    stop_loss_zscore: float,
) -> numpy.typing.NDArray[numpy.float64]:
    """
    Run the pairs-trading state machine; returns a position series.
    """
