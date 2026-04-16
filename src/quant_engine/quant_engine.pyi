"""
C++ quantitative engine (backtesting, metrics) — Python bindings
"""

from __future__ import annotations

import typing

import numpy

import quant_engine

__all__: list[str] = [
    "BacktestEngine",
    "BacktestResult",
    "MetricsCalculator",
    "PerformanceMetrics",
    "SlippageConfig",
    "SlippageModel",
    "hello",
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
        timestamps: numpy.ndarray[numpy.int64],
        open: numpy.ndarray[numpy.float64],
        high: numpy.ndarray[numpy.float64],
        low: numpy.ndarray[numpy.float64],
        close: numpy.ndarray[numpy.float64],
        volume: numpy.ndarray[numpy.float64],
        signals: numpy.ndarray[numpy.float64],
        slippage: SlippageConfig,
    ) -> BacktestResult: ...
    def run_scenarios(
        self,
        timestamps: numpy.ndarray[numpy.int64],
        open: numpy.ndarray[numpy.float64],
        high: numpy.ndarray[numpy.float64],
        low: numpy.ndarray[numpy.float64],
        close: numpy.ndarray[numpy.float64],
        volume: numpy.ndarray[numpy.float64],
        signals: numpy.ndarray[numpy.float64],
        scenarios: list[SlippageConfig],
    ) -> list[BacktestResult]: ...

class BacktestResult:
    @property
    def annualized_return(self) -> float: ...
    @property
    def annualized_volatility(self) -> float: ...
    @property
    def equity_curve(self) -> numpy.ndarray[numpy.float64]: ...
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

class MetricsCalculator:
    @staticmethod
    def annualized_return(
        equity_curve: numpy.ndarray[numpy.float64], annualization_factor: int
    ) -> float: ...
    @staticmethod
    def annualized_volatility(
        returns: numpy.ndarray[numpy.float64], annualization_factor: int
    ) -> float: ...
    @staticmethod
    def compute(
        equity_curve: numpy.ndarray[numpy.float64],
        annualization_factor: int,
        risk_free_rate: float = 0.0,
    ) -> PerformanceMetrics: ...
    @staticmethod
    def max_drawdown(equity_curve: numpy.ndarray[numpy.float64]) -> float: ...
    @staticmethod
    def sharpe_ratio(
        returns: numpy.ndarray[numpy.float64],
        annualization_factor: int,
        risk_free_rate: float = 0.0,
    ) -> float: ...
    @staticmethod
    def sortino_ratio(
        returns: numpy.ndarray[numpy.float64],
        annualization_factor: int,
        risk_free_rate: float = 0.0,
    ) -> float: ...
    @staticmethod
    def win_rate(returns: numpy.ndarray[numpy.float64]) -> float: ...

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

def hello() -> str:
    """
    Smoke-test hook confirming the compiled C++ extension is loadable.
    """
