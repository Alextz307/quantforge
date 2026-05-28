"""
Behaviour of :class:`LiveBarFetcher` implementations + the dispatcher.

The daily fetcher is a thin wrapper around :class:`YFinanceSource`;
this test pins the interval guard + the dispatcher's
``NotImplementedError`` on cadences that have no shipped implementation.
A network-dependent end-to-end fetch is out of scope here — the
integration test exercises the full path via a stub fetcher.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.core.types import Interval
from src.data.live_fetcher import DailyLiveBarFetcher, resolve_fetcher


def test_resolve_fetcher_daily_returns_daily_impl() -> None:
    fetcher = resolve_fetcher(Interval.DAILY)
    assert isinstance(fetcher, DailyLiveBarFetcher)


@pytest.mark.parametrize(
    "interval",
    [
        Interval.HOUR,
        Interval.MINUTE,
        Interval.FIVE_MINUTE,
        Interval.FIFTEEN_MINUTE,
    ],
)
def test_resolve_fetcher_non_daily_raises(interval: Interval) -> None:
    with pytest.raises(NotImplementedError, match="no LiveBarFetcher"):
        resolve_fetcher(interval)


def test_daily_fetcher_rejects_non_daily() -> None:
    """
    Defence-in-depth: even with a hand-built daily fetcher, asking for
    an hourly bar surfaces the cadence mismatch before any vendor call.
    """

    fetcher = DailyLiveBarFetcher()
    with pytest.raises(ValueError, match="only supports Interval.DAILY"):
        fetcher.fetch(
            "SPY",
            datetime(2026, 1, 1),
            datetime(2026, 1, 31),
            Interval.HOUR,
        )
