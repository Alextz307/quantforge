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

import pandas as pd
import pytest

from src.core.types import Interval
from src.data.live_fetcher import (
    DailyLiveBarFetcher,
    _drop_unclosed_last_session,
    resolve_fetcher,
)

_THURSDAY = "2026-05-28"
_FRIDAY = "2026-05-29"
# NYSE closes 16:00 ET = 20:00 UTC on 2026-05-29 (EDT). Straddle that instant.
_FRIDAY_BEFORE_CLOSE = pd.Timestamp("2026-05-29T17:00:00Z")
_FRIDAY_AFTER_CLOSE = pd.Timestamp("2026-05-29T21:00:00Z")
_SATURDAY_MIDDAY = pd.Timestamp("2026-05-30T12:00:00Z")


def _two_bar_frame(dates: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame({"close": [1.0, 2.0]}, index=pd.DatetimeIndex(list(dates)))


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


def test_drop_unclosed_last_session_drops_forming_bar() -> None:
    """
    During Friday's open session the still-forming Friday bar is dropped, so
    the latest complete bar is Thursday — the bar acted on at Friday's open.
    """

    bars = _two_bar_frame((_THURSDAY, _FRIDAY))
    kept = _drop_unclosed_last_session(bars, _FRIDAY_BEFORE_CLOSE)

    assert len(kept) == 1
    assert kept.index[-1] == pd.Timestamp(_THURSDAY)


def test_drop_unclosed_last_session_keeps_closed_bar() -> None:
    """
    After Friday's close the Friday bar is final and survives — it becomes
    the bar the next signal (for Monday's open) is computed from.
    """

    bars = _two_bar_frame((_THURSDAY, _FRIDAY))
    kept = _drop_unclosed_last_session(bars, _FRIDAY_AFTER_CLOSE)

    assert len(kept) == 2
    assert kept.index[-1] == pd.Timestamp(_FRIDAY)


def test_drop_unclosed_last_session_keeps_past_session_over_weekend() -> None:
    """
    Loaded on Saturday, Friday's session has long closed — nothing is dropped.
    """

    bars = _two_bar_frame((_THURSDAY, _FRIDAY))
    kept = _drop_unclosed_last_session(bars, _SATURDAY_MIDDAY)

    assert len(kept) == 2


def test_drop_unclosed_last_session_empty_frame_is_noop() -> None:
    empty = pd.DataFrame({"close": []}, index=pd.DatetimeIndex([]))
    assert _drop_unclosed_last_session(empty, _FRIDAY_BEFORE_CLOSE).empty
