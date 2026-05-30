"""
Behaviour of :class:`LiveBarFetcher` implementations + the dispatcher.

The daily fetcher is a thin wrapper around :class:`YFinanceSource`;
this test pins the interval guard + the dispatcher's
``NotImplementedError`` on cadences that have no shipped implementation.
A network-dependent end-to-end fetch is out of scope here - the
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
    _opens_of_opened_sessions,
    fetch_session_opens,
    resolve_fetcher,
)
from src.data.loader import YFinanceSource

_THURSDAY = "2026-05-28"
_FRIDAY = "2026-05-29"
# Far-past real trading days: their sessions have closed regardless of the
# wall-clock now the fetchers read, so the freshness filters are deterministic.
_PAST_THU = "2020-01-02"
_PAST_FRI = "2020-01-03"
_PAST_MON_MIDDAY = pd.Timestamp("2020-01-06T17:00:00Z")
# NYSE opens 09:30 ET = 13:30 UTC and closes 16:00 ET = 20:00 UTC on
# 2026-05-29 (EDT). Straddle both instants.
_FRIDAY_PREMARKET = pd.Timestamp("2026-05-29T12:00:00Z")
_FRIDAY_BEFORE_CLOSE = pd.Timestamp("2026-05-29T17:00:00Z")
_FRIDAY_AFTER_CLOSE = pd.Timestamp("2026-05-29T21:00:00Z")
_SATURDAY_MIDDAY = pd.Timestamp("2026-05-30T12:00:00Z")

_THU_OPEN = 100.0
_FRI_OPEN = 110.0


def _two_bar_frame(dates: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame({"close": [1.0, 2.0]}, index=pd.DatetimeIndex(list(dates)))


def _two_bar_open_frame(dates: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame({"open": [_THU_OPEN, _FRI_OPEN]}, index=pd.DatetimeIndex(list(dates)))


class _ExclusiveEndSource(YFinanceSource):
    """
    Stub vendor mimicking yfinance's exclusive ``end`` date.

    Returns the canned frame minus any bar dated on or after ``end``'s
    calendar date, exactly as the real vendor drops the end-dated bar. Used
    to prove the live fetchers compensate so the most recent bar survives.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        super().__init__()
        self._frame = frame

    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval = Interval.DAILY,
    ) -> pd.DataFrame:
        cutoff = pd.Timestamp(end).normalize()
        index = pd.DatetimeIndex(self._frame.index)
        return self._frame.loc[index < cutoff]


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
    the latest complete bar is Thursday - the bar acted on at Friday's open.
    """

    bars = _two_bar_frame((_THURSDAY, _FRIDAY))
    kept = _drop_unclosed_last_session(bars, _FRIDAY_BEFORE_CLOSE)

    assert len(kept) == 1
    assert kept.index[-1] == pd.Timestamp(_THURSDAY)


def test_drop_unclosed_last_session_keeps_closed_bar() -> None:
    """
    After Friday's close the Friday bar is final and survives - it becomes
    the bar the next signal (for Monday's open) is computed from.
    """

    bars = _two_bar_frame((_THURSDAY, _FRIDAY))
    kept = _drop_unclosed_last_session(bars, _FRIDAY_AFTER_CLOSE)

    assert len(kept) == 2
    assert kept.index[-1] == pd.Timestamp(_FRIDAY)


def test_drop_unclosed_last_session_keeps_past_session_over_weekend() -> None:
    """
    Loaded on Saturday, Friday's session has long closed - nothing is dropped.
    """

    bars = _two_bar_frame((_THURSDAY, _FRIDAY))
    kept = _drop_unclosed_last_session(bars, _SATURDAY_MIDDAY)

    assert len(kept) == 2


def test_drop_unclosed_last_session_empty_frame_is_noop() -> None:
    empty = pd.DataFrame({"close": []}, index=pd.DatetimeIndex([]))
    assert _drop_unclosed_last_session(empty, _FRIDAY_BEFORE_CLOSE).empty


def test_opens_of_opened_sessions_keeps_forming_open() -> None:
    """
    During Friday's open session evaluation keeps the still-forming Friday
    open - opposite of generation, which drops that bar. The open is fixed
    at the bell, so it is a valid exit price for the prior signal.
    """

    bars = _two_bar_open_frame((_THURSDAY, _FRIDAY))
    opens = _opens_of_opened_sessions(bars, _FRIDAY_BEFORE_CLOSE)

    assert len(opens) == 2
    assert opens.index[-1] == pd.Timestamp(_FRIDAY)
    assert opens.iloc[-1] == _FRI_OPEN


def test_opens_of_opened_sessions_excludes_unopened_session() -> None:
    """
    Pre-market on Friday the Friday session has not opened, so its open is
    not yet a usable price - only Thursday's open survives.
    """

    bars = _two_bar_open_frame((_THURSDAY, _FRIDAY))
    opens = _opens_of_opened_sessions(bars, _FRIDAY_PREMARKET)

    assert len(opens) == 1
    assert opens.index[-1] == pd.Timestamp(_THURSDAY)
    assert opens.iloc[-1] == _THU_OPEN


def test_opens_of_opened_sessions_normalizes_tz_aware_index() -> None:
    """
    A tz-aware vendor index is normalised to tz-naive session dates so it
    aligns with the tz-naive ``bar_ts`` anchors the scorer compares against.
    """

    tz_index = pd.DatetimeIndex([_THURSDAY, _FRIDAY]).tz_localize("UTC")
    bars = pd.DataFrame({"open": [_THU_OPEN, _FRI_OPEN]}, index=tz_index)
    opens = _opens_of_opened_sessions(bars, _FRIDAY_AFTER_CLOSE)

    assert pd.DatetimeIndex(opens.index).tz is None
    assert opens.index[-1] == pd.Timestamp(_FRIDAY)


def test_opens_of_opened_sessions_empty_frame_is_empty() -> None:
    empty = pd.DataFrame({"open": []}, index=pd.DatetimeIndex([]))
    assert _opens_of_opened_sessions(empty, _FRIDAY_BEFORE_CLOSE).empty


def test_fetch_session_opens_rejects_non_daily() -> None:
    with pytest.raises(NotImplementedError, match="only supports Interval.DAILY"):
        fetch_session_opens(
            "SPY",
            datetime(2026, 1, 1),
            datetime(2026, 1, 31),
            Interval.HOUR,
            _FRIDAY_AFTER_CLOSE,
        )


def test_daily_fetcher_keeps_bar_dated_on_end() -> None:
    """
    The vendor's ``end`` is exclusive of its date, so a naive ``end=as_of``
    drops the bar dated today - the most recent bar live inference needs. The
    daily fetcher must compensate; here the Friday bar dated exactly on ``end``
    survives instead of being silently chopped to Thursday.
    """

    source = _ExclusiveEndSource(_two_bar_frame((_PAST_THU, _PAST_FRI)))
    fetcher = DailyLiveBarFetcher(source=source)
    bars = fetcher.fetch("SPY", datetime(2019, 12, 1), datetime(2020, 1, 3), Interval.DAILY)

    assert bars.index[-1] == pd.Timestamp(_PAST_FRI)


def test_fetch_session_opens_keeps_session_dated_on_end() -> None:
    """
    Evaluation's open fetch faces the same exclusive-``end`` quirk: without
    compensation the latest session's open is dropped and a signal entered at
    that open can never be scored. The session dated on ``end`` must survive.
    """

    source = _ExclusiveEndSource(_two_bar_open_frame((_PAST_THU, _PAST_FRI)))
    opens = fetch_session_opens(
        "SPY",
        datetime(2019, 12, 1),
        datetime(2020, 1, 3),
        Interval.DAILY,
        _PAST_MON_MIDDAY,
        source=source,
    )

    assert opens.index[-1] == pd.Timestamp(_PAST_FRI)
    assert opens.iloc[-1] == _FRI_OPEN
