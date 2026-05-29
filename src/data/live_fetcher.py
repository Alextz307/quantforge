"""
Live OHLCV fetchers for the deployment layer.

Live inference needs bars right up to the present. The walk-forward
runner only ever asks for bars inside a frozen training/test window, so
its data path is purely cache-friendly — fingerprinted, deterministic.
Live inference is cache-friendly *as far as the past goes* but extends
the request out to ``now``, where the vendor is the only source of
truth and the cadence (daily vs. intraday) gates whether the most
recent bar is "complete enough to act on."

The :class:`LiveBarFetcher` protocol isolates that cadence-specific
freshness contract from the deployment op. Daily ships in MVP via
:class:`DailyLiveBarFetcher`; an intraday implementation lands later
without touching ``deployment.py``.

The :func:`resolve_fetcher` dispatcher selects an implementation by
:class:`~src.core.types.Interval` so the deployment op stays
cadence-agnostic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import cache
from typing import Protocol

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.core.constants import NYSE_CALENDAR_NAME
from src.core.types import Interval
from src.data.loader import YFinanceSource


class LiveBarFetcher(Protocol):
    """
    Cadence-specific live OHLCV fetcher.

    Implementations promise: ``fetch(ticker, start, end, interval)``
    returns a normalised, validated OHLCV frame whose last row's
    timestamp is **the latest bar the implementation considers complete
    for ``interval``** as of the call. Daily fetchers consider yesterday's
    bar the latest complete one until today's session closes; intraday
    fetchers must explicitly drop the in-progress bar to avoid the
    classic look-ahead-on-partial-bar bug.
    """

    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval,
    ) -> pd.DataFrame:
        """
        Fetch bars in ``[start, end]`` for ``ticker`` at ``interval``.
        """


def _vendor_end_inclusive(end: datetime) -> datetime:
    """
    Bump ``end`` one day so the vendor's exclusive end-date includes it.

    yfinance treats ``end`` as exclusive of its calendar date, so a live
    fetch with ``end=now`` silently drops the bar dated today — exactly the
    most recent bar live inference needs (and, when reproducing a training
    window, the bar at ``train_end`` itself). Adding a day pulls that bar in;
    the cadence-specific freshness filters
    (:func:`_drop_unclosed_last_session` / :func:`_opens_of_opened_sessions`)
    then decide whether it is actually actionable yet. The backtest path uses
    :class:`YFinanceSource` directly and keeps the vendor's exclusive
    semantics, so its cached windows and data hashes are unaffected.
    """

    return end + timedelta(days=1)


class DailyLiveBarFetcher:
    """
    Daily live fetcher backed by :class:`~src.data.loader.YFinanceSource`.

    yfinance exposes the *current* session's daily bar while it is still
    forming — its OHLC keeps moving until the close. Acting on that would
    pin a signal computed from an incomplete bar (the deployment signal log
    dedups by bar date and never recomputes). So the fetch drops the
    trailing bar whenever its NYSE session has not closed yet, leaving the
    latest *complete* session as the bar acted on. The cache layer
    (``~/.quant_cache``) masks vendor drift on historical bars so repeat
    predicts on the same date return identical inputs.
    """

    def __init__(self, source: YFinanceSource | None = None) -> None:
        self._source = source or YFinanceSource()

    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: Interval,
    ) -> pd.DataFrame:
        """
        Fetch via :meth:`YFinanceSource.fetch`, then drop an unclosed bar.
        """

        if interval is not Interval.DAILY:
            raise ValueError(
                f"DailyLiveBarFetcher only supports Interval.DAILY, got {interval}; "
                f"fix by routing non-daily intervals through resolve_fetcher() "
                f"once an intraday implementation lands."
            )
        bars = self._source.fetch(ticker, start, _vendor_end_inclusive(end), interval)
        return _drop_unclosed_last_session(bars, pd.Timestamp.now(tz="UTC"))


@cache
def _nyse_calendar() -> mcal.MarketCalendar:
    """
    Cached NYSE calendar — building it lazily memoises the holiday rule set.

    A fresh ``get_calendar(...)`` rebuilds that rule set on its first
    ``schedule`` call (tens of ms); reusing one instance keeps the
    per-fetch partial-bar check cheap.
    """

    return mcal.get_calendar(NYSE_CALENDAR_NAME)


def _drop_unclosed_last_session(bars: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """
    Drop a trailing daily bar whose NYSE session has not closed by ``now``.

    ``now`` is injected (not read from the wall clock here) so callers and
    tests control the reference instant, mirroring ``predict(as_of=...)``.
    The trailing bar is dropped only when its session's close is still in the
    future: historical windows and weekend/holiday-trailing fetches keep
    every row, since their last session closed in the past. Early-close
    half-days are handled correctly — the schedule carries the real close.
    """

    if bars.empty:
        return bars
    last_date = pd.Timestamp(bars.index[-1]).date()
    schedule = _nyse_calendar().schedule(start_date=last_date, end_date=last_date)
    if schedule.empty:
        return bars
    market_close = schedule["market_close"].iloc[0]
    if now < market_close:
        return bars.iloc[:-1]
    return bars


def _opens_of_opened_sessions(bars: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
    """
    Keep each session's open whose session has opened by ``now``.

    Open-boundary mirror of :func:`_drop_unclosed_last_session`: that
    drops a session whose *close* is still in the future (the close keeps
    moving, so generation can't trust it); this admits a session whose
    *open* has already printed. An open is fixed at the bell and never
    revised, so a currently-forming session contributes its open here —
    but never its still-moving close — letting a signal be scored the
    moment its exit session opens. Returns a tz-naive, normalised,
    ascending float Series; empty when the vendor has no bars.
    """

    if bars.empty:
        return pd.Series(dtype="float64")
    sessions = pd.DatetimeIndex(bars.index)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    sessions = sessions.normalize()
    opens = pd.Series(np.asarray(bars["open"], dtype=np.float64), index=sessions)
    schedule = _nyse_calendar().schedule(
        start_date=sessions[0].date(), end_date=sessions[-1].date()
    )
    market_open = schedule["market_open"].reindex(sessions)
    keep = np.asarray(market_open <= now)
    return opens[keep]


def fetch_session_opens(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: Interval,
    now: pd.Timestamp,
    *,
    source: YFinanceSource | None = None,
) -> pd.Series:
    """
    Open price per opened session for ``ticker`` over ``[start, end]``.

    The evaluation-side data source, deliberately separate from
    :class:`DailyLiveBarFetcher` (the generation source). Generation drops
    the still-forming bar because it reads the close; evaluation keeps it
    because it reads only the open. ``now`` is injected (not read from the
    wall clock) so callers and tests control the reference instant,
    mirroring ``predict(as_of=...)``. Daily cadence only today.
    """

    if interval is not Interval.DAILY:
        raise NotImplementedError(
            f"fetch_session_opens only supports Interval.DAILY, got {interval}; "
            f"intraday signal evaluation lands with the intraday fetcher."
        )
    src = source if source is not None else YFinanceSource()
    bars = src.fetch(ticker, start, _vendor_end_inclusive(end), interval)
    return _opens_of_opened_sessions(bars, now)


def resolve_fetcher(interval: Interval) -> LiveBarFetcher:
    """
    Pick the right :class:`LiveBarFetcher` for ``interval``.

    Single dispatch site for the deployment layer: ``predict()`` never
    knows which concrete fetcher it is using. When an intraday fetcher
    is added later, one new branch lands here and zero changes flow
    into ``deployment.py``.
    """

    if interval is Interval.DAILY:
        return DailyLiveBarFetcher()
    raise NotImplementedError(
        f"no LiveBarFetcher implemented for interval={interval}; daily is "
        f"the only cadence supported today. Intraday support is tracked as "
        f"a follow-up workstream — see the deployment extensibility seams."
    )
