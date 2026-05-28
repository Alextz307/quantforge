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

from datetime import datetime
from typing import Protocol

import pandas as pd

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


class DailyLiveBarFetcher:
    """
    Daily live fetcher backed by :class:`~src.data.loader.YFinanceSource`.

    Daily bars settle at session close: by the time yfinance exposes a
    bar for date ``T``, the session is over and the bar is final, so no
    partial-bar drop is needed. The cache layer (``~/.quant_cache``)
    masks vendor drift on historical bars so repeat predicts on the
    same date return identical inputs.
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
        Delegate to :meth:`YFinanceSource.fetch` (cache → fetch → normalize).
        """

        if interval is not Interval.DAILY:
            raise ValueError(
                f"DailyLiveBarFetcher only supports Interval.DAILY, got {interval}; "
                f"fix by routing non-daily intervals through resolve_fetcher() "
                f"once an intraday implementation lands."
            )
        return self._source.fetch(ticker, start, end, interval)


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
