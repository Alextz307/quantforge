"""
Backward scoring of a deployment's emitted signals.

Pure, read-only counterpart to signal *generation*
(:func:`src.orchestration.deployment.predict`). The two never mix:

* Generation reads a session's **close** to emit the *next* signal and
  appends it to the log.
* Evaluation reads session **opens** to score a signal already emitted.
  It touches no close price and writes nothing.

Scoring convention (open->open)
-------------------------------
A signal emitted from the close of bar ``t`` is entered at the open of
the next session (its ``signal_date``) and handed off to the following
signal at the next session's open. So the return attributable to it is

    listened_return = signal * (open[exit] / open[entry] - 1)

where ``entry`` is the first session strictly after ``bar_ts`` and
``exit`` is the session after ``entry``. These one-session windows tile
the calendar end-to-end - each owned by exactly one signal - so the
per-signal returns compound into the held-equity curve.

A signal is *scored* only once both opens exist in ``opens``; until the
exit session has opened it stays pending. The caller supplies an opens
series already filtered to opened sessions (open-discipline), so a
not-yet-opened session can never contribute an exit price.

Anti-leakage
------------
Frozen signals scored against subsequently-realised opens - no forecast
leakage. An open price is fixed at the bell and never revised, so using a
currently-forming session's open is safe (unlike its close). No
``bfill`` / ``fillna``: a FLAT (``0.0``) signal is a no-bet, recorded with
``hit=None``, never a zero-filled return.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScoredSignal:
    """
    One emitted signal scored open->open (or marked pending).

    ``scored`` is the transactional flag: every realised field
    (``entry_open`` .. ``net_cumulative_return``) is populated together
    when both the entry and exit session opens are available, and is
    ``None`` otherwise. ``hit`` is ``None`` for a FLAT signal (no
    directional bet).

    ``listened_return`` / ``cumulative_return`` are **gross** (frictionless).
    ``cost`` is the friction of the rebalance that established this signal's
    position (``|delta leverage| x cost_fraction``); ``net_listened_return`` and
    ``net_cumulative_return`` are gross minus that cost. With
    ``cost_fraction=0`` the net fields equal the gross ones.
    """

    bar_ts: pd.Timestamp
    signal: float
    entry_date: pd.Timestamp | None
    entry_open: float | None
    exit_date: pd.Timestamp | None
    exit_open: float | None
    asset_return: float | None
    listened_return: float | None
    hit: bool | None
    cumulative_return: float | None
    cost: float | None
    net_listened_return: float | None
    net_cumulative_return: float | None
    scored: bool


@dataclass(frozen=True)
class SignalEvaluation:
    """
    Per-signal scores plus headline stats over the scored subset.

    ``hit_rate`` is over directional (non-FLAT) scored signals only, so a
    log of all-FLAT signals reports ``hit_rate=None`` rather than a
    misleading 0. ``cumulative_return`` compounds every scored
    ``listened_return``; ``None`` when nothing is scored yet.
    """

    rows: tuple[ScoredSignal, ...]
    n_signals: int
    n_scored: int
    n_hits: int
    hit_rate: float | None
    cumulative_return: float | None
    mean_return: float | None
    net_cumulative_return: float | None
    net_mean_return: float | None


def _normalize(ts: pd.Timestamp) -> pd.Timestamp:
    naive = ts.tz_localize(None) if ts.tzinfo is not None else ts
    return naive.normalize()


def _hit(signal: float, asset_return: float) -> bool | None:
    if signal > 0.0:
        return asset_return > 0.0
    if signal < 0.0:
        return asset_return < 0.0
    return None


def evaluate_signals(
    bar_timestamps: Sequence[pd.Timestamp],
    signal_values: Sequence[float],
    opens: pd.Series,
    *,
    cost_fraction: float = 0.0,
) -> SignalEvaluation:
    """
    Score each emitted signal open->open against ``opens``.

    ``bar_timestamps`` / ``signal_values`` are the log in append (chrono)
    order. ``opens`` maps session timestamp -> open price; its index must
    be sorted ascending and span from before the earliest ``bar_ts`` so
    the first session strictly after each ``bar_ts`` resolves. Only
    opened sessions belong in ``opens`` (open-discipline) - that
    membership alone decides whether a signal's exit price exists.

    ``cost_fraction`` is the per-unit-turnover friction (slippage +
    commission, as a notional fraction). Each signal pays
    ``|signal - carried_leverage| x cost_fraction`` to rebalance into its
    position at the entry open; net returns subtract that. Costs compound
    with the gross series into a net cumulative curve. ``cost_fraction=0``
    makes net identical to gross.
    """

    opens = opens.sort_index()
    index = opens.index
    n_sessions = len(index)

    rows: list[ScoredSignal] = []
    scored_returns: list[float] = []
    scored_net_returns: list[float] = []
    n_hits = 0
    n_directional_scored = 0
    equity = 1.0
    net_equity = 1.0
    carried_leverage = 0.0

    for bar_ts, signal in zip(bar_timestamps, signal_values, strict=True):
        bar_ts_norm = _normalize(bar_ts)
        entry_pos = int(index.searchsorted(bar_ts_norm, side="right"))

        entry_date: pd.Timestamp | None = None
        entry_open: float | None = None
        exit_date: pd.Timestamp | None = None
        exit_open: float | None = None
        asset_return: float | None = None
        listened_return: float | None = None
        cumulative_return: float | None = None
        cost: float | None = None
        net_listened_return: float | None = None
        net_cumulative_return: float | None = None
        hit: bool | None = None
        scored = False

        if entry_pos < n_sessions:
            entry_date = pd.Timestamp(index[entry_pos])
            entry_open = float(opens.iloc[entry_pos])
            turnover_cost = abs(signal - carried_leverage) * cost_fraction
            exit_pos = entry_pos + 1
            if exit_pos < n_sessions:
                exit_date = pd.Timestamp(index[exit_pos])
                exit_open = float(opens.iloc[exit_pos])
                asset_return = exit_open / entry_open - 1.0
                listened_return = signal * asset_return
                hit = _hit(signal, asset_return)
                scored = True
                equity *= 1.0 + listened_return
                cumulative_return = equity - 1.0
                cost = turnover_cost
                net_listened_return = listened_return - turnover_cost
                net_equity *= 1.0 + net_listened_return
                net_cumulative_return = net_equity - 1.0
                scored_returns.append(listened_return)
                scored_net_returns.append(net_listened_return)
                if signal != 0.0:
                    n_directional_scored += 1
                    if hit:
                        n_hits += 1
            # The position is established whether or not the exit has printed,
            # so carry it forward for the next signal's turnover.
            carried_leverage = signal

        rows.append(
            ScoredSignal(
                bar_ts=bar_ts_norm,
                signal=signal,
                entry_date=entry_date,
                entry_open=entry_open,
                exit_date=exit_date,
                exit_open=exit_open,
                asset_return=asset_return,
                listened_return=listened_return,
                hit=hit,
                cumulative_return=cumulative_return,
                cost=cost,
                net_listened_return=net_listened_return,
                net_cumulative_return=net_cumulative_return,
                scored=scored,
            )
        )

    n_scored = len(scored_returns)
    return SignalEvaluation(
        rows=tuple(rows),
        n_signals=len(rows),
        n_scored=n_scored,
        n_hits=n_hits,
        hit_rate=(n_hits / n_directional_scored) if n_directional_scored > 0 else None,
        cumulative_return=(equity - 1.0) if n_scored > 0 else None,
        mean_return=(sum(scored_returns) / n_scored) if n_scored > 0 else None,
        net_cumulative_return=(net_equity - 1.0) if n_scored > 0 else None,
        net_mean_return=(sum(scored_net_returns) / n_scored) if n_scored > 0 else None,
    )


__all__ = [
    "ScoredSignal",
    "SignalEvaluation",
    "evaluate_signals",
]
