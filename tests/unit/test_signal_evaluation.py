"""
Open->open scoring of a deployment's emitted signals.

Pins the realised-return math, the directional hit rule (long / short /
flat), leverage scaling, cumulative compounding, and the pending states
(exit-open or entry-open not yet available). All synthetic — no network,
no calendar — so the scorer is exercised in isolation from the
open-discipline fetch that feeds it live.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.signal_evaluation import evaluate_signals

# Six consecutive NYSE sessions; opens chosen so every open->open window
# is a clean +/-10%. A signal at session D_i enters at D_{i+1}'s open and
# exits at D_{i+2}'s open, so the four scoreable bar_ts are D1..D4.
_SESSIONS = (
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
    "2026-01-08",
    "2026-01-09",
    "2026-01-12",
)
_OPENS = (95.0, 100.0, 110.0, 99.0, 108.9, 98.01)

_D1, _D2, _D3, _D4, _D5, _D6 = (pd.Timestamp(s) for s in _SESSIONS)

_LONG = 1.0
_SHORT = -1.0
_FLAT = 0.0
_LEVERAGE = 2.0

_GAIN = 0.10
_LOSS = -0.10
_TOL = 1e-9
_COST_FRACTION = 0.001  # 10 bps per unit turnover


def _opens_series() -> pd.Series:
    return pd.Series(list(_OPENS), index=pd.DatetimeIndex(list(_SESSIONS)))


def test_long_signal_into_rising_open_is_a_hit() -> None:
    result = evaluate_signals([_D1], [_LONG], _opens_series())
    row = result.rows[0]

    assert row.scored is True
    assert row.entry_date == _D2
    assert row.exit_date == _D3
    assert row.asset_return == pytest.approx(_GAIN, abs=_TOL)
    assert row.listened_return == pytest.approx(_GAIN, abs=_TOL)
    assert row.hit is True


def test_short_signal_into_falling_open_is_a_hit() -> None:
    result = evaluate_signals([_D2], [_SHORT], _opens_series())
    row = result.rows[0]

    assert row.asset_return == pytest.approx(_LOSS, abs=_TOL)
    # short profits from the drop: -1 * -0.10 = +0.10
    assert row.listened_return == pytest.approx(_GAIN, abs=_TOL)
    assert row.hit is True


def test_wrong_direction_is_a_miss() -> None:
    # D4's open->open is -10%; a long bet loses and misses.
    result = evaluate_signals([_D4], [_LONG], _opens_series())
    row = result.rows[0]

    assert row.asset_return == pytest.approx(_LOSS, abs=_TOL)
    assert row.listened_return == pytest.approx(_LOSS, abs=_TOL)
    assert row.hit is False


def test_flat_signal_is_a_no_bet() -> None:
    result = evaluate_signals([_D1], [_FLAT], _opens_series())
    row = result.rows[0]

    assert row.scored is True
    assert row.listened_return == pytest.approx(0.0, abs=_TOL)
    assert row.hit is None
    assert result.hit_rate is None  # no directional bet to score


def test_leverage_scales_the_return() -> None:
    result = evaluate_signals([_D4], [_LEVERAGE], _opens_series())
    row = result.rows[0]

    assert row.listened_return == pytest.approx(_LEVERAGE * _LOSS, abs=_TOL)
    assert row.hit is False


def test_pending_when_exit_open_missing() -> None:
    # D5 enters at D6's open but has no exit session in the frame yet.
    result = evaluate_signals([_D5], [_LONG], _opens_series())
    row = result.rows[0]

    assert row.scored is False
    assert row.exit_open is None
    assert row.listened_return is None
    assert result.n_scored == 0


def test_pending_when_entry_open_missing() -> None:
    # D6 is the last session; no session strictly after it to enter on.
    result = evaluate_signals([_D6], [_LONG], _opens_series())
    row = result.rows[0]

    assert row.scored is False
    assert row.entry_date is None


def test_entry_is_strictly_after_bar_ts() -> None:
    # bar_ts equal to a session date must enter on the NEXT session.
    result = evaluate_signals([_D2], [_LONG], _opens_series())

    assert result.rows[0].entry_date == _D3


def test_cumulative_compounds_across_signals() -> None:
    result = evaluate_signals([_D1, _D2], [_LONG, _SHORT], _opens_series())

    assert result.rows[0].cumulative_return == pytest.approx(_GAIN, abs=_TOL)
    expected = (1.0 + _GAIN) * (1.0 + _GAIN) - 1.0
    assert result.rows[1].cumulative_return == pytest.approx(expected, abs=_TOL)
    assert result.cumulative_return == pytest.approx(expected, abs=_TOL)


def test_hit_rate_counts_directional_only() -> None:
    # D1 long hit, D2 short hit, D3 flat (no bet), D4 long miss.
    bar_timestamps = [_D1, _D2, _D3, _D4]
    signals = [_LONG, _SHORT, _FLAT, _LONG]
    result = evaluate_signals(bar_timestamps, signals, _opens_series())

    assert result.n_scored == 4
    assert result.n_hits == 2
    assert result.hit_rate == pytest.approx(2.0 / 3.0, abs=_TOL)


def test_empty_log_yields_empty_summary() -> None:
    result = evaluate_signals([], [], _opens_series())

    assert result.n_signals == 0
    assert result.n_scored == 0
    assert result.hit_rate is None
    assert result.cumulative_return is None
    assert result.mean_return is None
    assert result.net_cumulative_return is None
    assert result.net_mean_return is None


def test_zero_cost_makes_net_equal_gross() -> None:
    result = evaluate_signals([_D1], [_LONG], _opens_series())
    row = result.rows[0]

    assert row.cost == pytest.approx(0.0, abs=_TOL)
    assert row.net_listened_return == pytest.approx(row.listened_return, abs=_TOL)
    assert result.net_cumulative_return == pytest.approx(result.cumulative_return, abs=_TOL)


def test_cost_subtracts_turnover_friction() -> None:
    result = evaluate_signals([_D1], [_LONG], _opens_series(), cost_fraction=_COST_FRACTION)
    row = result.rows[0]

    # carried leverage starts flat (0): turnover = |1 - 0| = 1, so cost = cost_fraction
    assert row.cost == pytest.approx(_COST_FRACTION, abs=_TOL)
    assert row.net_listened_return == pytest.approx(_GAIN - _COST_FRACTION, abs=_TOL)
    assert result.net_cumulative_return == pytest.approx(_GAIN - _COST_FRACTION, abs=_TOL)


def test_turnover_uses_previous_signal() -> None:
    # Long then long again: no leverage change at the second signal → no cost there.
    result = evaluate_signals(
        [_D1, _D2], [_LONG, _LONG], _opens_series(), cost_fraction=_COST_FRACTION
    )
    first, second = result.rows

    assert first.cost == pytest.approx(_COST_FRACTION, abs=_TOL)  # |1 - 0| * c
    assert second.cost == pytest.approx(0.0, abs=_TOL)  # |1 - 1| * c
    assert second.net_listened_return == pytest.approx(second.listened_return, abs=_TOL)
