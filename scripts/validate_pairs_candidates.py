"""
Screen candidate asset pairs for PairsTrading viability before committing them
to a study.

The incumbent study trades only one pair, IVV / VOO - two fund vehicles for the
same S&P 500 basket. Their relationship is cointegrated to the penny, but that
is exactly the problem: the spread never moves far enough to clear the 2 bp + 2
bp round-turn cost, so the leg generates no harvestable edge. A useful pair
needs three things, none of which IVV / VOO has all of:

* **cointegration that holds on every fold** - the spread
  ``close_a - beta * close_b`` is stationary (Engle-Granger ADF p-value below
  the trading threshold) not just over the whole dev window but on each
  walk-forward training slice the study actually fits on. This is the binding
  test: cointegration drifts across sub-windows, so a pair stationary over ten
  years can still fail on an early three-year fold, and the study's per-fold
  guard rejects it there. The screen mirrors the study's walk-forward geometry
  and reports the worst fold;
* **a tradable half-life** - the spread reverts on a horizon of days-to-weeks,
  fast enough to round-trip several times inside a holdout but slow enough not
  to be microstructure noise; and
* **amplitude that clears cost** - a typical entry-to-exit excursion
  (``entry_z * sigma_spread``, as a fraction of leg-A notional) is wider than
  the round-turn friction of opening and closing both legs at the ``normal``
  cost tier. This is the gate IVV / VOO fails.

Selection is leak-free: every statistic is computed on the **dev** slice only
(everything strictly before the study's holdout boundary, resolved with the
same helper the runner uses), so picking the strongest pairs never inspects the
out-of-sample window they are later judged on.

Run from the repo root (defaults screen the 10-year window the study's 10y legs
use, reserving the same 15% holdout):

    python -m scripts.validate_pairs_candidates
    python -m scripts.validate_pairs_candidates --start 2021-01-01 --holdout-pct 0.20
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from src.core.temporal import WalkForwardValidator, resolve_holdout_boundary
from src.core.types import Interval
from src.engine.scenarios import SlippageScenario, total_cost_fraction_for
from src.models.cointegration import CointegrationTester

# Economically linked, liquid US large-caps and ETFs. Same-sector partners give
# the cointegration test a structural relationship to lean on; the two controls
# at the end (the incumbent near-duplicate and a co-trending-but-unrelated
# mega-cap pair) check that the screen rejects what it should.
CANDIDATE_PAIRS: list[tuple[str, str, str]] = [
    ("ko_pep", "KO", "PEP"),  # beverages / consumer staples
    ("xom_cvx", "XOM", "CVX"),  # integrated energy majors
    ("hd_low", "HD", "LOW"),  # home-improvement retail
    ("v_ma", "V", "MA"),  # card payment networks
    ("gs_ms", "GS", "MS"),  # bulge-bracket investment banks
    ("jpm_bac", "JPM", "BAC"),  # money-center banks
    ("cat_de", "CAT", "DE"),  # heavy machinery
    ("duk_so", "DUK", "SO"),  # regulated electric utilities
    ("gld_gdx", "GLD", "GDX"),  # gold bullion vs gold miners
    ("ups_fdx", "UPS", "FDX"),  # parcel logistics
    ("cost_wmt", "COST", "WMT"),  # big-box retail
    ("adp_payx", "ADP", "PAYX"),  # payroll-processing duopoly
    ("mco_spgi", "MCO", "SPGI"),  # credit-ratings / financial-data duopoly
    ("wm_rsg", "WM", "RSG"),  # waste-management duopoly
    ("lin_apd", "LIN", "APD"),  # industrial-gases duopoly
    ("cme_ice", "CME", "ICE"),  # derivatives-exchange duopoly
    ("kmi_wmb", "KMI", "WMB"),  # midstream natural-gas pipelines
    ("ewa_ewc", "EWA", "EWC"),  # Australia / Canada country ETFs (textbook pair)
    ("ivv_voo", "IVV", "VOO"),  # incumbent pair - near-duplicate, should FLAG on amplitude
    ("aapl_msft", "AAPL", "MSFT"),  # co-trending mega-caps, should FLAG on cointegration
]

# Cointegration is a hard prerequisite (matches PairsTradingStrategy's default
# p_value_threshold); a spread with no stationary mean has nothing to revert to.
COINT_PVALUE = 0.05

# Half-life band, in trading days. Below the floor the spread is reverting
# faster than the entry/exit machinery can act on without churning cost; above
# the ceiling it may not complete a single round trip inside a holdout.
HALFLIFE_MIN_DAYS = 2.0
HALFLIFE_MAX_DAYS = 60.0

# A pair that barely ever reaches the entry band produces too few trades to
# judge. Annualized count of entry-band crossings on the dev window.
MIN_TRIPS_PER_YEAR = 3.0

# Signal geometry used for the amplitude and trade-frequency proxies. Mirrors
# the PairsTrading defaults so the screen estimates the same machinery the
# study will actually run.
ENTRY_Z = 2.0
ZSCORE_LOOKBACK = 60

# Round-turn friction of a pairs trade at the study's ``normal`` tier. Opening
# the position turns over both legs (~1 unit of leg-A notional each, assuming
# roughly balanced legs), closing it turns over both again: four leg-crossings,
# each charged slippage + commission. Expressed in basis points of leg-A
# notional so it compares directly against the entry excursion.
ROUND_TRIP_LEG_CROSSINGS = 4.0
COST_TIER = SlippageScenario.NORMAL

# Walk-forward geometry mirrors config/strategies/pairs_trading.yaml so the
# screen tests cointegration on the exact training windows the study fits on.
WF_N_SPLITS = 3
WF_TEST_SIZE = 252
WF_GAP = 5
WF_EXPANDING = True


@dataclass(frozen=True)
class PairScreen:
    """
    One candidate pair's dev-window screening verdict.
    """

    name: str
    ticker_a: str
    ticker_b: str
    rows: int
    dev_years: float
    return_corr: float
    full_dev_pvalue: float
    worst_fold_pvalue: float
    n_folds: int
    hedge_ratio: float
    half_life_days: float
    gross_edge_bps: float
    cost_hurdle_bps: float
    trips_per_year: float

    @property
    def edge_after_cost_bps(self) -> float:
        return self.gross_edge_bps - self.cost_hurdle_bps

    @property
    def is_cointegrated(self) -> bool:
        # The study fits per fold and hard-rejects any fold that is not
        # cointegrated, so the worst fold - not the full-window fit - is what
        # decides whether a leg runs end to end.
        return self.worst_fold_pvalue < COINT_PVALUE

    @property
    def half_life_ok(self) -> bool:
        return HALFLIFE_MIN_DAYS <= self.half_life_days <= HALFLIFE_MAX_DAYS

    @property
    def amplitude_ok(self) -> bool:
        return self.edge_after_cost_bps > 0.0

    @property
    def frequency_ok(self) -> bool:
        return self.trips_per_year >= MIN_TRIPS_PER_YEAR

    @property
    def passed(self) -> bool:
        return (
            self.is_cointegrated and self.half_life_ok and self.amplitude_ok and self.frequency_ok
        )

    @property
    def score(self) -> float:
        """
        Rank key for passing pairs: per-trip margin scaled by trade frequency.

        A wide-but-rare spread and a tight-but-frequent one both make weak
        legs; the geometric blend rewards pairs that clear cost *and* trade
        often enough to compound the edge.
        """

        if not self.passed:
            return float("-inf")
        return self.edge_after_cost_bps * math.sqrt(self.trips_per_year)

    def reasons(self) -> list[str]:
        notes: list[str] = []
        if not self.is_cointegrated:
            notes.append(
                f"not cointegrated on every fold (worst p={self.worst_fold_pvalue:.3f} "
                f">= {COINT_PVALUE})"
            )
        if not self.half_life_ok:
            notes.append(
                f"half-life {self.half_life_days:.1f}d outside "
                f"[{HALFLIFE_MIN_DAYS:.0f}, {HALFLIFE_MAX_DAYS:.0f}]d"
            )
        if not self.amplitude_ok:
            notes.append(
                f"spread too tight (edge {self.gross_edge_bps:.1f}bp < "
                f"cost {self.cost_hurdle_bps:.1f}bp)"
            )
        if not self.frequency_ok:
            notes.append(
                f"too few trades ({self.trips_per_year:.1f}/yr < {MIN_TRIPS_PER_YEAR:.0f})"
            )
        return notes


def _close_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted closes for ``tickers``, inner-joined on shared calendar.
    """

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned no data for {tickers} {start}..{end}")
    close_all = pd.DataFrame(raw["Close"])
    missing = [t for t in tickers if t not in close_all.columns]
    if missing:
        raise RuntimeError(
            f"yfinance returned no Close column for {missing} (requested {tickers}, "
            f"{start}..{end}); check the symbols and the window."
        )
    # A non-positive adjusted close (rare corporate-action artifact) would make
    # the log-return ratio non-finite; drop any bar missing on either leg.
    close = close_all[tickers].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return close


def _worst_fold_pvalue(dev: pd.DataFrame, ticker_a: str, ticker_b: str) -> tuple[float, int]:
    """
    Largest Engle-Granger p-value across the study's walk-forward train folds.

    Mirrors the runner's per-fold fit: for each expanding training slice the
    study would fit on, run cointegration and keep the worst (largest) p-value.
    That worst fold is the binding constraint - the study hard-rejects the leg
    on the first fold that fails. Returns ``(+inf, 0)`` when the dev window is
    too short to split (a pair that cannot even form folds is not tradable).
    """

    validator = WalkForwardValidator(
        n_splits=WF_N_SPLITS,
        test_size=WF_TEST_SIZE,
        gap=WF_GAP,
        expanding=WF_EXPANDING,
    )
    try:
        folds = list(validator.split(dev))
    except (ValueError, TypeError):
        return float("inf"), 0
    pvalues = [
        CointegrationTester.engle_granger(
            fold.train[ticker_a], fold.train[ticker_b], COINT_PVALUE
        ).p_value
        for fold in folds
    ]
    return max(pvalues), len(pvalues)


def _half_life_days(spread: pd.Series[float]) -> float:
    """
    Mean-reversion half-life from an AR(1) fit on the spread level.

    Regress the one-step change of the spread on its lagged level
    (``delta S_t = alpha + lambda * S_{t-1}``). For a mean-reverting spread
    ``lambda`` is negative and the half-life is ``-ln(2) / lambda``; a
    non-negative ``lambda`` means no reversion and returns ``+inf``.
    """

    lagged = spread.shift(1)
    delta = spread - lagged
    aligned = pd.concat([delta, lagged], axis=1).dropna()
    y = aligned.iloc[:, 0].to_numpy(dtype=np.float64)
    x = add_constant(aligned.iloc[:, 1].to_numpy(dtype=np.float64))
    lam = float(OLS(y, x).fit().params[1])
    if lam >= 0.0:
        return float("inf")
    return -math.log(2.0) / lam


def _trips_per_year(spread: pd.Series[float], dev_years: float) -> float:
    """
    Annualized count of entry-band crossings on the rolling spread z-score.

    Counts rising edges where ``|z|`` first reaches ``ENTRY_Z`` - each such
    crossing is one position the strategy would open and later unwind, so it
    proxies the round-trip count the backtest will see.
    """

    rolling = spread.rolling(ZSCORE_LOOKBACK)
    zscore = (spread - rolling.mean()) / rolling.std()
    above = zscore.abs() >= ENTRY_Z
    entries = int((above & ~above.shift(1, fill_value=False)).sum())
    if dev_years <= 0.0:
        return 0.0
    return entries / dev_years


def _screen_pair(
    name: str,
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
    holdout_pct: float,
) -> PairScreen:
    close = _close_prices([ticker_a, ticker_b], start, end)
    # Leak-free selection: resolve the same holdout boundary the runner reserves
    # and screen on the dev slice only, never the out-of-sample window.
    boundary = resolve_holdout_boundary(close, holdout_pct=holdout_pct)
    dev = close if boundary is None else close.loc[close.index < boundary]

    price_a = dev[ticker_a]
    price_b = dev[ticker_b]
    dev_years = len(dev) / Interval.DAILY.annualization_factor()

    ratio = dev / dev.shift(1)
    returns = pd.DataFrame(
        np.log(ratio.to_numpy(dtype=np.float64)),
        index=dev.index,
        columns=dev.columns,
    ).dropna(how="any")
    return_corr = float(returns[ticker_a].corr(returns[ticker_b]))

    # Full-window fit fixes the hedge ratio and spread scale used for the
    # half-life and amplitude proxies; the per-fold worst p-value is the gate.
    coint = CointegrationTester.engle_granger(price_a, price_b, COINT_PVALUE)
    worst_fold_pvalue, n_folds = _worst_fold_pvalue(dev, ticker_a, ticker_b)
    spread: pd.Series[float] = price_a - coint.hedge_ratio * price_b
    half_life = _half_life_days(spread)

    # Amplitude vs cost, in basis points of leg-A notional. A typical entry sits
    # ENTRY_Z standard deviations off the mean and exits near it, capturing
    # ~ENTRY_Z * sigma_spread of spread; normalize by the leg-A price level to
    # read it as a return, then compare against the round-turn friction.
    mean_price_a = float(price_a.mean())
    gross_edge_bps = ENTRY_Z * (coint.spread_std / mean_price_a) * 1.0e4
    cost_hurdle_bps = ROUND_TRIP_LEG_CROSSINGS * total_cost_fraction_for(COST_TIER) * 1.0e4
    trips_per_year = _trips_per_year(spread, dev_years)

    return PairScreen(
        name=name,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        rows=len(dev),
        dev_years=dev_years,
        return_corr=return_corr,
        full_dev_pvalue=coint.p_value,
        worst_fold_pvalue=worst_fold_pvalue,
        n_folds=n_folds,
        hedge_ratio=coint.hedge_ratio,
        half_life_days=half_life,
        gross_edge_bps=gross_edge_bps,
        cost_hurdle_bps=cost_hurdle_bps,
        trips_per_year=trips_per_year,
    )


def _print_screen(screen: PairScreen) -> None:
    print(f"\n=== {screen.name}: {screen.ticker_a} / {screen.ticker_b} ===")
    print(
        f"dev rows={screen.rows} (~{screen.dev_years:.1f}y), return corr={screen.return_corr:+.2f}"
    )
    worst = f"{screen.worst_fold_pvalue:.4f}" if math.isfinite(screen.worst_fold_pvalue) else "n/a"
    print(
        f"cointegration full-dev p={screen.full_dev_pvalue:.4f}, "
        f"worst-fold p={worst} over {screen.n_folds} folds "
        f"({'YES' if screen.is_cointegrated else 'no'}), "
        f"hedge_ratio={screen.hedge_ratio:.3f}"
    )
    half_life_str = (
        f"{screen.half_life_days:.1f}d" if math.isfinite(screen.half_life_days) else "inf"
    )
    print(
        f"half-life={half_life_str} "
        f"({'ok' if screen.half_life_ok else 'flag'}), "
        f"trips/yr={screen.trips_per_year:.1f} "
        f"({'ok' if screen.frequency_ok else 'flag'})"
    )
    print(
        f"edge={screen.gross_edge_bps:.1f}bp vs cost={screen.cost_hurdle_bps:.1f}bp "
        f"-> after cost {screen.edge_after_cost_bps:+.1f}bp "
        f"({'ok' if screen.amplitude_ok else 'flag'})"
    )
    verdict = "PASS" if screen.passed else "FLAG"
    detail = "tradable" if screen.passed else "; ".join(screen.reasons())
    print(f"{verdict}: {detail}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-01-01", help="Window start (YYYY-MM-DD).")
    parser.add_argument("--end", default="2025-12-31", help="Window end (YYYY-MM-DD).")
    parser.add_argument(
        "--holdout-pct",
        type=float,
        default=0.15,
        help="Fraction reserved as holdout; screening uses only the dev slice.",
    )
    return parser.parse_args()


def main() -> int:
    warnings.simplefilter("ignore")
    args = _parse_args()
    print(
        f"screening {len(CANDIDATE_PAIRS)} candidate pairs on dev window "
        f"{args.start}..{args.end} (holdout {args.holdout_pct:.0%} reserved, untouched)"
    )

    screens: list[PairScreen] = []
    for name, ticker_a, ticker_b in CANDIDATE_PAIRS:
        try:
            screen = _screen_pair(name, ticker_a, ticker_b, args.start, args.end, args.holdout_pct)
        except (RuntimeError, ValueError) as exc:
            print(f"\n=== {name}: {ticker_a} / {ticker_b} ===\nSKIP: {exc}", file=sys.stderr)
            continue
        screens.append(screen)
        _print_screen(screen)

    passing = sorted((s for s in screens if s.passed), key=lambda s: s.score, reverse=True)
    print(f"\n{'=' * 60}")
    print(f"{len(passing)}/{len(screens)} pairs PASS the per-fold tradability screen")
    if passing:
        print("\nranked recommendations (strongest first):")
        for rank, s in enumerate(passing, start=1):
            print(
                f"  {rank}. {s.ticker_a}/{s.ticker_b:<5} "
                f"worst-fold p {s.worst_fold_pvalue:.3f}, "
                f"edge-after-cost {s.edge_after_cost_bps:+6.1f}bp, "
                f"half-life {s.half_life_days:4.1f}d, trips/yr {s.trips_per_year:4.1f}"
            )
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
