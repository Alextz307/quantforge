"""
Validate the CrossAssetMomentum basket choices in ``config/universes/``.

CrossAssetMomentum predicts the PRIMARY asset's next-bar direction from the
LAGGED returns of its peer tickers, so a useful basket needs two things:

* shared structure without redundancy - the peers' contemporaneous return
  correlation with the primary is meaningfully above zero (a common factor
  the model can exploit) but not near one (a near-duplicate carries no
  cross-asset information of its own); and
* lead-lag content - some predictive correlation between a peer's momentum
  ending at ``t-1`` and the primary's return at ``t``, measured at the lags
  the strategy actually uses.

This reads each CrossAssetMomentum basket straight from ``main_study.yaml``
plus the universe profiles, fetches returns from yfinance, and prints the
contemporaneous correlation block and the lead-lag table per basket with a
pass / flag verdict. Run from the repo root:

    python scripts/validate_cam_baskets.py
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.core.config import load_study_spec
from src.orchestration.study import compose_leg_config, expand_spec_into_legs

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_STUDY_PATH = REPO_ROOT / "config" / "study" / "main_study.yaml"

# A peer whose contemporaneous correlation with the primary exceeds this is a
# near-duplicate (e.g. IVV vs VOO), carrying no independent cross-asset signal.
REDUNDANT_CORR = 0.98

# At least one peer must clear this contemporaneous correlation with the
# primary, else the basket shares no common factor to exploit.
MIN_SHARED_CORR = 0.10


@dataclass(frozen=True)
class Basket:
    """
    One CrossAssetMomentum basket resolved from a study leg.
    """

    name: str
    primary: str
    features: tuple[str, ...]
    lags: tuple[int, ...]
    start: str
    end: str

    @property
    def tickers(self) -> list[str]:
        return [self.primary, *self.features]


def _discover_baskets() -> list[Basket]:
    """
    Resolve every unique CrossAssetMomentum basket from the study spec.

    Dedupes by (primary, features) so the 5y and 10y variants of one basket
    are validated once, on the widest available window.
    """

    spec = load_study_spec(MAIN_STUDY_PATH)
    legs = [
        leg
        for leg in expand_spec_into_legs(spec, repo_root=REPO_ROOT)
        if leg.strategy == "CrossAssetMomentum"
    ]
    by_key: dict[tuple[str, tuple[str, ...]], Basket] = {}
    for leg in legs:
        cfg = compose_leg_config(leg)
        params = cfg.strategy.params
        primary = str(params["primary_ticker"])
        features = tuple(str(t) for t in _as_list(params["feature_tickers"]))
        lags = tuple(_as_int(x) for x in _as_list(params["lags"]))
        start = cfg.data.start.date().isoformat()
        end = cfg.data.end.date().isoformat()
        key = (primary, features)
        prior = by_key.get(key)
        if prior is None or start < prior.start:
            by_key[key] = Basket(leg.universe, primary, features, lags, start, end)
    return list(by_key.values())


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"expected a list, got {type(value).__name__}: {value!r}")
    return value


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected an int, got {type(value).__name__}: {value!r}")
    return value


def _log_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
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
    close = close_all[tickers]
    ratio = close / close.shift(1)
    returns = pd.DataFrame(
        np.log(ratio.to_numpy(dtype=np.float64)),
        index=close.index,
        columns=close.columns,
    )
    # A non-positive adjusted close (rare corporate-action artifact) makes the
    # log ratio +/-inf, which dropna(how="any") leaves in place; coerce to NaN
    # first so the bad bar is dropped instead of poisoning every correlation.
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def _validate_basket(basket: Basket) -> bool:
    print(f"\n=== {basket.name}: {basket.primary} <- {list(basket.features)} ===")
    print(f"window {basket.start}..{basket.end}, lags {list(basket.lags)}")
    rets = _log_returns(basket.tickers, basket.start, basket.end)
    print(f"rows={len(rets)}\n")

    print("contemporaneous return correlation:")
    print(rets.corr().round(2).to_string())

    primary_ret = rets[basket.primary]
    peer_corr = {f: float(primary_ret.corr(rets[f])) for f in basket.features}

    print("\nlead-lag (peer momentum ending t-1 -> primary return at t):")
    rows: list[tuple[str, ...]] = []
    best_leadlag = 0.0
    for f in basket.features:
        cells: list[str] = [f]
        for lag in basket.lags:
            momentum = rets[f].rolling(lag).sum().shift(1)
            df = pd.concat([primary_ret, momentum], axis=1).dropna()
            corr = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
            if math.isnan(corr):
                cells.append("n/a")
                continue
            cells.append(f"{corr:+.3f}")
            best_leadlag = max(best_leadlag, abs(corr))
        rows.append(tuple(cells))
    header = "peer".ljust(8) + "".join(f"lag{lag}".rjust(9) for lag in basket.lags)
    print(header)
    for row in rows:
        print(row[0].ljust(8) + "".join(c.rjust(9) for c in row[1:]))

    redundant = [f for f, c in peer_corr.items() if abs(c) > REDUNDANT_CORR]
    finite_shared = [abs(c) for c in peer_corr.values() if not math.isnan(c)]
    max_shared = max(finite_shared) if finite_shared else 0.0
    ok = not redundant and max_shared >= MIN_SHARED_CORR
    verdict = "PASS" if ok else "FLAG"
    notes: list[str] = []
    if redundant:
        notes.append(f"near-duplicate peers (corr>{REDUNDANT_CORR}): {redundant}")
    if max_shared < MIN_SHARED_CORR:
        notes.append(f"no shared factor (max peer corr {max_shared:.2f} < {MIN_SHARED_CORR})")
    notes.append(f"max |lead-lag| = {best_leadlag:.3f}")
    print(f"\n{verdict}: {'; '.join(notes)}")
    return ok


def main() -> int:
    warnings.simplefilter("ignore")
    baskets = _discover_baskets()
    if not baskets:
        print("no CrossAssetMomentum baskets found in the study spec", file=sys.stderr)
        return 1
    results = [_validate_basket(b) for b in baskets]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} baskets PASS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
