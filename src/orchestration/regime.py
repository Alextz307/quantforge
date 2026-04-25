"""Regime-detection primitives + a local registry.

The plan calls for three distinct kinds of "regime" — a calendar split
(pre / mid / post-COVID), a trend split (price vs a long-window MA), and
a volatility split (rolling-vol quintiles). Each detector tags every bar
with a regime label and exposes the run-length-encoded slices for plot
overlays.

Why this lives outside the global five registries
-------------------------------------------------
``ComponentRegistry`` already exists and is well-typed; instantiating one
more here keeps the regime layer decoupled from strategies / models /
features (which are the things a strategy needs at training time). A
detector is a *post-hoc analysis* tool, not part of the walk-forward
pipeline, so mixing it into the strategy-side registries would muddle
the boundary.

Why post-hoc cut-points are not leakage
---------------------------------------
:class:`VolatilityRegimeDetector` fits its quintile cut-points on the
*entire* fetched bar range — including days that, for a given fold, are
"future" relative to that fold's test window. This is intentional: regime
analysis interprets a finished walk-forward run, it does not feed any
information back into model training. The cut-points are descriptive
statistics, not signals; using them to assign each fold's test window to
a regime bucket is the only way to get a stable bucketing across folds.
The same logic applies to trend / period detectors (they observe the
full range to define their boundaries). If a future reader is uneasy
about the implication, the fix is to compute cuts on dev only — not to
re-frame this as leakage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.core.registry import ComponentRegistry
from src.orchestration.types import UNCLASSIFIED_LABEL, RegimeKind, RegimeSlice

_TREND_DEFAULT_WINDOW = 200
_VOLATILITY_DEFAULT_WINDOW = 20
_VOLATILITY_DEFAULT_N_QUANTILES = 5


@runtime_checkable
class IRegimeDetector(Protocol):
    """A regime detector tags each bar with a label and emits the RLE slices.

    ``tag(bars)`` returns a per-bar ``Series[str]`` aligned to ``bars.index``
    — the bar-level primitive the splitter consumes. ``slices(bars)`` is
    derived (run-length encoding of ``tag``) but kept on the protocol so
    plots / reports that want the contiguous-runs view don't have to RLE
    the series themselves.

    A detector that cannot tag a leading prefix (e.g. trend needs ``window``
    bars before the MA is defined) emits :data:`UNCLASSIFIED_LABEL` for
    those bars; the splitter treats the unclassified prefix as out-of-regime
    and excludes it from the day count.
    """

    @property
    def kind(self) -> RegimeKind: ...

    def tag(self, bars: pd.DataFrame) -> pd.Series: ...

    def slices(self, bars: pd.DataFrame) -> list[RegimeSlice]: ...


def _rle_slices(tagged: pd.Series) -> list[RegimeSlice]:
    """Run-length-encode a tagged Series into ``RegimeSlice`` ranges.

    Treats consecutive equal labels as one slice; emits one
    :class:`RegimeSlice` per contiguous run with ``start = bar_t``,
    ``end = bar_{t+run_len}`` (or last-bar for the final run, with a
    one-bar synthetic ``end`` since the last bar has no successor).
    """
    if len(tagged) == 0:
        return []
    labels = tagged.to_numpy()
    timestamps = tagged.index
    boundaries = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(labels)]))
    out: list[RegimeSlice] = []
    for s, e in zip(starts, ends, strict=True):
        # ``end`` points to the bar AFTER the last bar of this run.
        # When the run ends on the final bar, synthesise a successor
        # timestamp via the last delta (or +1ns for a single-bar series)
        # so the slice's ``end`` is exclusive in the same way every
        # other slice's ``end`` is.
        if e < len(timestamps):
            end_ts = pd.Timestamp(timestamps[e])
        else:
            last = pd.Timestamp(timestamps[-1])
            prev = pd.Timestamp(timestamps[-2]) if len(timestamps) >= 2 else last
            delta = last - prev if last != prev else pd.Timedelta("1ns")
            end_ts = last + delta
        out.append(
            RegimeSlice(
                label=str(labels[s]),
                start=pd.Timestamp(timestamps[s]),
                end=end_ts,
            )
        )
    return out


regime_registry: ComponentRegistry[IRegimeDetector] = ComponentRegistry()


@dataclass(frozen=True)
class _PeriodBoundary:
    """One labelled date range. ``start`` inclusive, ``end`` exclusive."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp


@regime_registry.register("period")
class PeriodRegimeDetector:
    """Tag each bar with a label drawn from explicit calendar boundaries.

    Configured via ``boundaries: list[{label, start, end}]`` in YAML
    (start inclusive, end exclusive). Boundaries must NOT overlap; the
    detector enforces this at construction. Bars outside every boundary
    fall through to :data:`UNCLASSIFIED_LABEL`.
    """

    def __init__(self, boundaries: Sequence[dict[str, object]]):
        if not boundaries:
            raise ValueError(
                "PeriodRegimeDetector needs at least one boundary; "
                "fix by listing {label, start, end} entries in the YAML."
            )
        parsed: list[_PeriodBoundary] = []
        for raw in boundaries:
            label = raw.get("label")
            start = raw.get("start")
            end = raw.get("end")
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"PeriodRegimeDetector boundary must include a non-empty 'label', got {raw}; "
                    f"fix by adding a 'label: <name>' entry to each boundary."
                )
            try:
                start_ts = pd.Timestamp(start)  # type: ignore[arg-type]
                end_ts = pd.Timestamp(end)  # type: ignore[arg-type]
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"PeriodRegimeDetector boundary '{label}' has invalid start/end "
                    f"(start={start!r}, end={end!r}); fix by passing ISO date strings."
                ) from e
            if start_ts >= end_ts:
                raise ValueError(
                    f"PeriodRegimeDetector boundary '{label}' has start >= end "
                    f"({start_ts} >= {end_ts}); fix by ensuring start < end."
                )
            parsed.append(_PeriodBoundary(label=label, start=start_ts, end=end_ts))
        parsed.sort(key=lambda b: b.start)
        for prev, curr in zip(parsed, parsed[1:], strict=False):
            if curr.start < prev.end:
                raise ValueError(
                    f"PeriodRegimeDetector boundaries '{prev.label}' and '{curr.label}' "
                    f"overlap ({prev.end} > {curr.start}); fix by trimming one boundary "
                    f"so each bar maps to at most one regime."
                )
        self._boundaries: tuple[_PeriodBoundary, ...] = tuple(parsed)

    @property
    def kind(self) -> RegimeKind:
        return RegimeKind.PERIOD

    def tag(self, bars: pd.DataFrame) -> pd.Series:
        labels = np.full(len(bars), UNCLASSIFIED_LABEL, dtype=object)
        idx_array = bars.index.to_numpy()
        for boundary in self._boundaries:
            in_range = (idx_array >= np.datetime64(boundary.start)) & (
                idx_array < np.datetime64(boundary.end)
            )
            labels[in_range] = boundary.label
        return pd.Series(labels, index=bars.index, dtype=object, name="regime")

    def slices(self, bars: pd.DataFrame) -> list[RegimeSlice]:
        return _rle_slices(self.tag(bars))


@regime_registry.register("trend")
class TrendRegimeDetector:
    """Tag bars as ``"bull"`` or ``"bear"`` based on close vs an N-bar MA.

    ``window`` bars at the start are :data:`UNCLASSIFIED_LABEL` (the MA is
    not yet defined). The MA is right-aligned (``min_periods = window``) so
    no future bars contribute to the regime classification of any given
    bar — important even for post-hoc analysis since the bar-level tag
    should still be meaningful in isolation.
    """

    def __init__(self, window: int = _TREND_DEFAULT_WINDOW):
        if window < 2:
            raise ValueError(
                f"TrendRegimeDetector window must be >= 2 bars, got {window}; "
                f"fix by passing a longer window (200 is the typical 'long-term' MA)."
            )
        self._window = window

    @property
    def kind(self) -> RegimeKind:
        return RegimeKind.TREND

    @property
    def window(self) -> int:
        return self._window

    def tag(self, bars: pd.DataFrame) -> pd.Series:
        if "close" not in bars.columns:
            raise ValueError(
                f"TrendRegimeDetector requires a 'close' column on bars, got "
                f"columns={list(bars.columns)}; fix by passing OHLCV bars."
            )
        ma = bars["close"].rolling(window=self._window, min_periods=self._window).mean()
        labels = np.where(
            ma.isna(),
            UNCLASSIFIED_LABEL,
            np.where(bars["close"].to_numpy() >= ma.to_numpy(), "bull", "bear"),
        )
        return pd.Series(labels, index=bars.index, dtype=object, name="regime")

    def slices(self, bars: pd.DataFrame) -> list[RegimeSlice]:
        return _rle_slices(self.tag(bars))


@regime_registry.register("volatility")
class VolatilityRegimeDetector:
    """Tag bars by quantile bin of rolling realised volatility.

    ``window`` is the rolling-stdev lookback in bars; ``n_quantiles`` is
    the number of buckets (5 → quintiles, labels ``"Q1"`` (lowest) to
    ``"Q5"`` (highest)). Quantile cut-points are fit POST-HOC on the
    full range — see the module docstring for why that is not leakage.

    Warmup bars where the rolling std is undefined receive
    :data:`UNCLASSIFIED_LABEL`.
    """

    def __init__(
        self,
        window: int = _VOLATILITY_DEFAULT_WINDOW,
        n_quantiles: int = _VOLATILITY_DEFAULT_N_QUANTILES,
    ):
        if window < 2:
            raise ValueError(
                f"VolatilityRegimeDetector window must be >= 2 bars, got {window}; "
                f"fix by passing a longer rolling window (20 is a typical default)."
            )
        if n_quantiles < 2:
            raise ValueError(
                f"VolatilityRegimeDetector n_quantiles must be >= 2, got {n_quantiles}; "
                f"fix by passing 5 for quintiles or 4 for quartiles."
            )
        self._window = window
        self._n_quantiles = n_quantiles

    @property
    def kind(self) -> RegimeKind:
        return RegimeKind.VOLATILITY

    @property
    def window(self) -> int:
        return self._window

    @property
    def n_quantiles(self) -> int:
        return self._n_quantiles

    def tag(self, bars: pd.DataFrame) -> pd.Series:
        if "close" not in bars.columns:
            raise ValueError(
                f"VolatilityRegimeDetector requires a 'close' column on bars, got "
                f"columns={list(bars.columns)}; fix by passing OHLCV bars."
            )
        log_returns = np.log(bars["close"]).diff()
        rolling_std = log_returns.rolling(window=self._window, min_periods=self._window).std()
        labels = np.full(len(bars), UNCLASSIFIED_LABEL, dtype=object)
        defined_mask = ~rolling_std.isna()
        defined_values = rolling_std[defined_mask]
        # Empty defined set → every bar is unclassified (series shorter
        # than the warmup window). Skip qcut to avoid an opaque error.
        if len(defined_values) == 0:
            return pd.Series(labels, index=bars.index, dtype=object, name="regime")
        # ``duplicates='drop'`` collapses tied edges (common when many bars
        # share the same rolling-std value). ``pd.qcut`` would otherwise
        # raise on duplicate edges and abort the analysis.
        bucket = pd.qcut(
            defined_values,
            q=self._n_quantiles,
            labels=[f"Q{i + 1}" for i in range(self._n_quantiles)],
            duplicates="drop",
        )
        labels[defined_mask.to_numpy()] = bucket.astype(str).to_numpy()
        return pd.Series(labels, index=bars.index, dtype=object, name="regime")

    def slices(self, bars: pd.DataFrame) -> list[RegimeSlice]:
        return _rle_slices(self.tag(bars))
