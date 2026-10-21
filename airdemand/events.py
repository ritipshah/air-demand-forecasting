"""Turn AR-Holt's internal states into a readable table of events.

Two kinds of event:

* "Regime shift" — the model stayed in the SHOCK regime (|tracking signal|
  >= tau) for at least 3 consecutive steps: the data moved persistently in
  one direction (e.g. the COVID collapse) and AR-Holt raised alpha to
  re-level quickly.
* "Outlier" — isolated points that were flagged (|error| > k sigma while
  calm) and clipped, e.g. Thanksgiving or a snowstorm. Consecutive flagged
  points are merged into one event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

MIN_SHIFT_RUN = 3


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of consecutive True runs (end inclusive)."""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _near_holiday(start: pd.Timestamp, end: pd.Timestamp, holidays: pd.DatetimeIndex,
                  days: int = 2) -> bool:
    lo, hi = start - pd.Timedelta(days=days), end + pd.Timedelta(days=days)
    return bool(((holidays >= lo) & (holidays <= hi)).any())


def event_table(components: pd.DataFrame, daily: bool = True) -> pd.DataFrame:
    """Event table from AR-Holt components (needs regime, is_outlier, alpha,
    error and fitted-on-adjusted-scale columns).

    Columns: start, end, type, peak_alpha, mean_error_pct, near_holiday.
    mean_error_pct is the average one-step error as % of the forecast
    (negative = demand fell below what the model expected).
    """
    c = components
    shock = (c["regime"] == "shock").to_numpy()
    outlier = c["is_outlier"].to_numpy(dtype=bool)
    xhat = (c["adjusted"] - c["error"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        err_pct = 100 * c["error"].to_numpy() / xhat
    idx = c.index
    holidays = pd.DatetimeIndex([])
    if daily:
        holidays = USFederalHolidayCalendar().holidays(idx.min() - pd.Timedelta(days=5),
                                                       idx.max() + pd.Timedelta(days=5))
    rows = []
    in_shift = np.zeros(len(c), dtype=bool)
    for s, e in _runs(shock):
        if e - s + 1 >= MIN_SHIFT_RUN:
            in_shift[s:e + 1] = True
            rows.append(("Regime shift", s, e))
    # The error that TRIGGERS a shift happens one step before the shock regime
    # starts, so outliers right before a shift belong to it.
    for s, e in _runs(outlier & ~in_shift & ~np.r_[in_shift[1:], False]):
        rows.append(("Outlier", s, e))
    out = []
    for kind, s, e in rows:
        start, end = idx[s], idx[e]
        out.append({
            "start": start.date() if daily else start.strftime("%Y-%m"),
            "end": end.date() if daily else end.strftime("%Y-%m"),
            "type": kind,
            "steps": e - s + 1,
            "peak_alpha": round(float(c["alpha"].iloc[s:e + 1].max()), 3),
            "mean_error_pct": round(float(np.nanmean(err_pct[s:e + 1])), 1),
            "near_holiday": _near_holiday(start, end, holidays) if daily else False,
        })
    df = pd.DataFrame(out, columns=["start", "end", "type", "steps", "peak_alpha",
                                    "mean_error_pct", "near_holiday"])
    return df.sort_values("start").reset_index(drop=True)
