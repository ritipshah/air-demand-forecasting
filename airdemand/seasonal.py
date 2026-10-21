"""Multiplicative seasonal adjustment (pre-processing for Holt's method).

Holt's linear trend method has a level and a trend but NO seasonal
component. Daily airport traffic has a strong weekday pattern (Fridays busy,
Saturdays quiet), so we:

1. estimate one multiplicative index per season position
   (e.g. Friday = 1.08 means Fridays are 8% above the weekly average),
2. divide the data by its index  -> seasonally adjusted series x_t,
3. forecast x_t with Holt / AR-Holt,
4. multiply the forecasts by the index again ("readjust").

Index estimation (classical ratio-to-moving-average method):
    ratio_t = y_t / CMA_t   where CMA is a centred moving average of length m
                            (a 2 x m MA when m is even, e.g. monthly data)
    index_j = median of the ratios at season position j
    indices normalised to have mean 1.
The MEDIAN (not the mean) keeps one Thanksgiving or snowstorm from
distorting the weekday index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Pandas *period* alias per frequency code, used to number season positions.
_PERIOD_ALIAS = {"D": "D", "W": "W", "M": "M", "Q": "Q", "A": "Y"}


def _freq_code(index: pd.DatetimeIndex) -> str:
    alias = (index.freqstr or pd.infer_freq(index) or "D").upper()
    for code in ("W", "MS", "ME", "M", "QS", "QE", "Q", "YS", "YE", "Y", "A", "D"):
        if alias.startswith(code):
            return {"MS": "M", "ME": "M", "QS": "Q", "QE": "Q", "YS": "A", "YE": "A", "Y": "A"}.get(code, code)
    return "D"


def season_positions(index: pd.DatetimeIndex, m: int) -> np.ndarray:
    """Season position (0..m-1) of each date.

    Uses the pandas period ordinal (days / months / quarters since 1970), so
    the numbering is the same for training dates and future dates. For daily
    data with m=7 this is a fixed weekday mapping; for monthly data with m=12
    it equals month - 1.
    """
    if m <= 1:
        return np.zeros(len(index), dtype=int)
    alias = _PERIOD_ALIAS[_freq_code(index)]
    return np.asarray(index.to_period(alias).asi8 % m, dtype=int)


def centred_moving_average(y: np.ndarray, m: int) -> np.ndarray:
    """Centred MA of length m (odd m) or 2 x m (even m). NaN at the ends."""
    if m % 2 == 1:
        w = np.ones(m) / m
    else:
        w = np.r_[0.5, np.ones(m - 1), 0.5] / m
    half = len(w) // 2
    out = np.full(len(y), np.nan)
    if len(y) >= len(w):
        out[half:len(y) - half] = np.convolve(y, w, mode="valid")
    return out


class SeasonalAdjuster:
    """Estimate multiplicative seasonal indices and apply / remove them.

    Parameters
    ----------
    m : seasonal period (7 daily, 12 monthly, 1 = no seasonality)
    window : use only the trailing `window` observations of the training
             data to estimate the indices (None = all of it)
    """

    def __init__(self, m: int, window: int | None = None):
        self.m = int(m)
        self.window = window
        self.indices_ = np.ones(max(self.m, 1))

    def fit(self, y: pd.Series) -> "SeasonalAdjuster":
        if self.m <= 1:
            self.indices_ = np.ones(1)
            return self
        if self.window is not None:
            y = y.iloc[-self.window:]
        values = y.to_numpy(dtype=float)
        cma = centred_moving_average(values, self.m)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = values / cma
        pos = season_positions(y.index, self.m)
        idx = np.ones(self.m)
        for j in range(self.m):
            r = ratio[(pos == j) & np.isfinite(ratio)]
            if len(r):
                idx[j] = np.median(r)
        self.indices_ = idx / idx.mean()  # normalise to mean 1
        return self

    def factors(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Seasonal index for each date in `index` (works for future dates)."""
        if self.m <= 1:
            return np.ones(len(index))
        return self.indices_[season_positions(index, self.m)]

    def adjust(self, y: pd.Series) -> pd.Series:
        """Remove seasonality: x_t = y_t / index."""
        return y / self.factors(y.index)

    def readjust(self, x: pd.Series) -> pd.Series:
        """Put seasonality back: y_t = x_t * index."""
        return x * self.factors(x.index)

    def index_table(self, index: pd.DatetimeIndex) -> pd.Series:
        """Readable table of the indices, keyed by a label of each position."""
        if self.m <= 1:
            return pd.Series([1.0], index=["(none)"])
        pos = season_positions(index, self.m)
        first = {}
        for d, p in zip(index, pos):
            first.setdefault(p, d)
        code = _freq_code(index)
        labels = {}
        for p, d in first.items():
            labels[p] = d.day_name() if code == "D" else (d.month_name() if code == "M" else f"Q{d.quarter}")
        return pd.Series([self.indices_[p] for p in sorted(labels)], index=[labels[p] for p in sorted(labels)])
