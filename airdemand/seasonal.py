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

Two modes
---------
* "fixed" (v1): the indices are estimated ONCE, on the trailing window of
  the training data, and used for every date afterwards.
* "rolling" (v2, the default): the indices are re-estimated at EVERY time
  step t, using only the `window` observations ending at t (26 weeks for
  daily data; window=None means all data up to t). Observation t is
  adjusted with the indices known on day t, and a forecast made at origin t
  is re-seasonalised with the indices estimated at t. Nothing after t is
  ever used, so the weekday pattern can drift without look-ahead.
"""

from __future__ import annotations

import hashlib

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


# Rolling index matrices are re-used by every model fitted on the same data,
# so they are cached by a fingerprint of the series (small, bounded cache).
_ROLLING_CACHE: dict = {}
_ROLLING_CACHE_MAX = 64


def _fingerprint(y: pd.Series, m: int, window) -> tuple:
    digest = hashlib.sha1(np.ascontiguousarray(y.to_numpy(dtype=float)).tobytes()).hexdigest()
    return (digest, len(y), y.index[0], y.index[-1], m, window)


class SeasonalAdjuster:
    """Estimate multiplicative seasonal indices and apply / remove them.

    Parameters
    ----------
    m : seasonal period (7 daily, 12 monthly, 1 = no seasonality)
    window : number of most recent observations used to estimate the
             indices (None = all available data)
    mode : "fixed" — estimate once on the training data (v1);
           "rolling" — re-estimate at every time step from the `window`
           observations ending there (v2)
    """

    # Before this many full seasonal cycles exist, rolling mode re-uses the
    # first available estimate (the same convention as initialising the
    # level from the first 2m points).
    MIN_CYCLES = 4

    def __init__(self, m: int, window: int | None = None, mode: str = "fixed"):
        if mode not in ("fixed", "rolling"):
            raise ValueError(f"mode must be 'fixed' or 'rolling', not {mode!r}")
        self.m = int(m)
        self.window = window
        self.mode = mode
        self.indices_ = np.ones(max(self.m, 1))

    @staticmethod
    def _estimate(ratio: np.ndarray, pos: np.ndarray, m: int) -> np.ndarray:
        """Median ratio per season position, normalised to mean 1."""
        idx = np.ones(m)
        ok = np.isfinite(ratio)
        for j in range(m):
            r = ratio[(pos == j) & ok]
            if len(r):
                idx[j] = np.median(r)
        return idx / idx.mean()

    def fit(self, y: pd.Series) -> "SeasonalAdjuster":
        """Estimate indices on the trailing window of y.

        In fixed mode these are THE indices; in rolling mode they are just
        the latest estimate (kept for display) — the rolling indices are
        computed from whatever series is passed to adjust() etc.
        """
        if self.m <= 1:
            self.indices_ = np.ones(1)
            return self
        if self.window is not None:
            y = y.iloc[-self.window:]
        values = y.to_numpy(dtype=float)
        cma = centred_moving_average(values, self.m)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = values / cma
        self.indices_ = self._estimate(ratio, season_positions(y.index, self.m), self.m)
        return self

    # --- the index set in force at each time step ------------------------
    def origin_indices(self, y: pd.Series) -> np.ndarray:
        """(n, m) array: row t = the indices known at time t.

        Fixed mode: every row equals indices_. Rolling mode: row t is
        estimated from the observations in (t - window, t] only. The centred
        moving average at point j needs y up to j + m/2, so inside that window
        the usable ratios are j in [start + half, t - half] — exactly what
        fit() would compute on the window, but without refitting n times.
        """
        n, m = len(y), self.m
        if m <= 1:
            return np.ones((n, 1))
        if self.mode == "fixed":
            return np.tile(self.indices_, (n, 1))
        key = _fingerprint(y, m, self.window)
        if key in _ROLLING_CACHE:
            return _ROLLING_CACHE[key]
        values = y.to_numpy(dtype=float)
        cma = centred_moving_average(values, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = values / cma
        pos = season_positions(y.index, m)
        half = m // 2  # (m-1)/2 for odd m; m/2 for the 2 x m MA of even m
        first = min(self.MIN_CYCLES * m - 1, n - 1)
        S = np.empty((n, m))
        for t in range(first, n):
            start = 0 if self.window is None else max(0, t - self.window + 1)
            lo, hi = start + half, t - half
            S[t] = self._estimate(ratio[lo:hi + 1], pos[lo:hi + 1], m) if hi >= lo else 1.0
        S[:first] = S[first]
        if len(_ROLLING_CACHE) >= _ROLLING_CACHE_MAX:
            _ROLLING_CACHE.pop(next(iter(_ROLLING_CACHE)))
        _ROLLING_CACHE[key] = S
        return S

    def concurrent_factors(self, y: pd.Series) -> np.ndarray:
        """Index used to adjust each observation: S_t[position of t]."""
        if self.m <= 1:
            return np.ones(len(y))
        S = self.origin_indices(y)
        return S[np.arange(len(y)), season_positions(y.index, self.m)]

    def forecast_factors(self, y: pd.Series, extended_index: pd.DatetimeIndex) -> np.ndarray:
        """(n, H) array: index for target t+h when forecasting from origin t,
        i.e. S_t[position of t+h]. `extended_index` = y's dates followed by
        the next H dates."""
        n = len(y)
        H = len(extended_index) - n
        if self.m <= 1:
            return np.ones((n, H))
        S = self.origin_indices(y)
        pos = season_positions(extended_index, self.m)
        rows = np.arange(n)
        return np.stack([S[rows, pos[rows + h]] for h in range(1, H + 1)], axis=1)

    def factors(self, index: pd.DatetimeIndex) -> np.ndarray:
        """The latest estimated indices for each date in `index` (fixed mode
        use, and for display; works for future dates)."""
        if self.m <= 1:
            return np.ones(len(index))
        return self.indices_[season_positions(index, self.m)]

    def adjust(self, y: pd.Series) -> pd.Series:
        """Remove seasonality: x_t = y_t / S_t[position of t]."""
        return y / self.concurrent_factors(y)

    def readjust(self, x: pd.Series) -> pd.Series:
        """Put seasonality back (exact inverse of adjust for fixed mode).
        In rolling mode the indices come from the data itself, so use
        readjust_from(y, x)."""
        return x * self.factors(x.index)

    def readjust_from(self, y: pd.Series, x: pd.Series) -> pd.Series:
        """Inverse of adjust in either mode: x_t * S_t[position of t], where
        the indices S_t are those estimated from the original series y."""
        return x * self.concurrent_factors(y)

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
