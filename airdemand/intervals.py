"""Locally-adaptive split-conformal prediction intervals.

Idea in plain words
-------------------
1. Look back over the last N forecast origins of the TRAINING data and record
   how wrong each h-step forecast was (on the seasonally adjusted scale).
2. Divide every error by the model's error scale sigma_t at that origin, so
   errors made in turbulent times count the same as errors in calm times.
   These "standardised errors" z are roughly exchangeable.
3. For each horizon h take empirical quantiles of z:
   (0.10, 0.90) for the 80% interval and (0.025, 0.975) for 95%.
4. A new interval is   forecast + q * sigma_now,   then multiplied by the
   seasonal index. Because sigma_now is large right after a shock, the
   intervals widen automatically in turbulent periods.

No normality assumption is needed — the quantiles come from the model's own
past errors ("conformal" calibration).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

LEVELS = {80: (0.10, 0.90), 95: (0.025, 0.975)}
MIN_ERRORS = 5  # fewer standardised errors than this -> borrow from a shorter horizon


def conformal_quantiles(x: np.ndarray, adj_fc: np.ndarray, sigma: np.ndarray,
                        H: int, n_calib: int | None, min_origin: int) -> dict:
    """Quantiles of standardised h-step errors, per horizon.

    Parameters
    ----------
    x        : adjusted series (length n) — the training data
    adj_fc   : (n, H) array, adj_fc[t, h-1] = forecast of x[t+h] made at origin t
    sigma    : (n,) error scale at each origin
    n_calib  : how many of the most recent origins to use (None = all)
    min_origin : first usable origin (skip the burn-in period)

    Returns {80: (lo[H], hi[H]), 95: (lo[H], hi[H])}.
    """
    n = len(x)
    start = min_origin if n_calib is None else max(min_origin, n - 1 - n_calib)
    origins = np.arange(start, n - 1)
    out = {lvl: (np.full(H, np.nan), np.full(H, np.nan)) for lvl in LEVELS}
    counts = np.zeros(H, dtype=int)
    for h in range(1, H + 1):
        t = origins[origins + h <= n - 1]
        if len(t) == 0:
            continue
        z = (x[t + h] - adj_fc[t, h - 1]) / np.maximum(sigma[t], 1e-12)
        z = z[np.isfinite(z)]
        counts[h - 1] = len(z)
        if len(z) >= MIN_ERRORS:
            for lvl, (ql, qh) in LEVELS.items():
                out[lvl][0][h - 1] = np.quantile(z, ql)
                out[lvl][1][h - 1] = np.quantile(z, qh)

    # Horizons with too few errors (only happens for short annual series):
    # borrow the last reliable horizon and widen by sqrt(h / h_ref), the
    # usual random-walk growth of forecast uncertainty.
    valid = [h for h in range(1, H + 1) if counts[h - 1] >= MIN_ERRORS]
    for lvl, (ql, qh) in LEVELS.items():
        lo, hi = out[lvl]
        for h in range(1, H + 1):
            if counts[h - 1] >= MIN_ERRORS:
                continue
            if valid:
                ref = max(v for v in valid if v <= h) if any(v <= h for v in valid) else valid[0]
                scale = np.sqrt(h / ref)
                lo[h - 1] = lo[ref - 1] * scale
                hi[h - 1] = hi[ref - 1] * scale
            else:  # no calibration data at all: normal quantiles
                lo[h - 1] = norm.ppf(ql) * np.sqrt(h)
                hi[h - 1] = norm.ppf(qh) * np.sqrt(h)
    return out


def apply_intervals(adj_fc: np.ndarray, sigma: np.ndarray, factor: np.ndarray,
                    q: dict) -> dict:
    """Turn adjusted-scale forecasts into original-scale mean + bands.

    adj_fc, factor : (n, H);  sigma : (n,);  q : output of conformal_quantiles.
    Returns arrays mean, lo80, hi80, lo95, hi95 (each (n, H)), ordered and
    with lower bounds clipped at 0 (passenger counts can't be negative).
    """
    s = np.maximum(sigma, 1e-12)[:, None]
    mean = adj_fc * factor
    lo80 = (adj_fc + q[80][0][None, :] * s) * factor
    hi80 = (adj_fc + q[80][1][None, :] * s) * factor
    lo95 = (adj_fc + q[95][0][None, :] * s) * factor
    hi95 = (adj_fc + q[95][1][None, :] * s) * factor
    # Passenger counts can't be negative, and we guarantee
    # lo95 <= lo80 <= mean <= hi80 <= hi95 (empirical quantiles can in rare
    # cases sit on the same side of zero).
    mean = np.clip(mean, 0, None)
    lo80 = np.clip(np.minimum(lo80, mean), 0, None)
    hi80 = np.maximum(hi80, mean)
    lo95 = np.clip(np.minimum(lo95, lo80), 0, None)
    hi95 = np.maximum(hi95, hi80)
    return {"mean": mean, "lo80": lo80, "hi80": hi80, "lo95": lo95, "hi95": hi95}
