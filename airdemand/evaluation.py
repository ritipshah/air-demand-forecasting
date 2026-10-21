"""Rolling-origin backtests, accuracy metrics and the Diebold–Mariano test.

Backtest protocol (realistic and fast):
  * fit parameters ONCE on the training span,
  * seasonal indices: fixed from the training span (v1) or re-estimated at
    every origin from the 26 weeks before it (v2, default) — see seasonal.py,
  * roll the states forward through the test period with the parameters
    fixed (no refitting),
  * at every origin record forecasts for h = 1..H,
  * keep only forecasts whose target date falls in the test period.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from scipy import stats

from .models import MODELS, BaseModel

METRIC_COLS = ["MAE", "RMSE", "MAPE", "sMAPE", "MASE", "cov80", "cov95", "winkler95"]


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def backtest_model(model: BaseModel, y_full: pd.Series, train_end, H: int,
                   test_end=None) -> pd.DataFrame:
    """Forecasts of an already-fitted model from every test origin.

    The first origin is the last training date; targets must lie in
    (train_end, test_end]. Returns a long table with one row per
    (origin, h): origin, h, target, actual, mean, lo80, hi80, lo95, hi95.
    """
    if test_end is not None:
        y_full = y_full[:test_end]
    bands = model.interval_paths(y_full, H)
    idx = y_full.index
    n = len(idx)
    n_train = int((idx <= pd.Timestamp(train_end)).sum())
    rows = []
    for h in range(1, H + 1):
        origins = np.arange(n_train - 1, n - h)  # target t+h stays inside the data
        if len(origins) == 0:
            continue
        rows.append(pd.DataFrame({
            "origin": idx[origins],
            "h": h,
            "target": idx[origins + h],
            "actual": y_full.to_numpy()[origins + h],
            **{c: bands[c][origins, h - 1] for c in ("mean", "lo80", "hi80", "lo95", "hi95")},
        }))
    return pd.concat(rows, ignore_index=True)


def run_backtest(y: pd.Series, freq: str, train_end, test_end, H: int,
                 models: dict | None = None, seasonal: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Fit each model on y[:train_end] and backtest on (train_end, test_end].

    `models` maps a display name to a zero-argument factory (default: every
    model in the registry, using `seasonal` = "rolling" / "fixed" indices,
    or the config default when None). Returns (long forecast table, fitted
    models).
    """
    models = models or {name: partial(cls, seasonal=seasonal) for name, cls in MODELS.items()}
    y = y[:test_end]
    y_train = y[:train_end]
    frames, fitted = [], {}
    for name, factory in models.items():
        model = factory().fit(y_train, freq=freq)
        fc = backtest_model(model, y, train_end, H)
        fc.insert(0, "model", name)
        frames.append(fc)
        fitted[name] = model
    return pd.concat(frames, ignore_index=True), fitted


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def mase_scale(y_train: pd.Series, m: int) -> float:
    """In-sample MAE of the seasonal-naive forecast (lag m; m=1 for annual)."""
    v = y_train.to_numpy(dtype=float)
    m = max(int(m), 1)
    return float(np.mean(np.abs(v[m:] - v[:-m])))


def winkler(actual, lo, hi, alpha: float = 0.05) -> np.ndarray:
    """Winkler interval score: width, plus a penalty of (2/alpha) x the miss."""
    width = hi - lo
    below = (2 / alpha) * (lo - actual) * (actual < lo)
    above = (2 / alpha) * (actual - hi) * (actual > hi)
    return width + below + above


def point_metrics(actual: np.ndarray, pred: np.ndarray, scale: float) -> dict:
    err = actual - pred
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(err) / np.abs(actual)
        sape = 2 * np.abs(err) / (np.abs(actual) + np.abs(pred))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(100 * np.nanmean(ape[np.isfinite(ape)])),
        "sMAPE": float(100 * np.nanmean(sape[np.isfinite(sape)])),
        "MASE": float(np.mean(np.abs(err)) / scale) if scale > 0 else np.nan,
    }


def metrics_by_horizon(fc: pd.DataFrame, scale: float) -> pd.DataFrame:
    """One row per (model, h) with all metrics."""
    rows = []
    for (model, h), g in fc.groupby(["model", "h"], sort=False):
        a = g["actual"].to_numpy()
        row = {"model": model, "h": int(h), "n": len(g)}
        row.update(point_metrics(a, g["mean"].to_numpy(), scale))
        row["cov80"] = float(np.mean((a >= g["lo80"]) & (a <= g["hi80"])))
        row["cov95"] = float(np.mean((a >= g["lo95"]) & (a <= g["hi95"])))
        row["winkler95"] = float(np.mean(winkler(a, g["lo95"].to_numpy(), g["hi95"].to_numpy())))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(by_h: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average each metric over horizons (equal weight per horizon)."""
    return by_h.groupby(keys + ["model"], sort=False)[METRIC_COLS].mean().reset_index()


# --------------------------------------------------------------------------
# Diebold–Mariano test with the Harvey–Leybourne–Newbold correction
# --------------------------------------------------------------------------
def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int, loss: str = "squared") -> dict:
    """Test H0: models 1 and 2 have equal expected loss at horizon h.

    d_t = L(e1_t) - L(e2_t). A NEGATIVE statistic means model 1 has the
    smaller loss. The long-run variance of d uses Newey–West (Bartlett)
    weights with lag h-1, because h-step errors overlap and are
    autocorrelated up to order h-1. HLN (1997) rescale the statistic for
    small samples and compare it with a t distribution with n-1 df.
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    L = (lambda e: e ** 2) if loss == "squared" else np.abs
    d = L(e1) - L(e2)
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    lrv = np.mean(dc * dc)
    for k in range(1, h):
        w = 1 - k / h  # Bartlett weight with bandwidth h (lag h-1)
        lrv += 2 * w * np.mean(dc[k:] * dc[:-k])
    if lrv <= 0:  # fall back to the plain variance
        lrv = np.mean(dc * dc)
    dm = dbar / np.sqrt(lrv / n) if lrv > 0 else 0.0
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"n": n, "mean_loss_diff": float(dbar), "dm_stat": float(stat), "p_value": float(p)}


def dm_table(fc: pd.DataFrame, proposed: str, horizons=(1, 7)) -> pd.DataFrame:
    """DM test of `proposed` against every other model, both losses."""
    rows = []
    wide = fc.assign(err=fc["actual"] - fc["mean"])
    for h in horizons:
        g = wide[wide["h"] == h]
        if g.empty:
            continue
        base = g[g["model"] == proposed].set_index("origin")["err"]
        for other in g["model"].unique():
            if other == proposed:
                continue
            oth = g[g["model"] == other].set_index("origin")["err"]
            common = base.index.intersection(oth.index)
            for loss in ("squared", "absolute"):
                r = diebold_mariano(base[common].to_numpy(), oth[common].to_numpy(), h, loss)
                if r["p_value"] < 0.05:
                    verdict = f"{proposed} better" if r["dm_stat"] < 0 else f"{other} better"
                else:
                    verdict = "no significant difference"
                rows.append({"h": h, "loss": loss, "model": proposed, "vs": other, **r,
                             "verdict": verdict})
    return pd.DataFrame(rows)
