"""All forecasting models behind one common interface.

Every model is a class with

    fit(y)            estimate parameters on a pd.Series with a DatetimeIndex
    forecast(h)       DataFrame with columns mean, lo80, hi80, lo95, hi95
    components_       DataFrame of internal states over time
    params_           dict of fitted parameters

plus `paths(y_full, H)`, used by the backtests: with the parameters FIXED,
run the model through a longer series and return the forecasts made at every
origin. Because each filter is causal (the state at time t only uses data up
to t), this is exactly "rolling the states forward" without refitting.

SES, Holt, DampedHolt, Trigg-Leach and AR-Holt are written from scratch in
numpy/python. statsmodels is used only for the Holt-Winters benchmark (and
in the tests, to cross-check our Holt).

Smoothing models run on the seasonally adjusted series x_t = y_t / s_t and
their forecasts are multiplied back by the seasonal index s_t.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from . import config
from .data import PANDAS_FREQ, season_period
from .intervals import apply_intervals, conformal_quantiles
from .seasonal import SeasonalAdjuster, _freq_code

# Number of points used to initialise level/trend: one season, or 3 for annual.
INIT_PERIOD = {"D": 7, "W": 4, "M": 12, "Q": 4, "A": 3}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def softplus(z: float) -> float:
    """log(1 + e^z): maps any real number to a positive one, smoothly."""
    return float(np.logaddexp(0.0, z))


def inv_softplus(v: float) -> float:
    return float(np.log(np.expm1(v)))


def phi_from(u: float) -> float:
    """Map an unconstrained number to the damping range [0.80, 1.00]."""
    return 0.80 + 0.20 * float(expit(u))


def u_from_phi(phi: float) -> float:
    return float(logit((phi - 0.80) / 0.20))


def damp_sums(phi: float, H: int) -> np.ndarray:
    """[phi, phi+phi^2, ..., phi+...+phi^H] — the damped-trend multipliers."""
    return np.cumsum(phi ** np.arange(1, H + 1))


def initial_states(x: np.ndarray, m: int) -> tuple[float, float, float]:
    """Level, trend and error scale at time 0 (shared by all smoothing models).

    l0 = mean of the first m points
    b0 = (mean of points m+1..2m  -  mean of points 1..m) / m
    s0 = 1.4826 * MAD of the first differences of the first 2m points
         (1.4826 * MAD is a robust estimate of a standard deviation)
    """
    m = max(1, min(m, len(x) // 2))
    first, second = x[:m], x[m:2 * m]
    l0 = float(np.mean(first))
    b0 = float((np.mean(second) - np.mean(first)) / m)
    d = np.diff(x[:2 * m])
    s0 = 1.4826 * float(np.median(np.abs(d - np.median(d)))) if len(d) else 0.0
    if not np.isfinite(s0) or s0 <= 0:
        s0 = float(np.std(d)) if len(d) and np.std(d) > 0 else max(abs(l0) * 0.01, 1e-6)
    return l0, b0, s0


def ewma_scale(errors: np.ndarray, s0: float, lam: float = 0.1) -> np.ndarray:
    """Running error scale sigma_t = lam*|e_t| + (1-lam)*sigma_{t-1}.

    Used for the conformal intervals of every model except AR-Holt (which
    has its own sigma_t built in). NaN errors leave sigma unchanged.
    """
    out = np.empty(len(errors))
    s = s0
    for t, e in enumerate(errors):
        if np.isfinite(e):
            s = lam * abs(e) + (1 - lam) * s
        out[t] = s
    return out


def extend_index(index: pd.DatetimeIndex, H: int, freq: str) -> pd.DatetimeIndex:
    """`index` followed by the next H dates at the same frequency."""
    future = pd.date_range(index[-1], periods=H + 1, freq=PANDAS_FREQ[freq])[1:]
    return index.append(future)


def _nelder_mead(objective, starts: list[np.ndarray]) -> np.ndarray:
    """Run Nelder–Mead from several starting points, keep the best."""
    best_x, best_f = None, np.inf
    for x0 in starts:
        res = minimize(objective, np.asarray(x0, float), method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-9})
        if res.fun < best_f:
            best_x, best_f = res.x, res.fun
    return best_x


# --------------------------------------------------------------------------
# The recursions (plain python loops over floats: easy to read, and fast
# enough — a 2,300-day series takes ~2 ms per pass)
# --------------------------------------------------------------------------
def holt_filter(x, alpha, beta, phi, l0, b0, store=True, burn=0):
    """(Damped) Holt's linear trend method.

        one-step forecast  f_t = l_{t-1} + phi*b_{t-1}
        level   l_t = alpha*x_t + (1-alpha)*f_t
        trend   b_t = beta*(l_t - l_{t-1}) + (1-beta)*phi*b_{t-1}

    phi = 1 gives classic Holt; beta = 0 and b0 = 0 gives SES.
    Returns the one-step SSE after `burn` points, and (if store) the arrays.
    """
    n = len(x)
    l, b = l0, b0
    sse = 0.0
    if store:
        xhat, level, trend = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        f = l + phi * b
        e = x[t] - f
        if t >= burn:
            sse += e * e
        l_new = f + alpha * e  # = alpha*x + (1-alpha)*f
        b = beta * (l_new - l) + (1 - beta) * phi * b
        l = l_new
        if store:
            xhat[t], level[t], trend[t] = f, l, b
    if store:
        return sse, {"xhat": xhat, "level": level, "trend": trend}
    return sse, None


def trigg_leach_filter(x, beta, l0, b0, s0, gamma=0.2, store=True, burn=0):
    """Holt with Trigg & Leach's adaptive response rate (ARR).

        E_t = gamma*e_t + (1-gamma)*E_{t-1}        smoothed error
        M_t = gamma*|e_t| + (1-gamma)*M_{t-1}      smoothed absolute error
        TS_t = E_t / M_t                           tracking signal in [-1, 1]
        alpha_t = clip(|TS_{t-1}|, 0.05, 0.95)
    """
    n = len(x)
    l, b, E, M, TS = l0, b0, 0.0, s0, 0.0
    sse = 0.0
    if store:
        xhat, level, trend, alphas, tss = (np.empty(n) for _ in range(5))
    for t in range(n):
        alpha = min(max(abs(TS), 0.05), 0.95)
        f = l + b
        e = x[t] - f
        if t >= burn:
            sse += e * e
        l_new = f + alpha * e
        b = beta * (l_new - l) + (1 - beta) * b
        l = l_new
        E = gamma * e + (1 - gamma) * E
        M = gamma * abs(e) + (1 - gamma) * M
        TS = E / M if M > 0 else 0.0
        if store:
            xhat[t], level[t], trend[t], alphas[t], tss[t] = f, l, b, alpha, TS
    if store:
        return sse, {"xhat": xhat, "level": level, "trend": trend, "alpha": alphas, "ts": tss}
    return sse, None


def ar_holt_filter(x, a, bcoef, beta, phi, l0, b0, s0,
                   gamma=0.15, k=2.5, lam=0.1, tau=0.5, store=True, burn=0):
    """AR-Holt: Adaptive Robust damped Holt (the proposed method).

    For each t:
        f_t     = l_{t-1} + phi*b_{t-1}                 one-step forecast
        e_t     = x_t - f_t
        regime  = shock if |TS_{t-1}| >= tau else calm
        e~_t    = e_t                                   (shock: trust the data)
                = sigma*clip(e_t/sigma, -k, k)          (calm: Huber-clip outliers)
        alpha_t = expit(a + bcoef*|TS_{t-1}|)           smooth transition
        l_t     = f_t + alpha_t*e~_t
        b_t     = phi*b_{t-1} + alpha_t*beta*e~_t
        sigma_t = lam*|e~_t| + (1-lam)*sigma_{t-1}       running error scale
        E_t, M_t, TS_t as in Trigg & Leach, but on the cleaned error e~_t
    """
    n = len(x)
    l, b, s, E, M, TS = l0, b0, s0, 0.0, s0, 0.0
    s_floor = 1e-6 * s0
    sse = 0.0
    if store:
        cols = {c: np.empty(n) for c in ("xhat", "level", "trend", "alpha", "ts", "sigma",
                                         "sigma_prev", "error", "clean_error")}
        shock_flags = np.zeros(n, dtype=bool)
        outlier_flags = np.zeros(n, dtype=bool)
    for t in range(n):
        f = l + phi * b
        e = x[t] - f
        if t >= burn:
            sse += e * e
        shock = abs(TS) >= tau
        z = e / s
        if shock:
            et = e
            outlier = False
        else:
            et = s * min(max(z, -k), k)  # Huber-type cleaning
            outlier = abs(z) > k
        alpha = float(expit(a + bcoef * abs(TS)))
        s_prev = s
        l = f + alpha * et
        b = phi * b + alpha * beta * et
        s = max(lam * abs(et) + (1 - lam) * s, s_floor)
        E = gamma * et + (1 - gamma) * E
        M = gamma * abs(et) + (1 - gamma) * M
        TS = E / M if M > 0 else 0.0
        if store:
            cols["xhat"][t], cols["level"][t], cols["trend"][t] = f, l, b
            cols["alpha"][t], cols["ts"][t], cols["sigma"][t] = alpha, TS, s
            cols["sigma_prev"][t], cols["error"][t], cols["clean_error"][t] = s_prev, e, et
            shock_flags[t], outlier_flags[t] = shock, outlier
    if store:
        cols["shock"] = shock_flags
        cols["is_outlier"] = outlier_flags
        return sse, cols
    return sse, None


# --------------------------------------------------------------------------
# Base class: seasonal adjustment, intervals, backtest paths
# --------------------------------------------------------------------------
class BaseModel:
    name = "Base"
    uses_adjustment = True  # False: the model works on the raw series

    def __init__(self, seasonal: str | None = None):
        """seasonal: "rolling" or "fixed" seasonal indices (None = config default)."""
        self.seasonal = seasonal

    def fit(self, y: pd.Series, freq: str | None = None) -> "BaseModel":
        y = y.astype(float)
        self.freq_ = freq or _freq_code(y.index)
        self.m_ = season_period(self.freq_)
        self.init_m_ = INIT_PERIOD.get(self.freq_, max(self.m_, 3))
        self.burn_ = 2 * self.init_m_
        self.y_ = y
        m_adj = self.m_ if self.uses_adjustment else 1
        self.seasonal_mode_ = self.seasonal or config.SEASONAL_MODE
        self.adjuster_ = SeasonalAdjuster(m_adj, window=config.SEASONAL_WINDOW.get(self.freq_),
                                          mode=self.seasonal_mode_).fit(y)
        x = self.adjuster_.adjust(y).to_numpy()
        self.l0_, self.b0_, self.s0_ = initial_states(x, self.init_m_)
        self.params_ = self._estimate(x)
        run = self._run(x)
        self.components_ = self._components(run, x, y)
        self._cache = {}
        return self

    # --- to be provided by each model -----------------------------------
    def _estimate(self, x: np.ndarray) -> dict:
        return {}

    def _run(self, x: np.ndarray) -> dict:
        raise NotImplementedError

    def _hstep(self, run: dict, x: np.ndarray, H: int) -> np.ndarray:
        """(n, H) array: forecast of x[t+h] made at origin t."""
        raise NotImplementedError

    # --- shared machinery ------------------------------------------------
    def _components(self, run: dict, x: np.ndarray, y: pd.Series) -> pd.DataFrame:
        # The one-step forecast of y_t is made at t-1, so it is re-seasonalised
        # with the indices known at t-1 (identical to the fixed indices in v1).
        ahead = self.adjuster_.forecast_factors(y, extend_index(y.index, 1, self.freq_))[:, 0]
        factors = np.r_[self.adjuster_.concurrent_factors(y)[:1], ahead[:-1]]
        df = pd.DataFrame({"actual": y.to_numpy(), "adjusted": x,
                           "fitted": run["xhat"] * factors}, index=y.index)
        for key in ("level", "trend", "alpha", "ts", "sigma"):
            if key in run:
                df[key] = run[key]
        return df

    def paths(self, y_full: pd.Series, H: int) -> dict:
        """Point forecasts from every origin of y_full, parameters fixed.

        Parameters come from the training fit only. Seasonal indices are the
        training-fit indices (fixed mode) or, in rolling mode, re-estimated at
        each origin from the window ending there — never from later data.
        Returns x (adjusted series), adj (n, H) adjusted-scale forecasts,
        sigma (n,) error scale at each origin and factor (n, H) seasonal
        index of each forecast target.
        """
        y_full = y_full.astype(float)
        x = self.adjuster_.adjust(y_full).to_numpy()
        run = self._run(x)
        adj = self._hstep(run, x, H)
        if "sigma" in run:
            sigma = run["sigma"]
        else:
            sigma = ewma_scale(x - run["xhat"], self.s0_, config.AR_HOLT_HYPER["lam"])
        factor = self.adjuster_.forecast_factors(y_full, extend_index(y_full.index, H, self.freq_))
        return {"x": x, "adj": adj, "sigma": sigma, "factor": factor, "run": run}

    def quantiles(self, H: int) -> dict:
        """Conformal quantiles calibrated on the training data (cached)."""
        key = ("q", H)
        if key not in self._cache:
            p = self.paths(self.y_, H)
            self._cache[key] = conformal_quantiles(
                p["x"], p["adj"], p["sigma"], H,
                n_calib=config.CONFORMAL_N.get(self.freq_), min_origin=self.burn_)
        return self._cache[key]

    def interval_paths(self, y_full: pd.Series, H: int) -> dict:
        """Mean and 80/95% bands from every origin of y_full: (n, H) arrays."""
        p = self.paths(y_full, H)
        out = apply_intervals(p["adj"], p["sigma"], p["factor"], self.quantiles(H))
        out["sigma"] = p["sigma"]
        return out

    def forecast(self, h: int) -> pd.DataFrame:
        """Forecast the next h periods after the end of the training data."""
        bands = self.interval_paths(self.y_, h)
        idx = extend_index(self.y_.index, h, self.freq_)[-h:]
        return pd.DataFrame({c: bands[c][-1] for c in ("mean", "lo80", "hi80", "lo95", "hi95")},
                            index=idx)


# --------------------------------------------------------------------------
# 1-2. Benchmarks on the raw series
# --------------------------------------------------------------------------
class Naive(BaseModel):
    """Tomorrow = today."""
    name = "Naive"
    uses_adjustment = False

    def _run(self, x):
        xhat = np.r_[x[0], x[:-1]]
        return {"xhat": xhat}

    def _hstep(self, run, x, H):
        return np.repeat(x[:, None], H, axis=1)


class SeasonalNaive(BaseModel):
    """Same season last cycle (e.g. same weekday last week). For annual data = Naive."""
    name = "SeasonalNaive"
    uses_adjustment = False

    def _run(self, x):
        m = self.m_
        xhat = np.r_[x[:m], x[:-m]] if len(x) > m else x.copy()
        return {"xhat": xhat}

    def _hstep(self, run, x, H):
        m, n = self.m_, len(x)
        out = np.empty((n, H))
        for h in range(1, H + 1):
            lag = m * math.ceil(h / m) - h  # target t+h -> t+h - m*ceil(h/m)
            src = np.arange(n) - lag
            out[:, h - 1] = x[np.clip(src, 0, None)]
        return out


# --------------------------------------------------------------------------
# 3-5. SES, Holt, damped Holt (from scratch)
# --------------------------------------------------------------------------
class _HoltFamily(BaseModel):
    """Shared code for SES / Holt / DampedHolt."""

    def _unpack(self, theta) -> dict:
        raise NotImplementedError

    def _starts(self) -> list:
        raise NotImplementedError

    def _estimate(self, x):
        n_eff = max(len(x) - self.burn_, 1)
        scale = n_eff * self.s0_ ** 2

        def objective(theta):
            p = self._unpack(theta)
            sse, _ = holt_filter(x, p["alpha"], p["beta"], p["phi"], self.l0_,
                                 self._b0(), store=False, burn=self.burn_)
            return sse / scale if np.isfinite(sse) else 1e300

        return self._unpack(_nelder_mead(objective, self._starts()))

    def _b0(self):
        return self.b0_

    def _run(self, x):
        p = self.params_
        _, run = holt_filter(x, p["alpha"], p["beta"], p["phi"], self.l0_, self._b0())
        return run

    def _hstep(self, run, x, H):
        ds = damp_sums(self.params_["phi"], H)
        return run["level"][:, None] + ds[None, :] * run["trend"][:, None]


class SES(_HoltFamily):
    """Simple exponential smoothing: level only, flat forecasts."""
    name = "SES"

    def _b0(self):
        return 0.0

    def _unpack(self, theta):
        return {"alpha": float(expit(theta[0])), "beta": 0.0, "phi": 1.0}

    def _starts(self):
        return [[logit(0.2)], [logit(0.5)], [logit(0.8)]]


class Holt(_HoltFamily):
    """Classic additive Holt's linear trend method."""
    name = "Holt"

    def _unpack(self, theta):
        return {"alpha": float(expit(theta[0])), "beta": float(expit(theta[1])), "phi": 1.0}

    def _starts(self):
        return [[logit(0.2), logit(0.05)], [logit(0.5), logit(0.1)], [logit(0.8), logit(0.3)]]


class DampedHolt(_HoltFamily):
    """Holt with a damped trend (Gardner & McKenzie 1985), phi in [0.80, 1.00]."""
    name = "DampedHolt"

    def _unpack(self, theta):
        return {"alpha": float(expit(theta[0])), "beta": float(expit(theta[1])),
                "phi": phi_from(theta[2])}

    def _starts(self):
        return [[logit(0.2), logit(0.05), u_from_phi(0.98)],
                [logit(0.5), logit(0.1), u_from_phi(0.90)],
                [logit(0.8), logit(0.3), u_from_phi(0.95)]]


# --------------------------------------------------------------------------
# 6. Trigg-Leach adaptive response rate Holt
# --------------------------------------------------------------------------
class TriggLeachHolt(BaseModel):
    """Holt whose alpha_t = |tracking signal| (Trigg & Leach 1967). Only beta is fitted."""
    name = "TriggLeachHolt"
    gamma = 0.2

    def _estimate(self, x):
        scale = max(len(x) - self.burn_, 1) * self.s0_ ** 2

        def objective(theta):
            sse, _ = trigg_leach_filter(x, float(expit(theta[0])), self.l0_, self.b0_, self.s0_,
                                        self.gamma, store=False, burn=self.burn_)
            return sse / scale if np.isfinite(sse) else 1e300

        theta = _nelder_mead(objective, [[logit(0.05)], [logit(0.1)], [logit(0.3)]])
        return {"beta": float(expit(theta[0])), "gamma": self.gamma}

    def _run(self, x):
        _, run = trigg_leach_filter(x, self.params_["beta"], self.l0_, self.b0_, self.s0_, self.gamma)
        return run

    def _hstep(self, run, x, H):
        return run["level"][:, None] + np.arange(1, H + 1)[None, :] * run["trend"][:, None]


# --------------------------------------------------------------------------
# 7. AR-Holt — the proposed method
# --------------------------------------------------------------------------
class AdaptiveRobustHolt(BaseModel):
    """Adaptive Robust Holt (AR-Holt).

    A damped Holt in error-correction form whose smoothing parameter moves
    smoothly from a "calm" value expit(a) towards a "shock" value expit(a+b)
    as the |tracking signal| grows, with Huber clipping of isolated outliers
    that switches OFF once a persistent shift (shock regime) is detected.

    Fitted: a (real), b >= 0 (via softplus), beta in (0,1), phi in [0.80, 1.00].
    Fixed:  gamma=0.15, k=2.5, lam=0.1, tau=0.5 (config.AR_HOLT_HYPER).

    Options used for the ablation study:
        fix_b=0.0      -> alpha is constant ("robust only")
        k=np.inf       -> no clipping ("adaptive only")
    With fix_b=0 and k=inf the model is exactly DampedHolt (nesting test).
    """
    name = "AR-Holt"

    def __init__(self, fix_b: float | None = None, seasonal: str | None = None, **hyper):
        super().__init__(seasonal)
        self.fix_b = fix_b
        self.hyper = {**config.AR_HOLT_HYPER, **hyper}

    def _unpack(self, theta) -> dict:
        b = self.fix_b if self.fix_b is not None else softplus(theta[1])
        return {"a": float(theta[0]), "b": float(b), "beta": float(expit(theta[2])),
                "phi": phi_from(theta[3])}

    def _filter(self, x, p, store=True, burn=0):
        h = self.hyper
        return ar_holt_filter(x, p["a"], p["b"], p["beta"], p["phi"], self.l0_, self.b0_, self.s0_,
                              gamma=h["gamma"], k=h["k"], lam=h["lam"], tau=h["tau"],
                              store=store, burn=burn)

    def _estimate(self, x):
        scale = max(len(x) - self.burn_, 1) * self.s0_ ** 2

        def objective(theta):
            sse, _ = self._filter(x, self._unpack(theta), store=False, burn=self.burn_)
            return sse / scale if np.isfinite(sse) else 1e300

        starts = [[logit(0.2), inv_softplus(1.0), logit(0.05), u_from_phi(0.98)],
                  [logit(0.4), inv_softplus(3.0), logit(0.1), u_from_phi(0.90)],
                  [logit(0.6), inv_softplus(2.0), logit(0.2), u_from_phi(0.95)]]
        p = self._unpack(_nelder_mead(objective, starts))
        p["alpha_calm"] = float(expit(p["a"]))
        p["alpha_shock"] = float(expit(p["a"] + p["b"]))  # limit as |TS| -> 1
        p.update({k: v for k, v in self.hyper.items()})
        return p

    def _run(self, x):
        _, run = self._filter(x, self.params_)
        return run

    def _hstep(self, run, x, H):
        ds = damp_sums(self.params_["phi"], H)
        return run["level"][:, None] + ds[None, :] * run["trend"][:, None]

    def current_state(self) -> dict:
        """State after the last observation: the regime and the alpha that
        will be applied to the NEXT observation (both depend on TS_n)."""
        last = self.components_.iloc[-1]
        ts = float(last["ts"])
        return {
            "regime": "shock" if abs(ts) >= self.hyper["tau"] else "calm",
            "alpha_next": float(expit(self.params_["a"] + self.params_["b"] * abs(ts))),
            "ts": ts,
            "sigma": float(last["sigma"]),
        }

    def _components(self, run, x, y):
        df = super()._components(run, x, y)
        df["error"] = run["error"]
        df["regime"] = np.where(run["shock"], "shock", "calm")
        df["is_outlier"] = run["is_outlier"]
        return df


# --------------------------------------------------------------------------
# 8. Holt-Winters benchmark (statsmodels)
# --------------------------------------------------------------------------
class HoltWinters(BaseModel):
    """statsmodels ExponentialSmoothing: damped additive trend, multiplicative
    seasonality (period m), on the raw series. For annual data: damped Holt
    without a seasonal component.

    For the backtest, statsmodels is re-run with the fitted parameters and
    initial states fixed (optimized=False) over the longer series. Because
    the recursion is causal this gives exactly the same forecasts as
    re-running it at every origin (checked in tests/test_models.py).
    """
    name = "HoltWinters"
    uses_adjustment = False

    def _spec(self, x):
        seasonal = None
        if self.m_ > 1:
            seasonal = "mul" if np.all(x > 0) else "add"
        return {"trend": "add", "damped_trend": True, "seasonal": seasonal,
                "seasonal_periods": self.m_ if seasonal else None}

    def _estimate(self, x):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.spec_ = self._spec(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ExponentialSmoothing(x, initialization_method="estimated", **self.spec_).fit()
        p = res.params
        out = {"alpha": float(p["smoothing_level"]), "beta": float(p["smoothing_trend"]),
               "phi": float(p["damping_trend"]), "initial_level": float(p["initial_level"]),
               "initial_trend": float(p["initial_trend"])}
        if self.spec_["seasonal"]:
            out["gamma"] = float(p["smoothing_seasonal"])
            out["initial_seasons"] = np.asarray(p["initial_seasons"], float)
        out["seasonal"] = self.spec_["seasonal"] or "none"
        return out

    def _sm_fixed(self, x):
        """statsmodels results with every parameter and initial state fixed."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        p = self.params_
        init = {"initial_level": p["initial_level"], "initial_trend": p["initial_trend"]}
        fit_kw = {"smoothing_level": p["alpha"], "smoothing_trend": p["beta"], "damping_trend": p["phi"]}
        if self.spec_["seasonal"]:
            init["initial_seasonal"] = p["initial_seasons"]
            fit_kw["smoothing_seasonal"] = p["gamma"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(x, initialization_method="known", **init, **self.spec_)
            return model.fit(optimized=False, **fit_kw)

    def _run(self, x):
        res = self._sm_fixed(x)
        run = {"xhat": np.asarray(res.fittedvalues, float), "level": np.asarray(res.level, float),
               "trend": np.asarray(res.trend, float)}
        if self.spec_["seasonal"]:
            # season state i - m belongs to time i; prepend the initial seasons
            run["season_ext"] = np.r_[self.params_["initial_seasons"], np.asarray(res.season, float)]
        return run

    def _hstep(self, run, x, H):
        n, m = len(x), self.m_
        ds = damp_sums(self.params_["phi"], H)
        base = run["level"][:, None] + ds[None, :] * run["trend"][:, None]
        if not self.spec_["seasonal"]:
            return base
        out = np.empty((n, H))
        t = np.arange(n)
        for h in range(1, H + 1):
            k = (h - 1) // m
            s = run["season_ext"][t + h - m * (k + 1) + m]
            out[:, h - 1] = base[:, h - 1] * s if self.spec_["seasonal"] == "mul" else base[:, h - 1] + s
        return out

    def _components(self, run, x, y):
        df = pd.DataFrame({"actual": y.to_numpy(), "adjusted": x, "fitted": run["xhat"],
                           "level": run["level"], "trend": run["trend"]}, index=y.index)
        return df


# --------------------------------------------------------------------------
# Registry: name -> class, so the app and the backtests can loop over models
# --------------------------------------------------------------------------
MODELS = {
    "Naive": Naive,
    "SeasonalNaive": SeasonalNaive,
    "SES": SES,
    "Holt": Holt,
    "DampedHolt": DampedHolt,
    "TriggLeachHolt": TriggLeachHolt,
    "HoltWinters": HoltWinters,
    "AR-Holt": AdaptiveRobustHolt,
}

MODEL_DESCRIPTIONS = {
    "Naive": "Tomorrow equals today.",
    "SeasonalNaive": "Same day last week (same month last year).",
    "SES": "Simple exponential smoothing — level only.",
    "Holt": "Holt's linear trend — level + trend, fixed smoothing.",
    "DampedHolt": "Holt with a damped (flattening) trend.",
    "TriggLeachHolt": "Holt with Trigg–Leach adaptive response rate.",
    "HoltWinters": "statsmodels Holt–Winters, damped trend × seasonality.",
    "AR-Holt": "Proposed: adaptive α + robust outlier clipping + damped trend.",
}


def make_model(name: str, **kwargs) -> BaseModel:
    return MODELS[name](**kwargs)
