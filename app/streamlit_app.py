"""AeroCast — Streamlit dashboard for short-term air passenger demand forecasting.

Run locally from the repository root:

    streamlit run app/streamlit_app.py

Design notes
------------
* All modelling lives in the `airdemand` package; this file only builds the UI.
* Backtest results are READ from results/ (pre-computed by
  scripts/run_backtests.py), never recomputed, so the app stays light enough
  for a 1 GB EC2 instance.
* Data loading and model fitting are cached (st.cache_data / st.cache_resource).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airdemand import config  # noqa: E402
from airdemand.capacity import (QUANTILES, flight_plan, forecast_at_quantile,  # noqa: E402
                                peak_hours, staffing_plan)
from airdemand.data import (BUILTIN, UploadError, load_hourly_profile,  # noqa: E402
                            load_series, parse_uploaded_csv, season_period)
from airdemand.evaluation import (dm_table, mase_scale, metrics_by_horizon,  # noqa: E402
                                  run_backtest, summarise)
from airdemand.events import event_table  # noqa: E402
from airdemand.models import MODEL_DESCRIPTIONS, MODELS, AdaptiveRobustHolt, Holt  # noqa: E402

st.set_page_config(layout="wide", page_title="AeroCast", page_icon="✈️")

# --------------------------------------------------------------------------
# Palette (mid-tone colours that read on both light and dark backgrounds)
# --------------------------------------------------------------------------
COL = {"AR-Holt": "#eb6834", "Holt": "#2a78d6", "DampedHolt": "#1baf7a", "HoltWinters": "#7a6ae0",
       "TriggLeachHolt": "#9a9893", "SES": "#9a9893", "SeasonalNaive": "#9a9893", "Naive": "#9a9893"}
ACTUAL = "#8a8984"
SHOCK = "rgba(235,104,52,0.14)"
FREQ_UNIT = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "A": "year"}
UNIT_SCALE = {"AirPassengers": 1_000, "Australia": 1_000_000}  # series units -> passengers


def band_rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def show(fig: go.Figure, height: int = 420) -> None:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=70, b=10), hovermode="x unified",
                      title=dict(x=0, xanchor="left", y=0.98, yanchor="top"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    st.plotly_chart(fig, theme="streamlit", width="stretch")


# --------------------------------------------------------------------------
# Cached data + models
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_series(name: str) -> tuple[pd.Series, str]:
    return load_series(name)


@st.cache_data(show_spinner=False)
def cached_upload(content: bytes) -> tuple[pd.Series, str]:
    return parse_uploaded_csv(content)


@st.cache_data(show_spinner=False)
def cached_results(version: str = config.SEASONAL_MODE) -> dict:
    """Backtest results: "rolling" = v2 (default, results/), "fixed" = v1."""
    r = config.RESULTS_DIR if version == "rolling" else config.RESULTS_V1_DIR
    return {name: pd.read_csv(r / f"{name}.csv") for name in
            ("metrics_summary", "metrics_by_horizon", "dm_tests")}


@st.cache_data(show_spinner=False)
def cached_profile() -> pd.DataFrame:
    return load_hourly_profile()


@st.cache_resource(show_spinner=False, max_entries=24)
def fit_model(series_key: str, model_name: str, start: str, end: str, _y: pd.Series, freq: str):
    """Fit one model. `series_key`, `start` and `end` identify the data for
    the cache; `_y` (not hashed) is the data itself."""
    return MODELS[model_name]().fit(_y, freq=freq)


@st.cache_data(show_spinner=False, max_entries=8)
def live_backtest(series_key: str, _y: pd.Series, freq: str, H: int):
    """Quick backtest for uploaded data: train on the first 80%, test on the rest."""
    n = len(_y)
    train_end = _y.index[int(0.8 * n) - 1]
    fc, _ = run_backtest(_y, freq, train_end, _y.index[-1], H)
    by_h = metrics_by_horizon(fc, mase_scale(_y[:train_end], season_period(freq)))
    summary = summarise(by_h.assign(experiment="live"), ["experiment"])
    dm = dm_table(fc, config.PROPOSED, sorted({1, min(7, H)}))
    return summary, by_h, dm, train_end


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("✈️ AeroCast")
st.sidebar.caption("Short-term air passenger demand forecasting")

options = [*config.AIRPORTS, config.ALL7, "AirPassengers", "Australia", "Upload CSV"]
labels = {**BUILTIN, "Upload CSV": "Upload CSV…"}
dataset = st.sidebar.selectbox("Dataset", options, index=options.index(config.ALL7),
                               format_func=lambda k: labels[k])

if dataset == "Upload CSV":
    up = st.sidebar.file_uploader("CSV with two columns: date, value", type=["csv", "txt"])
    st.sidebar.caption("Frequency (daily, weekly, monthly, quarterly, annual) is detected automatically.")
    if up is None:
        st.title("AeroCast")
        st.info("Upload a CSV with two columns — a date and a passenger count — to start. "
                "Example: `date,passengers` / `2024-01-01,51234`.")
        st.stop()
    content = up.getvalue()
    try:
        y_all, freq = cached_upload(content)
    except UploadError as exc:
        st.title("AeroCast")
        st.error(f"Could not use this file: {exc}")
        st.stop()
    series_key = "upload-" + hashlib.md5(content).hexdigest()
    series_label = f"Uploaded: {up.name}"
else:
    y_all, freq = cached_series(dataset)
    series_key = dataset
    series_label = BUILTIN[dataset]

is_tsa = dataset in config.AIRPORTS or dataset == config.ALL7
m = season_period(freq)
unit = FREQ_UNIT[freq]

model_name = st.sidebar.selectbox("Model", config.MODEL_ORDER, index=config.MODEL_ORDER.index(config.PROPOSED))
st.sidebar.caption(MODEL_DESCRIPTIONS[model_name])

h_min, h_max, h_def = config.HORIZON_RANGE.get(freq, (1, 12, 6))
horizon = st.sidebar.slider(f"Horizon ({unit}s)", h_min, h_max, h_def)

min_points = max(4 * max(m, 3), 24)
latest_start = y_all.index[-min_points]
start = st.sidebar.date_input("Training start date", value=y_all.index[0].date(),
                              min_value=y_all.index[0].date(), max_value=latest_start.date())
y = y_all[pd.Timestamp(start):]
if len(y) < min_points:
    st.sidebar.warning(f"Need at least {min_points} observations; using the full series.")
    y = y_all
start_key, end_key = str(y.index[0].date()), str(y.index[-1].date())

with st.spinner(f"Fitting {model_name}…"):
    model = fit_model(series_key, model_name, start_key, end_key, y, freq)
    ar = model if model_name == config.PROPOSED else fit_model(series_key, config.PROPOSED, start_key, end_key, y, freq)
    fc = model.forecast(horizon)

st.sidebar.divider()
st.sidebar.caption(f"{len(y):,} {unit}s · {y.index[0].date()} → {y.index[-1].date()}")

# --------------------------------------------------------------------------
# Header + tabs
# --------------------------------------------------------------------------
st.title("AeroCast")
st.caption(f"{series_label} · model **{model_name}** · {horizon}-{unit} horizon")

tabs = st.tabs(["📈 Forecast", "🧭 How the model adapts", "🏁 Model comparison",
                "💥 Shock simulator", "🛫 Capacity planner", "ℹ️ About"])


def fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


# ===== 1. Forecast ========================================================
with tabs[0]:
    state = ar.current_state()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Last actual ({y.index[-1].date()})", fmt(y.iloc[-1]), border=True)
    nxt = fc["mean"].iloc[0]
    k2.metric(f"Next {unit} forecast ({fc.index[0].date()})", fmt(nxt),
              f"{100 * (nxt / y.iloc[-1] - 1):+.1f}% vs last", border=True)
    k3.metric(f"Total over next {horizon} {unit}s", fmt(fc["mean"].sum()),
              f"80%: {fmt(fc['lo80'].sum())} – {fmt(fc['hi80'].sum())}", delta_color="off", delta_arrow="off", border=True)
    k4.metric("Current regime (AR-Holt)", state["regime"].capitalize(),
              f"α = {state['alpha_next']:.2f} · TS = {state['ts']:+.2f}", delta_color="off", delta_arrow="off", border=True)

    hist = y.iloc[-180:]
    colour = COL[model_name]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc.hi95) + list(fc.lo95[::-1]),
                             fill="toself", fillcolor=band_rgba(colour, 0.13), line=dict(width=0),
                             name="95% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]), y=list(fc.hi80) + list(fc.lo80[::-1]),
                             fill="toself", fillcolor=band_rgba(colour, 0.26), line=dict(width=0),
                             name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="actual", line=dict(color=ACTUAL, width=1.6)))
    fig.add_trace(go.Scatter(x=[hist.index[-1], *fc.index], y=[hist.iloc[-1], *fc["mean"]],
                             name=f"{model_name} forecast", line=dict(color=colour, width=2.4),
                             mode="lines+markers", marker=dict(size=5)))
    fig.update_layout(title=f"Last {len(hist)} {unit}s and the {horizon}-{unit} forecast",
                      yaxis_title="passengers")
    show(fig, 440)

    table = fc.copy()
    table.index = table.index.date
    table.index.name = "date"
    c1, c2 = st.columns([3, 1])
    c1.dataframe(table.style.format("{:,.0f}" if y.mean() > 100 else "{:,.2f}"), width="stretch")
    c2.download_button("Download forecast (CSV)", table.to_csv().encode(), file_name=f"aerocast_{series_key}_{model_name}.csv",
                       mime="text/csv", width="stretch")
    c2.caption("Intervals are locally-adaptive conformal intervals: calibrated on the model's own past "
               "errors and scaled by its current error level, so they widen after shocks.")
    if is_tsa and y.index[-1] < pd.Timestamp.today() - pd.Timedelta(days=60):
        c2.caption(f"Data end on {y.index[-1].date()}, so the forecast starts there.")


# ===== 2. How the model adapts ===========================================
with tabs[1]:
    st.subheader("How AR-Holt adapts")
    if model_name != config.PROPOSED:
        st.caption(f"This tab always shows AR-Holt (the selected model, {model_name}, has no adaptive α).")
    c = ar.components_
    ev = event_table(c, daily=(freq == "D"))
    p = ar.params_

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("actual vs one-step forecast", "smoothing parameter α_t",
                                        "tracking signal TS_t (dotted: ±τ)"))
    fig.add_trace(go.Scatter(x=c.index, y=c.actual, name="actual", line=dict(color=ACTUAL, width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=c.index, y=c.fitted, name="one-step forecast",
                             line=dict(color=COL["AR-Holt"], width=1.2)), 1, 1)
    out = c[c.is_outlier]
    fig.add_trace(go.Scatter(x=out.index, y=out.actual, mode="markers", name="clipped outlier",
                             marker=dict(color=COL["Holt"], size=7, symbol="x")), 1, 1)
    fig.add_trace(go.Scatter(x=c.index, y=c.alpha, name="α_t", line=dict(color=COL["AR-Holt"], width=1),
                             showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=c.index, y=c.ts, name="TS_t", line=dict(color=COL["Holt"], width=1),
                             showlegend=False), 3, 1)
    for v in (p["tau"], -p["tau"]):
        fig.add_hline(y=v, line=dict(color=ACTUAL, dash="dot", width=1), row=3, col=1)
    step = pd.Timedelta(days=1) if freq == "D" else (c.index[1] - c.index[0])
    for _, e in ev[ev.type == "Regime shift"].iterrows():
        fig.add_vrect(x0=pd.Timestamp(e.start), x1=pd.Timestamp(e.end) + step, fillcolor=SHOCK,
                      line_width=0, layer="below", row="all", col=1)
    fig.update_layout(hovermode="x unified")
    show(fig, 640)
    st.caption("Shaded: regime shifts (shock regime for ≥ 3 consecutive steps).")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Calm α", f"{p['alpha_calm']:.3f}", border=True)
    a2.metric("Shock α (|TS| → 1)", f"{p['alpha_shock']:.3f}", border=True)
    a3.metric("Damping φ", f"{p['phi']:.3f}", border=True)
    a4.metric("Trend smoothing β", f"{p['beta']:.4f}", border=True)

    st.markdown("**Event table**")
    kind = st.radio("Show", ["Regime shifts", "Outliers", "All"], horizontal=True, label_visibility="collapsed")
    view = ev if kind == "All" else ev[ev.type == ("Regime shift" if kind == "Regime shifts" else "Outlier")]
    st.dataframe(view.sort_values("start", ascending=False), width="stretch", hide_index=True, height=300)

    with st.expander("What am I looking at? (plain English)", expanded=True):
        st.markdown(f"""
* AR-Holt keeps a running **level** (today's underlying demand) and **trend**. Every new data point it
  compares the forecast with the actual value and moves the level by **α × error**.
* The **tracking signal** TS is the smoothed error divided by the smoothed absolute error. Errors that
  flip sign (normal noise) keep TS near 0; errors that are persistently one-signed push TS towards ±1 —
  the model is biased and the level has shifted.
* α rises smoothly with |TS|: from **{p['alpha_calm']:.2f}** in calm periods towards **{p['alpha_shock']:.2f}**.
  So the model is steady in normal times and re-levels quickly during a shift such as COVID.
* In the **calm** regime a single huge error (Thanksgiving, a snowstorm) is **clipped** to
  {p['k']:g} × its usual size, so one odd day can't drag the level. Once |TS| ≥ {p['tau']:g} the model
  enters the **shock** regime and stops clipping, because a persistent shift is not an outlier.
""")


# ===== 3. Model comparison ===============================================
def highlight_best(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Bold + tint the best value per column (closest to nominal for coverage)."""
    style = "font-weight:700; background-color:rgba(235,104,52,0.22)"

    def best(col):
        if col.name == "cov80":
            target = (col - 0.80).abs()
        elif col.name == "cov95":
            target = (col - 0.95).abs()
        else:
            target = col
        return [style if v == target.min() else "" for v in target]

    return df.style.apply(best, axis=0).format(
        {"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "MAPE": "{:.2f}", "sMAPE": "{:.2f}", "MASE": "{:.3f}",
         "cov80": "{:.2f}", "cov95": "{:.2f}", "winkler95": "{:,.0f}"})


with tabs[2]:
    st.subheader("Model comparison")
    cols = ["MAE", "RMSE", "MAPE", "MASE", "cov80", "cov95", "winkler95"]
    if dataset == "Upload CSV":
        H = min(h_def, max(1, len(y) // 10))
        st.caption(f"Live backtest: parameters fitted on the first 80% of your data, forecasts for h = 1..{H} "
                   f"from every origin in the last 20%, seasonal indices re-estimated at every origin (v2).")
        with st.spinner("Running a quick backtest of all models…"):
            summary, by_h, dm, train_end = live_backtest(series_key + start_key, y, freq, H)
        lead = summary.set_index("model")[cols].loc[config.MODEL_ORDER]
        dm_view = dm
    else:
        version = st.radio("Seasonal indices", list(config.SEASONAL_MODES), format_func=config.SEASONAL_MODES.get,
                           horizontal=True,
                           help="v2 (default): weekday/month indices re-estimated at every forecast origin from the "
                                "26 weeks before it. v1: indices estimated once on the training data and kept fixed. "
                                "Only the seasonally adjusted models (SES, Holt, DampedHolt, TriggLeachHolt, AR-Holt) "
                                "differ between the two.")
        res = cached_results(version)
        exps = {"A": "A — normal operations (Jun 2024 → May 2025)", "B": "B — COVID stress test (Mar → Dec 2020)"} \
            if is_tsa else ({"C1": "C1 — AirPassengers, test 1958–1960"} if dataset == "AirPassengers"
                            else {"C2": "C2 — Australia, test 2007–2016"})
        exp = st.radio("Experiment", list(exps), format_func=exps.get, horizontal=True)
        s = res["metrics_summary"]
        lead = s[(s.experiment == exp) & (s.series == dataset)].set_index("model")[cols].loc[config.MODEL_ORDER]
        b = res["metrics_by_horizon"]
        by_h = b[(b.experiment == exp) & (b.series == dataset)]
        d = res["dm_tests"]
        dm_view = d[(d.experiment == exp) & (d.series == dataset)]
        st.caption("Pre-computed by `scripts/run_backtests.py`: parameters fitted once on the training span, "
                   "states rolled forward with parameters fixed, seasonal indices "
                   + ("re-estimated at every origin" if version == "rolling" else "fixed from the training span")
                   + ", forecasts from every origin; metrics averaged over horizons. "
                   "This experiment uses the full training span, not the sidebar start date.")

    st.markdown("**Leaderboard** — best value per column highlighted (coverage: closest to 0.80 / 0.95)")
    st.dataframe(highlight_best(lead), width="stretch")

    fig = go.Figure()
    for mname in config.MODEL_ORDER:
        q = by_h[by_h.model == mname]
        main = mname in ("AR-Holt", "Holt", "DampedHolt", "HoltWinters")
        fig.add_trace(go.Scatter(x=q.h, y=q.MAPE, name=mname, mode="lines+markers" if mname == "AR-Holt" else "lines",
                                 line=dict(color=COL[mname], width=3 if mname == "AR-Holt" else (2 if main else 1.2),
                                           dash="solid" if main else "dot")))
    fig.update_layout(title="MAPE by forecast horizon", xaxis_title=f"horizon h ({unit}s)", yaxis_title="MAPE (%)",
                      hovermode="x unified")
    show(fig, 400)

    st.markdown("**Diebold–Mariano tests: AR-Holt vs each model** (negative statistic = AR-Holt has lower loss; HLN-corrected)")
    st.dataframe(dm_view[["h", "loss", "vs", "n", "dm_stat", "p_value", "verdict"]]
                 .style.format({"dm_stat": "{:+.2f}", "p_value": "{:.3f}"}), width="stretch", hide_index=True)


# ===== 4. Shock simulator ================================================
with tabs[3]:
    st.subheader("Shock simulator: Holt vs AR-Holt")
    st.caption("Inject an artificial shock into a recent stretch of the data. Both models keep the "
               "parameters they learnt *before* the shock, then run through it — just like in real life.")
    sc1, sc2, sc3, sc4 = st.columns(4)
    shock_type = sc1.selectbox("Shock type", ["Sudden drop", "Gradual decline", "Sudden surge", "One-day outlier"])
    magnitude = sc2.slider("Magnitude (%)", 10, 95, 60, step=5)
    max_back = max(10, min(len(y) - min_points, {"D": 120, "W": 52, "M": 36, "Q": 12, "A": 8}.get(freq, 30)))
    duration = sc3.slider(f"Duration ({unit}s)", 1, max(2, max_back - 1),
                          min(30 if freq == "D" else 6, max_back - 1), disabled=shock_type == "One-day outlier")
    back = sc4.slider(f"Start ({unit}s before the end of data)", 2, max_back, min(60 if freq == "D" else max_back, max_back))

    n = len(y)
    s0 = n - back
    mult = np.ones(n)
    mag = magnitude / 100
    if shock_type == "Sudden drop":
        mult[s0:s0 + duration] = 1 - mag
    elif shock_type == "Sudden surge":
        mult[s0:s0 + duration] = 1 + mag
    elif shock_type == "Gradual decline":
        ramp = np.linspace(1, 1 - mag, duration + 1)[1:]
        mult[s0:s0 + duration] = ramp[:len(mult[s0:s0 + duration])]
        mult[s0 + duration:] = 1 - mag
    else:
        mult[s0] = 1 - mag
    y_shock = y * mult
    train = y.iloc[:s0]
    with st.spinner("Fitting Holt and AR-Holt on the data before the shock…"):
        m_holt = fit_model(series_key, "Holt", start_key, str(train.index[-1].date()), train, freq)
        m_ar = fit_model(series_key, "AR-Holt", start_key, str(train.index[-1].date()), train, freq)
        H7 = 7 if freq == "D" else min(3, horizon)
        paths = {nm: mod.interval_paths(y_shock, H7) for nm, mod in (("Holt", m_holt), ("AR-Holt", m_ar))}
        alpha = m_ar.paths(y_shock, 1)["run"]["alpha"]

    lo = max(0, s0 - (30 if freq == "D" else 12))
    idx = y.index
    win = slice(lo, n)

    def ahead(mean: np.ndarray, h: int) -> np.ndarray:
        """Forecast for each date made h steps earlier (NaN where unavailable)."""
        out = np.full(n, np.nan)
        out[h:] = mean[:n - h, h - 1]
        return out

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.4, 0.4, 0.2],
                        subplot_titles=("one-step-ahead forecasts", f"{H7}-step-ahead forecasts", "AR-Holt α_t"))
    for row, h in ((1, 1), (2, H7)):
        fig.add_trace(go.Scatter(x=idx[win], y=y.iloc[win], name="original data", showlegend=row == 1,
                                 line=dict(color=ACTUAL, dash="dot", width=1)), row, 1)
        fig.add_trace(go.Scatter(x=idx[win], y=y_shock.iloc[win], name="actual (with shock)", showlegend=row == 1,
                                 line=dict(color=ACTUAL, width=2)), row, 1)
        for nm in ("Holt", "AR-Holt"):
            fig.add_trace(go.Scatter(x=idx[win], y=ahead(paths[nm]["mean"], h)[win], name=nm, showlegend=row == 1,
                                     line=dict(color=COL[nm], width=2.2 if nm == "AR-Holt" else 1.8)), row, 1)
    fig.add_trace(go.Scatter(x=idx[win], y=alpha[win], name="α_t", showlegend=False,
                             line=dict(color=COL["AR-Holt"], width=1.5)), 3, 1)
    fig.add_vline(x=idx[s0], line=dict(color=ACTUAL, dash="dash", width=1))
    show(fig, 680)

    def recovery(fc1: np.ndarray, actual: np.ndarray, start: int, tol: float = 0.10, hold: int = 3):
        """Steps from the shock start until the one-step error is back within
        tol (for `hold` consecutive steps) AFTER it first left that band.
        0 = the error never left the band; None = not back by the end of data."""
        ape = np.abs(actual - fc1) / np.abs(actual)
        bad = np.flatnonzero(ape[start:] > tol)
        if len(bad) == 0:
            return 0
        for t in range(start + bad[0], len(actual) - hold + 1):
            if np.all(ape[t:t + hold] <= tol):
                return t - start
        return None

    r1, r2, r3 = st.columns(3)
    act = y_shock.to_numpy()
    rec = {nm: recovery(ahead(paths[nm]["mean"], 1), act, s0) for nm in ("Holt", "AR-Holt")}
    for col, nm in ((r1, "Holt"), (r2, "AR-Holt")):
        v = rec[nm]
        col.metric(f"{nm}: {unit}s until error back within 10%", "not within window" if v is None else f"{v}",
                   border=True)
    post = slice(s0, n)
    mape = {nm: 100 * np.nanmean(np.abs(act[post] - ahead(paths[nm]["mean"], 1)[post]) / act[post]) for nm in rec}
    r3.metric("One-step MAPE after the shock start", f"AR-Holt {mape['AR-Holt']:.1f}%",
              f"Holt {mape['Holt']:.1f}%", delta_color="off", delta_arrow="off", border=True)
    st.caption("“Back within 10%” = the one-step error stays within 10% of actual for 3 consecutive "
               f"{unit}s. What to look for: a **one-day outlier** barely moves AR-Holt (it is clipped) but "
               f"knocks Holt off for several {unit}s. On the first {unit}s of a *sudden* jump AR-Holt also "
               "clips the error (it could be an outlier) and waits for the tracking signal to confirm the "
               "shift; once confirmed, α rises and it re-levels quickly. A slow, perfectly linear decline "
               "suits Holt's undamped trend, so Holt can win there — the simulator shows whichever happens.")


# ===== 5. Capacity planner ===============================================
with tabs[4]:
    st.subheader(f"Checkpoint staffing planner — {BUILTIN[dataset]}" if is_tsa else "Flight / seat planner")
    q_label = st.radio("Plan for demand at quantile", list(QUANTILES), index=1, horizontal=True,
                       help="P50 = the central forecast; P80 = demand is expected to be at or below this "
                            "on 4 days out of 5 (interpolated from the forecast intervals).")
    demand = forecast_at_quantile(fc, QUANTILES[q_label])
    if is_tsa:
        i1, i2 = st.columns(2)
        lane_tp = i1.number_input("Assumption: passengers per lane per hour", 50, 400, config.LANE_THROUGHPUT, step=10)
        opl = i2.number_input("Assumption: officers per lane", 1, 6, config.OFFICERS_PER_LANE)
        st.caption("These are illustrative planning assumptions, not official TSA figures — edit them freely.")
        prof = cached_profile().query("airport == @dataset")
        plan = staffing_plan(demand, prof, lane_throughput=lane_tp, officers_per_lane=int(opl))
        grid = plan.pivot(index="date", columns="hour", values="lanes")
        ylabels = [d.strftime("%a %d %b") for d in grid.index]
        fig = go.Figure(go.Heatmap(z=grid.values, x=[f"{h:02d}:00" for h in grid.columns], y=ylabels,
                                   colorscale="Blues", colorbar=dict(title="lanes"),
                                   text=grid.values, texttemplate="%{text}",
                                   hovertemplate="%{y} %{x}<br>lanes: %{z}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title=f"Security lanes needed per hour ({q_label} demand)", hovermode="closest")
        show(fig, 120 + 34 * len(grid))
        t1, t2, t3 = st.columns(3)
        t1.metric("Peak lanes in one hour", int(plan.lanes.max()), border=True)
        t2.metric("Total officer-hours", f"{plan.officers.sum():,}", border=True)
        t3.metric(f"{q_label} passengers over the window", f"{demand.sum():,.0f}", border=True)
        st.markdown("**Peak hours**")
        pk = peak_hours(plan, 10).assign(date=lambda d: d.date.dt.date)
        st.dataframe(pk.style.format({"pax_per_hour": "{:,.0f}"}), width="stretch", hide_index=True)
    else:
        f1, f2, f3 = st.columns(3)
        default_scale = UNIT_SCALE.get(dataset, 1)
        scale = f1.selectbox("One unit of the series equals", [1, 1_000, 1_000_000],
                             index=[1, 1_000, 1_000_000].index(default_scale), format_func=lambda v: f"{v:,} passengers")
        lf = f2.slider("Target load factor", 0.50, 1.00, config.LOAD_FACTOR, 0.01)
        seats = f3.number_input("Seats per aircraft", 50, 600, config.SEATS_PER_AIRCRAFT, step=10)
        st.caption("Load factor and aircraft size are assumptions — edit them freely.")
        plan = flight_plan(demand * scale, load_factor=lf, seats_per_aircraft=int(seats))
        fig = go.Figure(go.Bar(x=plan.index, y=plan.flights, marker_color=COL["Holt"], name="flights"))
        fig.update_layout(title=f"Flights needed per {unit} ({q_label} demand)", yaxis_title="flights",
                          hovermode="closest")
        show(fig, 360)
        tbl = plan.copy()
        tbl.index = tbl.index.date
        st.dataframe(tbl.style.format({"demand": "{:,.0f}", "seats_needed": "{:,.0f}", "flights": "{:,}"}),
                     width="stretch")


# ===== 6. About ==========================================================
with tabs[5]:
    team = config.current_team()
    st.subheader(config.PROJECT_TITLE)
    st.markdown(f"**{team['course']}** — {', '.join(team['members'])}")
    st.markdown(
        "AeroCast forecasts short-term air passenger demand with **AR-Holt**, an *adaptive robust* "
        "version of Holt's linear trend method. It combines the tracking signal (Trigg & Leach 1967), "
        "smooth-transition smoothing (Taylor 2004), robust Huber-type cleaning (Gelper, Fried & Croux 2010) "
        "and the damped trend (Gardner & McKenzie 1985). Seasonality is removed first with multiplicative "
        "indices and put back afterwards; by default (v2) the indices are re-estimated at every forecast "
        "origin from the 26 weeks before it, so the weekday pattern can drift.")
    st.markdown("**The AR-Holt update, every time step**")
    st.latex(r"\hat x_t=\ell_{t-1}+\phi b_{t-1},\qquad e_t=x_t-\hat x_t")
    st.latex(r"\tilde e_t=\begin{cases}e_t & |TS_{t-1}|\ge\tau\ \text{(shock)}\\ "
             r"\sigma_{t-1}\,\mathrm{clip}(e_t/\sigma_{t-1},-k,k) & \text{otherwise (calm)}\end{cases}")
    st.latex(r"\alpha_t=\mathrm{expit}\big(a+b\,|TS_{t-1}|\big)")
    st.latex(r"\ell_t=\hat x_t+\alpha_t\tilde e_t,\qquad b_t=\phi b_{t-1}+\alpha_t\beta\,\tilde e_t")
    st.latex(r"\sigma_t=\lambda|\tilde e_t|+(1-\lambda)\sigma_{t-1}")
    st.latex(r"E_t=\gamma\tilde e_t+(1-\gamma)E_{t-1},\quad M_t=\gamma|\tilde e_t|+(1-\gamma)M_{t-1},"
             r"\quad TS_t=E_t/M_t")
    st.latex(r"\hat x_{n+h}=\ell_n+(\phi+\phi^2+\dots+\phi^h)\,b_n")
    st.markdown(
        f"Fitted: $a$, $b\\ge0$, $\\beta$, $\\phi\\in[0.80,1]$ (one-step SSE, Nelder–Mead). "
        f"Fixed: $\\gamma={config.AR_HOLT_HYPER['gamma']}$, $k={config.AR_HOLT_HYPER['k']}$, "
        f"$\\lambda={config.AR_HOLT_HYPER['lam']}$, $\\tau={config.AR_HOLT_HYPER['tau']}$. "
        "Prediction intervals are locally-adaptive split-conformal intervals.")
    st.markdown("**Data sources and credits**")
    st.markdown(
        "* U.S. Transportation Security Administration — checkpoint throughput, FOIA Electronic Reading Room "
        "(public domain); parsed to CSV by the open-source project "
        "[mikelor/TsaThroughput](https://github.com/mikelor/TsaThroughput).\n"
        "* AirPassengers (Box & Jenkins) and Australian air passengers (Hyndman & Athanasopoulos, *FPP3*), via "
        "[vincentarelbundock/Rdatasets](https://github.com/vincentarelbundock/Rdatasets).\n"
        "* Built with Python, NumPy, pandas, SciPy, statsmodels, Plotly and Streamlit.")
    st.markdown(f"**Source code:** [{config.GITHUB_URL}]({config.GITHUB_URL})")
    other = [c for c in config.TEAMS if c != config.current_course()]
    st.caption(f"This deployment shows the {config.current_course().upper()} team (PROJECT_COURSE). "
               f"The same code base is also submitted for {', '.join(o.upper() for o in other)}.")

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
team = config.current_team()
st.caption(f"{config.PROJECT_TITLE} · {team['course']} · {', '.join(team['members'])}")
