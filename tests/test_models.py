"""Tests for seasonal adjustment, the models, intervals and capacity maths."""

import math

import numpy as np
import pandas as pd
import pytest

from airdemand.capacity import flight_plan, forecast_at_quantile, staffing_plan
from airdemand.events import event_table
from airdemand.models import (MODELS, AdaptiveRobustHolt, DampedHolt, Holt,
                              HoltWinters, holt_filter)
from airdemand.seasonal import SeasonalAdjuster


# 1. Seasonal adjust -> readjust round trip ---------------------------------
def test_seasonal_round_trip(daily_series, monthly_series):
    for y, m in [(daily_series, 7), (monthly_series, 12)]:
        sa = SeasonalAdjuster(m).fit(y)
        back = sa.readjust(sa.adjust(y))
        np.testing.assert_allclose(back.to_numpy(), y.to_numpy(), rtol=1e-12)
        assert abs(sa.indices_.mean() - 1) < 1e-12  # normalised to mean 1


def test_rolling_round_trip(daily_series, monthly_series):
    for y, m, w in [(daily_series, 7, 26 * 7), (monthly_series, 12, None)]:
        sa = SeasonalAdjuster(m, window=w, mode="rolling").fit(y)
        back = sa.readjust_from(y, sa.adjust(y))
        np.testing.assert_allclose(back.to_numpy(), y.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("which,m,window", [("daily_series", 7, 26 * 7), ("monthly_series", 12, None)])
def test_rolling_indices_equal_refit_on_window_before_origin(which, m, window, request):
    """Row t of the rolling indices = a fresh fit on ONLY the window ending at t."""
    y = request.getfixturevalue(which)
    S = SeasonalAdjuster(m, window=window, mode="rolling").origin_indices(y)
    for t in (4 * m - 1, len(y) // 2, len(y) - 1):
        past = y.iloc[:t + 1]
        refit = SeasonalAdjuster(m, window=window).fit(past).indices_
        np.testing.assert_allclose(S[t], refit, rtol=1e-12)
        assert abs(S[t].mean() - 1) < 1e-12


def test_rolling_indices_use_no_future_data(daily_series):
    sa = SeasonalAdjuster(7, window=26 * 7, mode="rolling")
    t = 300
    changed = daily_series.copy()
    changed.iloc[t + 1:] *= np.linspace(0.5, 2.0, len(changed) - t - 1)
    np.testing.assert_allclose(sa.origin_indices(daily_series)[:t + 1],
                               sa.origin_indices(changed)[:t + 1], rtol=1e-12)


def test_rolling_and_fixed_agree_at_the_last_training_origin(daily_series):
    """At the first backtest origin both versions use the same 26 weeks."""
    train = daily_series[:"2024-01-31"]
    fixed = AdaptiveRobustHolt(seasonal="fixed").fit(train, freq="D")
    rolling = AdaptiveRobustHolt(seasonal="rolling").fit(train, freq="D")
    t = len(train) - 1
    np.testing.assert_allclose(fixed.paths(daily_series, 7)["factor"][t],
                               rolling.paths(daily_series, 7)["factor"][t], rtol=1e-12)
    # ...but later origins re-estimate the indices in the rolling version only
    assert not np.allclose(fixed.paths(daily_series, 7)["factor"][t + 60],
                           rolling.paths(daily_series, 7)["factor"][t + 60])


def test_seasonal_indices_recover_pattern(daily_series):
    sa = SeasonalAdjuster(7).fit(daily_series)
    by_day = sa.index_table(daily_series.index)
    # Synthetic Sunday factor is 1.15 and Saturday 0.85 (before normalising).
    assert by_day["Sunday"] > by_day["Saturday"] + 0.2


# 2. From-scratch Holt ~ statsmodels Holt -----------------------------------
def test_holt_matches_statsmodels(annual_series):
    from statsmodels.tsa.holtwinters import Holt as SMHolt
    model = Holt().fit(annual_series, freq="A")
    x = annual_series.to_numpy()
    # (a) identical recursion: same alpha, beta and initial states
    sm = SMHolt(x, initialization_method="known", initial_level=model.l0_,
                initial_trend=model.b0_).fit(smoothing_level=model.params_["alpha"],
                                             smoothing_trend=model.params_["beta"],
                                             optimized=False)
    _, run = holt_filter(x, model.params_["alpha"], model.params_["beta"], 1.0,
                         model.l0_, model.b0_)
    np.testing.assert_allclose(run["xhat"], sm.fittedvalues, rtol=1e-10)
    np.testing.assert_allclose(model.forecast(5)["mean"].to_numpy(), sm.forecast(5), rtol=1e-10)
    # (b) when both optimise alpha/beta themselves, forecasts are close
    sm_opt = SMHolt(x, initialization_method="known", initial_level=model.l0_,
                    initial_trend=model.b0_).fit()
    np.testing.assert_allclose(model.forecast(5)["mean"].to_numpy(), sm_opt.forecast(5), rtol=0.03)


def test_holt_winters_roll_forward_equals_refit(monthly_series):
    """Running statsmodels once with fixed params == re-running at each origin."""
    train = monthly_series[:"1957-12"]
    hw = HoltWinters().fit(train, freq="M")
    paths = hw.paths(monthly_series, 12)
    for t in (len(train) - 1, len(train) + 10):
        trunc = hw._sm_fixed(monthly_series.to_numpy()[:t + 1])
        np.testing.assert_allclose(paths["adj"][t], trunc.forecast(12), rtol=1e-8)


# 3. Nesting: AR-Holt with b=0, k=inf is DampedHolt ---------------------------
def test_ar_holt_nests_damped_holt(daily_series):
    damped = DampedHolt().fit(daily_series, freq="D")
    p = damped.params_
    ar = AdaptiveRobustHolt(fix_b=0.0, k=np.inf).fit(daily_series, freq="D")  # rolling indices (default)
    # Put DampedHolt's parameters into AR-Holt: a = logit(alpha).
    ar.params_.update({"a": math.log(p["alpha"] / (1 - p["alpha"])), "b": 0.0,
                       "beta": p["beta"], "phi": p["phi"]})
    x = ar.adjuster_.adjust(daily_series).to_numpy()
    run_ar = ar._run(x)
    _, run_dh = holt_filter(x, p["alpha"], p["beta"], p["phi"], damped.l0_, damped.b0_)
    np.testing.assert_allclose(run_ar["level"], run_dh["level"], rtol=1e-10)
    np.testing.assert_allclose(run_ar["trend"], run_dh["trend"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(run_ar["xhat"], run_dh["xhat"], rtol=1e-10)


# 4. No look-ahead leakage ----------------------------------------------------
@pytest.mark.parametrize("seasonal", ["rolling", "fixed"])
@pytest.mark.parametrize("name", list(MODELS))
def test_no_lookahead(name, seasonal, daily_series):
    train = daily_series[:"2024-01-31"]
    model = MODELS[name](seasonal=seasonal).fit(train, freq="D")
    a = model.interval_paths(daily_series, 7)
    origin = len(train) + 30
    changed = daily_series.copy()
    changed.iloc[origin + 1:] *= 3.0  # wreck the future
    b = model.interval_paths(changed, 7)
    for col in ("mean", "lo80", "hi80", "lo95", "hi95"):
        np.testing.assert_allclose(a[col][:origin + 1], b[col][:origin + 1], rtol=1e-12)


# 5. Finite forecasts, ordered intervals -------------------------------------
@pytest.mark.parametrize("seasonal", ["rolling", "fixed"])
@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("which,freq,h", [("daily_series", "D", 14),
                                          ("monthly_series", "M", 24),
                                          ("annual_series", "A", 10)])
def test_forecasts_finite_and_ordered(name, which, freq, h, seasonal, request):
    y = request.getfixturevalue(which)
    fc = MODELS[name](seasonal=seasonal).fit(y, freq=freq).forecast(h)
    assert len(fc) == h
    assert np.isfinite(fc.to_numpy()).all()
    assert (fc["lo95"] <= fc["lo80"] + 1e-9).all()
    assert (fc["lo80"] <= fc["mean"] + 1e-9).all()
    assert (fc["mean"] <= fc["hi80"] + 1e-9).all()
    assert (fc["hi80"] <= fc["hi95"] + 1e-9).all()
    assert (fc["lo95"] >= 0).all()


# 6. Synthetic -60% level shift ---------------------------------------------------
def synthetic_shock_series(seed: int = 0, ramp_days: int = 7) -> pd.Series:
    """500 days: a history with a few level steps and holiday-like outliers,
    then a -60% collapse starting at day 400 spread over `ramp_days` days
    (COVID-like: TSA traffic fell over ~3 weeks in March 2020)."""
    rng = np.random.default_rng(seed)
    n = 500
    level = np.full(n, 10_000.0)
    level[100:] *= 1.25
    level[220:] *= 0.70
    level[330:] *= 1.20
    drop = np.ones(n)
    drop[400:] = 0.4
    drop[400:400 + ramp_days] = np.linspace(1, 0.4, ramp_days + 1)[1:]
    y = level * drop * (1 + rng.normal(0, 0.02, n))
    spikes = rng.choice(np.r_[10:395, 440:500], 15, replace=False)
    y[spikes] *= rng.choice([0.6, 1.35], 15)
    return pd.Series(y, index=pd.date_range("2022-01-01", periods=n, freq="D"))


def test_ar_holt_adapts_to_level_shift():
    """Fit on the history only, then roll through the unseen -60% shock with
    parameters fixed (exactly the backtest protocol)."""
    y = synthetic_shock_series()
    train = y.iloc[:400]
    holt = Holt().fit(train, freq="D")
    ar = AdaptiveRobustHolt().fit(train, freq="D")
    p_ar, p_holt = ar.paths(y, 1), holt.paths(y, 1)
    alpha = p_ar["run"]["alpha"]
    # alpha_t rises during the collapse and the shock regime switches on
    assert alpha[401:415].max() > ar.params_["alpha_calm"] + 0.2
    assert p_ar["run"]["shock"][401:415].any()
    # post-shift one-step error (days 1-30) is lower than Holt's
    err_ar = np.abs(p_ar["x"][401:431] - p_ar["adj"][400:430, 0]).mean()
    err_holt = np.abs(p_holt["x"][401:431] - p_holt["adj"][400:430, 0]).mean()
    assert err_ar < err_holt


def test_isolated_outlier_is_clipped_not_a_regime_shift(daily_series):
    y = daily_series.copy()
    y.iloc[400] *= 0.3  # a one-day "snowstorm"
    ar = AdaptiveRobustHolt().fit(y, freq="D")
    ev = event_table(ar.components_)
    hit = ev[pd.to_datetime(ev["start"]) == y.index[400]]
    assert len(hit) == 1 and hit["type"].iloc[0] == "Outlier"


# 7. Capacity maths on a hand-computed example -------------------------------------
def test_staffing_plan_hand_example():
    # One Monday with 10,000 passengers; 20% in hour 6, 1.5% in hour 0.
    shares = np.full(24, 0.0)
    shares[6], shares[0] = 0.20, 0.015
    shares[1:6] = (1 - 0.215) / 5
    profile = pd.DataFrame({"weekday": 0, "hour": range(24), "share": shares})
    demand = pd.Series([10_000.0], index=[pd.Timestamp("2025-06-02")])  # a Monday
    plan = staffing_plan(demand, profile, lane_throughput=150, officers_per_lane=3)
    h6 = plan[plan["hour"] == 6].iloc[0]
    assert h6["pax_per_hour"] == pytest.approx(2000)  # 10,000 x 0.20
    assert h6["lanes"] == 14  # ceil(2000/150) = ceil(13.33)
    assert h6["officers"] == 42  # 14 x 3
    h0 = plan[plan["hour"] == 0].iloc[0]
    assert h0["lanes"] == 1 and h0["officers"] == 3  # ceil(150/150) = 1 exactly
    assert plan[plan["hour"] == 20]["lanes"].iloc[0] == 0


def test_flight_plan_and_quantiles():
    fp = flight_plan(pd.Series([1530.0, 1531.0]), load_factor=0.85, seats_per_aircraft=180)
    assert fp["seats_needed"].iloc[0] == pytest.approx(1800)  # 1530 / 0.85
    assert list(fp["flights"]) == [10, 11]  # 1800/180 = 10 exactly; 1801.2/180 -> 11
    fc = pd.DataFrame({"lo95": [80.0], "lo80": [90.0], "mean": [100.0], "hi80": [140.0], "hi95": [170.0]})
    assert forecast_at_quantile(fc, 0.5).iloc[0] == pytest.approx(100)
    assert forecast_at_quantile(fc, 0.8).iloc[0] == pytest.approx(130)  # 100 + 0.75*40
    assert forecast_at_quantile(fc, 0.95).iloc[0] == pytest.approx(160)  # 140 + (2/3)*30
