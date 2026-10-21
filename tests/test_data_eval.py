"""Tests for data loading, CSV upload parsing and the evaluation code."""

import numpy as np
import pandas as pd
import pytest

from airdemand import config
from airdemand.data import UploadError, load_series, parse_uploaded_csv
from airdemand.evaluation import diebold_mariano, run_backtest, winkler


def test_processed_tsa_data():
    df = pd.read_csv(config.TSA_DAILY_CSV, parse_dates=["date"])
    assert set(df["airport"]) == set(config.AIRPORTS) | {config.ALL7}
    imputed = df[(df["airport"] == "ATL") & (df["imputed"] == 1)]["date"].dt.strftime("%Y-%m-%d")
    assert list(imputed) == ["2022-07-02"] + [f"2024-11-{d}" for d in range(17, 24)]
    y, freq = load_series("ALL7")
    assert freq == "D" and y.isna().sum() == 0 and y.index.freqstr == "D"
    prof = pd.read_csv(config.TSA_HOURLY_CSV)
    np.testing.assert_allclose(prof.groupby(["airport", "weekday"])["share"].sum(), 1.0, atol=1e-6)


def test_upload_parsing_monthly_and_errors():
    csv = "month,pax\n" + "\n".join(f"2020-{m:02d}-01,{100 + m}" for m in range(1, 13)) + \
          "\n" + "\n".join(f"2021-{m:02d}-01,{120 + m}" for m in range(1, 13))
    s, freq = parse_uploaded_csv(csv.encode())
    assert freq == "M" and len(s) == 24
    with pytest.raises(UploadError):
        parse_uploaded_csv(b"only_one_column\n1\n2\n3")
    with pytest.raises(UploadError):
        parse_uploaded_csv(b"date,value\nhello,1\nworld,2\n")
    with pytest.raises(UploadError):
        parse_uploaded_csv(b"date,value\n2020-01-01,1\n2020-01-02,2\n")  # too short


def test_winkler_score():
    # inside: width only; below: width + (2/0.05) x miss
    assert winkler(np.array([5.0]), np.array([0.0]), np.array([10.0]))[0] == 10
    assert winkler(np.array([-1.0]), np.array([0.0]), np.array([10.0]))[0] == 10 + 40


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(3)
    e_good, e_bad = rng.normal(0, 1, 300), rng.normal(0, 2, 300)
    r = diebold_mariano(e_good, e_bad, h=1)
    assert r["dm_stat"] < 0 and r["p_value"] < 0.01
    r_same = diebold_mariano(e_good, rng.normal(0, 1, 300), h=7)
    assert r_same["p_value"] > 0.01


def test_backtest_targets_stay_in_test_period(daily_series):
    fc, _ = run_backtest(daily_series, "D", "2024-01-31", "2024-03-31", H=7)
    assert fc["target"].min() > pd.Timestamp("2024-01-31")
    assert fc["target"].max() <= pd.Timestamp("2024-03-31")
    assert fc["origin"].min() == pd.Timestamp("2024-01-31")
