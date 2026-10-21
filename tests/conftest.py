"""Shared test fixtures: small synthetic and real series."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def daily_series():
    """~1.5 years of synthetic daily traffic: trend + weekday pattern + noise."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=540, freq="D")
    weekday = np.array([1.05, 0.95, 0.9, 1.0, 1.1, 0.85, 1.15])[idx.dayofweek]
    level = 10_000 + 5 * np.arange(len(idx))
    y = level * weekday * (1 + rng.normal(0, 0.03, len(idx)))
    return pd.Series(y, index=idx)


@pytest.fixture
def monthly_series():
    from airdemand.data import load_series
    return load_series("AirPassengers")[0]


@pytest.fixture
def annual_series():
    from airdemand.data import load_series
    return load_series("Australia")[0]
