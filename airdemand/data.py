"""Loading the built-in datasets and parsing user-uploaded CSV files.

Every loader returns a `pd.Series` with a regular DatetimeIndex (with `.freq`
set) plus a one-letter frequency code:

    "D" daily, "W" weekly, "M" monthly, "Q" quarterly, "A" annual.

The frequency code decides the seasonal period, the horizon range, etc.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import config

# Pandas frequency alias for each of our frequency codes.
PANDAS_FREQ = {"D": "D", "W": "W", "M": "MS", "Q": "QS", "A": "YS"}

# Seasonal period per frequency code (weekly data is treated as non-seasonal
# because 52 weeks of history per season would be needed).
PERIOD = {"D": 7, "W": 1, "M": 12, "Q": 4, "A": 1}

BUILTIN = {
    **{code: config.AIRPORT_NAMES[code] for code in config.AIRPORTS},
    config.ALL7: config.AIRPORT_NAMES[config.ALL7],
    "AirPassengers": "AirPassengers (monthly, 1949–1960)",
    "Australia": "Australia air passengers (annual, 1970–2016)",
}


def season_period(freq: str) -> int:
    """Seasonal period m for a frequency code (1 means no seasonality)."""
    return PERIOD.get(freq, 1)


# --------------------------------------------------------------------------
# Built-in data (processed CSVs in data/, produced by scripts/prepare_data.py)
# --------------------------------------------------------------------------
def load_tsa_daily() -> pd.DataFrame:
    """Long table: date, airport, passengers, imputed."""
    df = pd.read_csv(config.TSA_DAILY_CSV, parse_dates=["date"])
    return df


def load_hourly_profile() -> pd.DataFrame:
    """Long table: airport, weekday (0=Mon), hour, share."""
    return pd.read_csv(config.TSA_HOURLY_CSV)


def load_series(name: str) -> tuple[pd.Series, str]:
    """Return (series, freq_code) for a built-in dataset name."""
    if name in config.AIRPORTS or name == config.ALL7:
        df = load_tsa_daily()
        s = df[df["airport"] == name].set_index("date")["passengers"].astype(float)
        s = s.asfreq("D")
        s.name = name
        return s, "D"
    if name == "AirPassengers":
        df = pd.read_csv(config.AIRPASSENGERS_CSV, parse_dates=["date"])
        s = df.set_index("date")["passengers"].astype(float).asfreq("MS")
        s.name = name
        return s, "M"
    if name == "Australia":
        df = pd.read_csv(config.AUS_CSV, parse_dates=["date"])
        s = df.set_index("date")["passengers_millions"].astype(float).asfreq("YS")
        s.name = name
        return s, "A"
    raise KeyError(f"Unknown dataset: {name}")


def imputed_flags(name: str) -> pd.Series:
    """For TSA series: 1 where the value was imputed (missing in the source)."""
    df = load_tsa_daily()
    s = df[df["airport"] == name].set_index("date")["imputed"]
    return s


# --------------------------------------------------------------------------
# Uploaded CSVs
# --------------------------------------------------------------------------
class UploadError(ValueError):
    """Raised with a friendly, user-facing message when an upload is invalid."""


def _freq_code_from_alias(alias: str | None) -> str | None:
    """Map a pandas frequency alias (e.g. 'MS', 'W-SUN', 'YE-DEC') to our code."""
    if alias is None:
        return None
    a = alias.upper()
    if a.startswith("D"):
        return "D"
    if a.startswith("W"):
        return "W"
    if a.startswith(("MS", "ME", "M")) and not a.startswith("MIN"):
        return "M"
    if a.startswith(("QS", "QE", "Q")):
        return "Q"
    if a.startswith(("YS", "YE", "Y", "AS", "A")):
        return "A"
    return None


def parse_uploaded_csv(content: bytes | str) -> tuple[pd.Series, str]:
    """Parse a two-column CSV (date, value) into (series, freq_code).

    Raises UploadError with a plain-English message for every problem we can
    detect, so the app can show it instead of a stack trace.
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception as exc:  # pandas raises many different error types
        raise UploadError(f"Could not read the file as CSV ({exc}).") from exc

    if df.shape[1] < 2:
        raise UploadError("The CSV needs two columns: a date column and a value column.")
    df = df.iloc[:, :2]
    df.columns = ["date", "value"]

    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    if dates.isna().mean() > 0.05:
        raise UploadError("The first column doesn't look like dates (e.g. 2024-01-31).")
    values = pd.to_numeric(df["value"], errors="coerce")
    if values.isna().mean() > 0.05:
        raise UploadError("The second column must be numbers.")

    s = pd.Series(values.to_numpy(), index=dates).dropna()
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < 12:
        raise UploadError(f"Need at least 12 observations; this file has {len(s)}.")
    if (s < 0).any():
        raise UploadError("Values must be non-negative (passenger counts).")

    alias = pd.infer_freq(s.index) if len(s) >= 3 else None
    code = _freq_code_from_alias(alias)
    if code is None:
        # infer_freq fails if there are gaps — fall back to the median spacing.
        step = pd.Series(s.index).diff().dt.days.median()
        if step <= 1.5:
            code = "D"
        elif 6 <= step <= 8:
            code = "W"
        elif 27 <= step <= 32:
            code = "M"
        elif 88 <= step <= 93:
            code = "Q"
        elif 360 <= step <= 370:
            code = "A"
        else:
            raise UploadError("Could not detect a regular frequency (daily, weekly, monthly, quarterly or annual).")

    # Put the series on a regular grid and fill short gaps by interpolation.
    if code == "W":
        s = s.resample("W").sum(min_count=1)
    else:
        s = s.resample(PANDAS_FREQ[code]).mean()
    n_gaps = int(s.isna().sum())
    if n_gaps > 0.1 * len(s):
        raise UploadError(f"Too many missing periods ({n_gaps} of {len(s)}).")
    s = s.interpolate(limit_direction="both")
    if not np.isfinite(s).all() or s.sum() <= 0:
        raise UploadError("The values could not be converted to a usable series.")
    s.name = "Uploaded"
    return s.astype(float), code
