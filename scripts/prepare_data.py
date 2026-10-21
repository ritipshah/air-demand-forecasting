"""Turn the raw downloads in raw/ into the small processed CSVs in data/.

Run from the repository root:

    python scripts/prepare_data.py

Outputs
-------
data/tsa_airports_daily.csv     date, airport, passengers, imputed
data/tsa_hourly_profile.csv     airport, weekday, hour, share
data/airpassengers_monthly.csv  date, passengers
data/aus_airpassengers_annual.csv  date, passengers_millions
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airdemand import config  # noqa: E402


def read_tsa_hourly(code: str) -> pd.DataFrame:
    """Hourly totals for one airport: columns date, hour, passengers.

    Each raw file has one column per checkpoint; a blank cell means no data,
    so the hourly total is the sum of the non-blank checkpoints.
    """
    path = config.TSA_RAW_DIR / f"TsaThroughput.{code}.csv"
    raw = pd.read_csv(path)
    checkpoints = raw.columns[2:]
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"]),
        "hour": pd.to_datetime(raw["Hour"], format="%H:%M:%S").dt.hour,
        "passengers": raw[checkpoints].sum(axis=1, min_count=1).fillna(0.0),
    })
    # Some files contain the same hour twice; add them up.
    return out.groupby(["date", "hour"], as_index=False)["passengers"].sum()


def daily_with_imputation(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily totals on a complete calendar; missing days filled and flagged.

    A missing day is filled with the average of the same weekday one week
    before and one week after (or whichever of the two exists).
    """
    daily = hourly.groupby("date")["passengers"].sum()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    observed = daily.copy()  # neighbours are always taken from real data
    imputed = daily.isna()
    for day in daily.index[imputed]:
        neighbours = [observed.get(day - pd.Timedelta(days=7)),
                      observed.get(day + pd.Timedelta(days=7))]
        neighbours = [v for v in neighbours if v is not None and not np.isnan(v)]
        daily[day] = float(np.mean(neighbours))
    return pd.DataFrame({"date": daily.index, "passengers": daily.to_numpy(),
                         "imputed": imputed.astype(int).to_numpy()})


def hourly_profile(hourly: pd.DataFrame, weeks: int = 52) -> pd.DataFrame:
    """Average share of the day's passengers in each hour, per weekday.

    Uses the most recent `weeks` weeks. For each day we compute hour/day
    shares, then average them per (weekday, hour) and re-normalise so the
    24 shares of each weekday add up to exactly 1.
    """
    last = hourly["date"].max()
    recent = hourly[hourly["date"] > last - pd.Timedelta(weeks=weeks)].copy()
    # Complete 24-hour grid per day (hours with no row = 0 passengers).
    grid = (recent.pivot_table(index="date", columns="hour", values="passengers", aggfunc="sum")
            .reindex(columns=range(24)).fillna(0.0))
    shares = grid.div(grid.sum(axis=1), axis=0)
    shares["weekday"] = shares.index.dayofweek
    prof = shares.groupby("weekday").mean()
    prof = prof.div(prof.sum(axis=1), axis=0)
    long = prof.reset_index().melt(id_vars="weekday", var_name="hour", value_name="share")
    long["hour"] = long["hour"].astype(int)
    return long.sort_values(["weekday", "hour"])[["weekday", "hour", "share"]]


def main() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)

    # ---- TSA airports ---------------------------------------------------
    daily_frames, profile_frames = [], []
    for code in config.AIRPORTS:
        hourly = read_tsa_hourly(code)
        d = daily_with_imputation(hourly)
        d.insert(1, "airport", code)
        daily_frames.append(d)
        p = hourly_profile(hourly)
        p.insert(0, "airport", code)
        profile_frames.append(p)
        print(f"{code}: {len(d)} days, {d['imputed'].sum()} imputed, "
              f"{d['date'].min().date()} to {d['date'].max().date()}")

    daily = pd.concat(daily_frames, ignore_index=True)
    # Combined series: sum of the 7 airports. A day is flagged as imputed if
    # any airport was imputed on it.
    all7 = (daily.groupby("date")
            .agg(passengers=("passengers", "sum"), imputed=("imputed", "max"))
            .reset_index())
    all7.insert(1, "airport", config.ALL7)
    daily = pd.concat([daily, all7], ignore_index=True)
    daily["passengers"] = daily["passengers"].round().astype(int)
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily.to_csv(config.TSA_DAILY_CSV, index=False)

    # Hourly profile for the combined series: the simple average of the 7
    # airport profiles (re-normalised so each weekday sums to 1).
    profile = pd.concat(profile_frames, ignore_index=True)
    all7_prof = profile.groupby(["weekday", "hour"], as_index=False)["share"].mean()
    all7_prof["share"] = all7_prof["share"] / all7_prof.groupby("weekday")["share"].transform("sum")
    all7_prof.insert(0, "airport", config.ALL7)
    profile = pd.concat([profile, all7_prof], ignore_index=True)
    profile["share"] = profile["share"].round(8)
    profile.to_csv(config.TSA_HOURLY_CSV, index=False)
    check = profile.groupby(["airport", "weekday"])["share"].sum()
    print(f"hourly profile: shares per (airport, weekday) sum to "
          f"{check.min():.6f}..{check.max():.6f}")

    # ---- AirPassengers (monthly, 1949-1960) -----------------------------
    ap = pd.read_csv(config.RDATASETS_DIR / "datasets" / "AirPassengers.csv")
    ap_dates = pd.date_range("1949-01-01", periods=len(ap), freq="MS")
    pd.DataFrame({"date": ap_dates.strftime("%Y-%m-%d"), "passengers": ap["value"].astype(int)}) \
        .to_csv(config.AIRPASSENGERS_CSV, index=False)
    print(f"AirPassengers: {len(ap)} months from {ap_dates[0].date()}")

    # ---- Australia annual (1970-2016) ------------------------------------
    au = pd.read_csv(config.RDATASETS_DIR / "fpp3" / "aus_airpassengers.csv")
    pd.DataFrame({"date": [f"{int(y)}-01-01" for y in au["Year"]],
                  "passengers_millions": au["Passengers"]}) \
        .to_csv(config.AUS_CSV, index=False)
    print(f"Australia: {len(au)} years, {int(au['Year'].min())}-{int(au['Year'].max())}")


if __name__ == "__main__":
    main()
