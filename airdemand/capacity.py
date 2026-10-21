"""The real-world application: turning a demand forecast into resources.

1. Checkpoint staffing planner (airports)
   daily forecast at a chosen quantile  x  hourly profile  -> passengers/hour
   lanes    = ceil(passengers per hour / lane throughput)
   officers = lanes x officers per lane

2. Flight / seat planner (any series)
   seats   = demand / target load factor
   flights = ceil(seats / seats per aircraft)

The default throughput, staffing and aircraft numbers are ASSUMPTIONS for
illustration (editable in the app), not official TSA or airline figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Which quantile each forecast column represents. An 80% central interval
# runs from the 10th to the 90th percentile, a 95% one from 2.5th to 97.5th;
# the point forecast is treated as the median.
_QUANTILE_COLS = [(0.025, "lo95"), (0.10, "lo80"), (0.50, "mean"), (0.90, "hi80"), (0.975, "hi95")]
QUANTILES = {"P50": 0.50, "P80": 0.80, "P95": 0.95}


def forecast_at_quantile(fc: pd.DataFrame, q: float) -> pd.Series:
    """Linearly interpolate the forecast bands at quantile q (0..1).

    e.g. P80 lies between the median (0.50) and hi80 (0.90):
         value = mean + (0.80-0.50)/(0.90-0.50) * (hi80 - mean)
    """
    levels = np.array([lv for lv, _ in _QUANTILE_COLS])
    values = fc[[col for _, col in _QUANTILE_COLS]].to_numpy()
    out = np.array([np.interp(q, levels, row) for row in values])
    return pd.Series(out, index=fc.index, name=f"P{round(q * 100)}")


def staffing_plan(daily_demand: pd.Series, profile: pd.DataFrame,
                  lane_throughput: float = config.LANE_THROUGHPUT,
                  officers_per_lane: int = config.OFFICERS_PER_LANE) -> pd.DataFrame:
    """Hour-by-hour lanes and officers for each forecast day.

    daily_demand : passengers per day (index = dates)
    profile      : rows weekday (0=Mon), hour, share for ONE airport
    Returns columns date, weekday, hour, pax_per_hour, lanes, officers.
    """
    prof = profile.set_index(["weekday", "hour"])["share"]
    rows = []
    for day, pax in daily_demand.items():
        wd = day.dayofweek
        for hour in range(24):
            share = float(prof.get((wd, hour), 0.0))
            pph = pax * share
            lanes = int(np.ceil(pph / lane_throughput - 1e-9)) if pph > 0 else 0
            rows.append({"date": day, "weekday": day.day_name(), "hour": hour,
                         "pax_per_hour": pph, "lanes": lanes, "officers": lanes * officers_per_lane})
    return pd.DataFrame(rows)


def peak_hours(plan: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """The busiest hours of the plan (most lanes needed)."""
    return (plan.sort_values(["lanes", "pax_per_hour"], ascending=False)
            .head(top).reset_index(drop=True))


def flight_plan(demand: pd.Series, load_factor: float = config.LOAD_FACTOR,
                seats_per_aircraft: int = config.SEATS_PER_AIRCRAFT) -> pd.DataFrame:
    """Seats and flights needed to carry the forecast demand."""
    seats = demand / load_factor
    flights = np.ceil(seats / seats_per_aircraft - 1e-9).astype(int)
    return pd.DataFrame({"demand": demand, "seats_needed": seats, "flights": flights})
