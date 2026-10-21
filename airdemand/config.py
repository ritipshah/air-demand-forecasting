"""Project-wide settings: teams, file paths and modelling defaults.

Everything that might need changing in one place lives here, so the app,
the scripts and the notebook never disagree.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Project identity
# --------------------------------------------------------------------------
PROJECT_TITLE = "Short-Term Air Passenger Demand Forecasting via Adaptive Holt's Linear Trend Method"
APP_NAME = "AeroCast"
GITHUB_URL = "https://github.com/ritipshah/air-demand-forecasting"

# The same code base is submitted for two courses with different teams.
TEAMS = {
    "atsa": {
        "course": "Applied Time Series Analysis (ATSA)",
        "members": ["Riti Shah", "Pradhi Kanani", "Shreeya"],
    },
    "vcc": {
        "course": "Virtualization & Cloud Computing (VCC)",
        "members": ["Riti Shah", "Pradhi Kanani", "Dhruv Parekh"],
    },
}


def current_course() -> str:
    """Which course the running app represents: PROJECT_COURSE env var, default 'vcc'."""
    course = os.environ.get("PROJECT_COURSE", "vcc").strip().lower()
    return course if course in TEAMS else "vcc"


def current_team() -> dict:
    """The team dict for the current course (used by the app footer and About tab)."""
    return TEAMS[current_course()]


# --------------------------------------------------------------------------
# Paths (relative to the repository root, so they work in Docker too)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"

TSA_RAW_DIR = RAW_DIR / "TsaThroughput" / "data" / "processed" / "tsa" / "throughput"
RDATASETS_DIR = RAW_DIR / "Rdatasets" / "csv"

TSA_DAILY_CSV = DATA_DIR / "tsa_airports_daily.csv"
TSA_HOURLY_CSV = DATA_DIR / "tsa_hourly_profile.csv"
AIRPASSENGERS_CSV = DATA_DIR / "airpassengers_monthly.csv"
AUS_CSV = DATA_DIR / "aus_airpassengers_annual.csv"

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
AIRPORTS = ["ATL", "DFW", "DEN", "ORD", "LAX", "JFK", "SEA"]
ALL7 = "ALL7"  # combined series = sum of the 7 airports

AIRPORT_NAMES = {
    "ATL": "Atlanta (ATL)",
    "DFW": "Dallas/Fort Worth (DFW)",
    "DEN": "Denver (DEN)",
    "ORD": "Chicago O'Hare (ORD)",
    "LAX": "Los Angeles (LAX)",
    "JFK": "New York JFK (JFK)",
    "SEA": "Seattle-Tacoma (SEA)",
    "ALL7": "All 7 hubs",
}

# Seasonal period per frequency: 7 for daily (weekday pattern), 12 for monthly,
# 1 = no seasonality for annual data.
SEASONAL_PERIOD = {"D": 7, "M": 12, "A": 1}

# Trailing window used to estimate seasonal indices (None = all training data).
SEASONAL_WINDOW = {"D": 26 * 7, "M": None, "A": None}

# Number of recent origins used to calibrate the conformal intervals
# (None = all available).
CONFORMAL_N = {"D": 365, "M": 36, "A": None}

# Allowed forecast horizons in the app: (min, max, default).
HORIZON_RANGE = {"D": (1, 28, 14), "M": (1, 24, 12), "A": (1, 10, 5)}

# --------------------------------------------------------------------------
# AR-Holt fixed hyperparameters (NOT tuned on the test set)
# --------------------------------------------------------------------------
AR_HOLT_HYPER = {
    "gamma": 0.15,  # smoothing for the tracking-signal numerator/denominator
    "k": 2.5,  # Huber clipping threshold in units of sigma
    "lam": 0.1,  # smoothing for the running error scale sigma_t
    "tau": 0.5,  # |TS| above this = "shock" regime (clipping switched off)
}

# --------------------------------------------------------------------------
# Capacity planner defaults — these are ASSUMPTIONS, not TSA facts
# --------------------------------------------------------------------------
LANE_THROUGHPUT = 150  # passengers per security lane per hour
OFFICERS_PER_LANE = 3
LOAD_FACTOR = 0.85  # target share of seats filled
SEATS_PER_AIRCRAFT = 180

MODEL_ORDER = ["Naive", "SeasonalNaive", "SES", "Holt", "DampedHolt",
               "TriggLeachHolt", "HoltWinters", "AR-Holt"]
PROPOSED = "AR-Holt"
