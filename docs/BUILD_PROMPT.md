# Claude Code build prompt — AeroCast: Short-Term Air Passenger Demand Forecasting via Adaptive Holt's Linear Trend Method

You are building a complete university project on my laptop, in the current folder. The folder contains only this file; move it to `docs/BUILD_PROMPT.md` once the structure exists.
I'm a B.Tech Data Science student and I'm not strong at coding, so:

- write clean, well-commented code,
- after each phase, give me a 3–5 line plain-English summary of what you built,
- explain anything I'll need to defend in a viva.

Work through ALL phases without stopping. Only ask me before:

- installing system-level software (Python, Git, Docker),
- creating anything on GitHub.

Use a TODO list to track the phases.

## 0. Context

**Project title:** "Short-Term Air Passenger Demand Forecasting via Adaptive Holt's Linear Trend Method".
**App name:** AeroCast.

**The same codebase serves two courses:**

- **ATSA (Applied Time Series Analysis)**
  - Team: Riti Shah, Pradhi Kanani, Shreeya.
  - Deliverable: the project and analysis (the notebook). No deployment.
- **VCC (Virtualization & Cloud Computing)**
  - Team: Riti Shah, Pradhi Kanani, Dhruv Parekh.
  - Deliverable: the same app, containerised with Docker and deployed on AWS EC2 (done later with a separate prompt).

**Team-name handling:**

- Put both teams in `airdemand/config.py`.
- The environment variable `PROJECT_COURSE` (`vcc` | `atsa`, default `vcc`) decides which team the app footer and About tab show.
- The notebook title page always shows the ATSA team.

**Deployment contract — the later AWS prompt depends on this, so do not change it:**

- `Dockerfile` at the repo root.
- Streamlit serves on port **8501**.
- `GET /_stcore/health` returns `ok`.
- `PROJECT_COURSE=vcc` env var is honoured.
- The Python package folder is named `airdemand/`.

**Tech stack:**

- Python 3.11 or 3.12 in a `.venv`.
- numpy, pandas, scipy, statsmodels, plotly, streamlit, matplotlib, pytest, jupyter/nbconvert.
- Pin exact versions in `requirements.txt` (app runtime only) and `requirements-dev.txt` (notebook + tests).
- Detect my OS (Windows/macOS/Linux) and use matching commands.

**Golden rule:** never type a result number by hand. Every number in the README, the notebook or the app must come from code you actually ran. If a result is unflattering to the proposed model, report it honestly and explain it.

## 1. Target structure

```
air-demand-forecasting/
├── airdemand/            # the library (all modelling logic lives here)
│   ├── __init__.py
│   ├── config.py         # teams, paths, defaults
│   ├── data.py           # load built-in datasets, parse uploaded CSVs
│   ├── seasonal.py       # seasonal adjustment (pre-processing for Holt)
│   ├── models.py         # all forecasting models, common interface
│   ├── intervals.py      # conformal prediction intervals
│   ├── evaluation.py     # rolling-origin backtest, metrics, Diebold–Mariano
│   ├── events.py         # outlier vs regime-shift detection
│   └── capacity.py       # staffing / flight planning application
├── app/streamlit_app.py
├── scripts/prepare_data.py
├── scripts/run_backtests.py
├── data/                 # small processed CSVs (committed)
├── results/              # backtest outputs (committed, read by the app)
├── notebooks/ATSA_Analysis.ipynb
├── reports/figures/
├── tests/
├── deploy/DEPLOY.md
├── deploy/CLAUDE_CODE_DEPLOY_PROMPT.md
├── .streamlit/config.toml
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── README.md
└── VIVA_NOTES.md
```

## 2. Phase 1 — Data (real, public, reproducible)

### Primary dataset

U.S. TSA airport security-checkpoint passenger throughput.

- Public-domain U.S. government data from TSA's FOIA reading room.
- Parsed to CSV by the open-source repo `mikelor/TsaThroughput`.
- Hourly, per checkpoint, for 7 hub airports: ATL, DFW, DEN, ORD, LAX, JFK, SEA.

That repo is large (thousands of PDFs), so fetch only the 7 files with a partial checkout. These commands were tested. The checkout can take a few minutes. Put clones in `raw/`, which must be gitignored.

```
mkdir raw && cd raw
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/mikelor/TsaThroughput
cd TsaThroughput
git -c gc.auto=0 checkout HEAD -- data/processed/tsa/throughput/TsaThroughput.ATL.csv data/processed/tsa/throughput/TsaThroughput.DFW.csv data/processed/tsa/throughput/TsaThroughput.DEN.csv data/processed/tsa/throughput/TsaThroughput.ORD.csv data/processed/tsa/throughput/TsaThroughput.LAX.csv data/processed/tsa/throughput/TsaThroughput.JFK.csv data/processed/tsa/throughput/TsaThroughput.SEA.csv
cd ..
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/vincentarelbundock/Rdatasets
cd Rdatasets
git -c gc.auto=0 checkout HEAD -- csv/datasets/AirPassengers.csv csv/fpp3/aus_airpassengers.csv
```

### File formats (verified)

**TsaThroughput.<CODE>.csv**

- Columns: `Date` (YYYY-MM-DD), `Hour` (HH:MM:SS), then one column per checkpoint.
- A blank cell means no data.
- Data covers 2018-12-30 to 2025-05-31.
- The same 8 days are missing for every airport: 2022-07-02 and 2024-11-17 to 2024-11-23.
- The series contains genuine shocks. Do NOT remove them; they are the whole point of an adaptive model:
  - the COVID collapse (Mar–Apr 2020, about 95% drop),
  - Thanksgiving dips,
  - winter storms, e.g. DFW 2021-02-17, DEN 2021-03-14, JFK 2021-02-01 and 2022-01-29.

**AirPassengers.csv**

- Columns: `rownames,time,value`.
- 144 monthly values from 1949-01 (the Box–Jenkins classic).

**aus_airpassengers.csv**

- Columns: `rownames,Year,Passengers`.
- Annual, 1970–2016, in millions.
- This is the textbook Holt's-method example from Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*.

### `scripts/prepare_data.py` produces

1. **`data/tsa_airports_daily.csv`** with columns `date, airport, passengers, imputed`.
   - Daily total = sum over all checkpoints and hours.
   - Fill each missing day with the average of the same weekday one week before and after, and set `imputed=1`.
   - Also add a combined series `airport="ALL7"` (the sum of the 7 airports).
2. **`data/tsa_hourly_profile.csv`** with columns `airport, weekday, hour, share`.
   - The average share of the day's passengers in each hour, by airport and weekday.
   - Use the most recent 52 weeks.
   - Shares sum to 1 per (airport, weekday).
3. **`data/airpassengers_monthly.csv`** with columns `date, passengers` (month starts).
4. **`data/aus_airpassengers_annual.csv`** with columns `date, passengers_millions`.

Write a credits/data-sources section into the README.

## 3. Phase 2 — The model library (`airdemand/`)

### Common interface

Every model is a class with:

- `fit(y: pd.Series)`
- `forecast(h) -> pd.DataFrame` with columns `mean, lo80, hi80, lo95, hi95`
- `components_` (a DataFrame of internal states over time, where meaningful)
- `params_` (a dict)

Use `scipy.special.expit` for logistic functions to avoid overflow.

### `seasonal.py` — SeasonalAdjuster

Holt's linear trend method has no seasonal component, so remove seasonality first and put it back after forecasting.

- Use multiplicative indices: ratio of y to a centred moving average (2×m MA for even m).
- Each index is the **median** ratio per season position. The median makes it robust to holidays.
- Normalise the indices to mean 1.
- Use only the trailing `window` of the training data: 26 weeks for daily, all training data for monthly.
- Periods: m=7 for daily, 12 for monthly, none for annual.
- Methods: `fit`, `adjust`, `readjust`. Include a round-trip unit test.

### `models.py` — implement from scratch

SES, Holt and the adaptive models are written from scratch in numpy, NOT via statsmodels. Statsmodels is used only for the Holt-Winters benchmark and for cross-checking.

**Notation.** All smoothing models run on the seasonally adjusted series x_t. ŷ is put back on the original scale by multiplying by the seasonal index.

**1. `Naive`** — last observed value. Run on the raw series.

**2. `SeasonalNaive`** — same season last cycle. Run on the raw series.

**3. `SES`** — simple exponential smoothing on x_t.

**4. `Holt`** — classic additive Holt's linear trend:

- ℓ_t = α x_t + (1−α)(ℓ_{t−1} + b_{t−1})
- b_t = β(ℓ_t − ℓ_{t−1}) + (1−β) b_{t−1}
- x̂_{t+h} = ℓ_t + h·b_t

Fit α and β by minimising one-step-ahead SSE after a burn-in of 2m points. Use scipy minimize with Nelder–Mead, parameters mapped through logit for bounds, and 3 starting points.

Unit test: the result is close to statsmodels `Holt` on the same data with the same initialisation.

**5. `DampedHolt`** — as Holt, but b_{t−1} is replaced by φ·b_{t−1}.

- φ ∈ [0.80, 1.00]
- x̂_{t+h} = ℓ_t + (φ + φ² + … + φ^h) b_t

**6. `TriggLeachHolt` (ARR-Holt)** — the classic adaptive-response-rate idea (Trigg & Leach 1967) applied to Holt.

- E_t = γe_t + (1−γ)E_{t−1}
- M_t = γ|e_t| + (1−γ)M_{t−1}
- Tracking signal TS_t = E_t / M_t
- α_t = clip(|TS_{t−1}|, 0.05, 0.95), with γ = 0.2
- β is fitted.

This is the textbook adaptive baseline. It is known to be unstable, and that's fine.

**7. `AdaptiveRobustHolt` (AR-Holt) — THE PROPOSED METHOD.**

It is a damped Holt in error-correction form whose smoothing parameter moves smoothly between a "calm" value and a "shock" value, driven by the tracking signal. It also has regime-switching robustness:

- isolated outliers (e.g. a holiday, a snowstorm) are clipped,
- but once a persistent shift is detected (e.g. COVID), clipping switches off so the model can re-level fast.

It combines four published ideas:

- Trigg & Leach (1967): tracking signal
- Taylor (2004): smooth-transition exponential smoothing
- Gelper, Fried & Croux (2010): robust Holt–Winters
- Gardner & McKenzie (1985): damped trend

**Parameters:**

- Fitted: a, b ≥ 0 (use softplus), β ∈ (0,1), φ ∈ [0.80, 1.00].
- Fixed hyperparameters: γ=0.15, k=2.5, λ=0.1, τ=0.5.

**Initialisation:**

- ℓ_0 = mean(x_1..x_m)
- b_0 = (mean(x_{m+1..2m}) − mean(x_1..x_m)) / m
- σ_0 = 1.4826·MAD of the first differences of the first 2m points
- E_0 = 0, M_0 = σ_0, TS_0 = 0
- Use m=7 for daily, 12 for monthly, 3 for annual.

**For t = 1..n:**

```
x̂_t   = ℓ_{t−1} + φ b_{t−1}                      (one-step forecast)
e_t    = x_t − x̂_t
regime = "shock" if |TS_{t−1}| ≥ τ else "calm"
ẽ_t    = e_t                                if shock
       = σ_{t−1}·clip(e_t/σ_{t−1}, −k, k)    if calm     (Huber-type cleaning)
α_t    = expit(a + b·|TS_{t−1}|)            (smooth transition; calm α = expit(a), shock α → expit(a+b))
ℓ_t    = x̂_t + α_t ẽ_t
b_t    = φ b_{t−1} + α_t β ẽ_t
σ_t    = λ|ẽ_t| + (1−λ)σ_{t−1}             (running error scale → adaptive interval width)
E_t    = γ ẽ_t + (1−γ)E_{t−1};  M_t = γ|ẽ_t| + (1−γ)M_{t−1};  TS_t = E_t/M_t
```

**Forecast:** x̂_{n+h} = ℓ_n + (φ + … + φ^h)·b_n.

Store per step: ℓ, b, α_t, TS_t, σ_t, regime, and an `is_outlier` flag (|e_t/σ_{t−1}| > k while calm).

**Nesting test (important for the viva):** with b=0 and k=∞, AR-Holt must reproduce DampedHolt exactly.

**Parameter estimation:** minimise one-step SSE after burn-in, using Nelder–Mead with 3 starting points.

**8. `HoltWinters`** — a strong, independent benchmark: statsmodels `ExponentialSmoothing` with a damped additive trend and multiplicative seasonality (period m), on the raw series.

**Annual data (no seasonality):**

- SeasonalNaive equals Naive.
- HoltWinters falls back to statsmodels damped Holt with no seasonal component.
- MASE uses m=1.

**Model registry:** `MODELS = {name: class}`, so the app and the backtests can loop over all models.

**Sanity ballpark from a quick prototype of exactly this design** (Nelder–Mead, SSE, no refit, daily origins, mean MAPE over horizons):

- COVID stress test, ALL7, h=1..7: Holt ≈ 30%, Damped ≈ 20%, AR-Holt ≈ 15%.
- Normal year, ALL7, h=1..14: Holt ≈ 7.0%, AR-Holt ≈ 6.6%.

Being within a few points of these is fine. Being far off means look for a bug. Do NOT tune anything on the test set. If you tune the hyperparameters (γ, k, λ, τ) at all:

- use only a validation window inside the training data (its last 26 weeks),
- document it in the notebook.

### `intervals.py` — locally-adaptive split-conformal prediction intervals

For each model, use the last N training origins: N=365 for daily, 36 for monthly, all available for annual.

1. Compute h-step errors on the seasonally adjusted scale.
2. Divide each error by that model's error scale σ_t at the origin. AR-Holt has σ_t built in. For the other models, use an EWMA (λ=0.1) of their own absolute one-step errors.
3. Take lower/upper empirical quantiles per horizon: (0.10, 0.90) for 80% and (0.025, 0.975) for 95%.
4. Interval = x̂_{n+h} + q·σ_n, then multiply by the seasonal index.

Intervals therefore widen automatically in turbulent periods. Clip lower bounds at 0.

### `evaluation.py`

**Rolling-origin backtest.**

- Fit parameters once on the training span.
- Then roll the states forward through the test period with the parameters fixed (no refit). This is realistic and fast.
- At every origin, record forecasts for h=1..H. Use daily origins for daily data.
- Seasonal indices come from the training span only.
- For HoltWinters, roll forward by re-running statsmodels with the fitted parameters fixed (`optimized=False`) at each origin.

**Metrics, per model and per horizon:**

- MAE, RMSE, MAPE, sMAPE
- MASE (scaled by the in-sample seasonal-naive MAE)
- 80% and 95% interval coverage
- mean Winkler score (95%)

**Diebold–Mariano test.**

- With the Harvey–Leybourne–Newbold correction.
- Newey–West variance with lag h−1, t-distribution.
- For AR-Holt vs each other model, at h=1 and h=7, with squared and absolute loss.

### Experiments (`scripts/run_backtests.py`)

Write everything to `results/`:

- `metrics_by_horizon.csv`
- `metrics_summary.csv`
- `dm_tests.csv`
- `ablation.csv`

| Exp | Series | Train | Test | H |
|---|---|---|---|---|
| A "Normal operations" | each airport + ALL7 | start → 2024-05-31 | 2024-06-01 → 2025-05-31 | 14 |
| B "COVID stress test" | each airport + ALL7 | start → 2020-02-29 | 2020-03-01 → 2020-12-31 | 7 |
| C1 classic | AirPassengers | 1949-01 → 1957-12 | 1958-01 → 1960-12 | 12 |
| C2 classic | Australia annual | 1970 → 2006 | 2007 → 2016 | 5 |

**Ablation study (ALL7, experiments A and B).** Compare:

- DampedHolt (neither feature)
- AR-Holt with b fixed at 0 (robust only)
- AR-Holt with k=∞ (adaptive only)
- full AR-Holt

This shows which ingredient helps where.

### `events.py`

Turn AR-Holt's components into an event table with columns: `start, end, type, peak_alpha, mean_error_pct, near_holiday`.

- **type** is either:
  - "Regime shift": the shock regime lasting ≥ 3 consecutive steps, or
  - "Outlier": isolated flagged points.
- **near_holiday:** within ±2 days of a US federal holiday, using pandas `USFederalHolidayCalendar`, which includes Thanksgiving.

For the TSA data, COVID should appear as a regime shift, and Thanksgiving/storms as outliers. Check this and mention it in the notebook.

### `capacity.py` — the real-world application

1. **Checkpoint staffing planner (airports).** Take:
   - a daily forecast at a chosen quantile (P50/P80/P95, interpolated from the intervals),
   - the hourly profile for that airport and weekday.

   Compute:
   - passengers per hour,
   - security lanes needed = ceil(pax_per_hour ÷ lane_throughput),
   - officers = lanes × officers_per_lane.

   Defaults are user-adjustable assumptions, clearly labelled as such: 150 passengers/lane/hour and 3 officers per lane.
2. **Flight/seat planner (any series).** Compute:
   - seats needed = demand ÷ target load factor (default 0.85),
   - flights = ceil(seats ÷ seats per aircraft) (default 180).

## 4. Phase 3 — Tests (`tests/`, pytest)

1. Seasonal adjust → readjust round-trip.
2. From-scratch Holt ≈ statsmodels Holt.
3. AR-Holt with b=0, k=∞ equals DampedHolt (nesting).
4. **No look-ahead leakage:** changing data after the origin must not change the forecasts made at that origin.
5. Forecasts are finite and intervals are ordered (lo95 ≤ lo80 ≤ mean ≤ hi80 ≤ hi95).
6. On a synthetic series with a −60% level shift, AR-Holt's α_t rises and its post-shift error is lower than Holt's.
7. Capacity maths on a hand-computed example.

All tests must pass. Show me the pytest output.

## 5. Phase 4 — ATSA notebook (`notebooks/ATSA_Analysis.ipynb`, executed, outputs saved)

This is the ATSA deliverable, so make it read like a polished report. Use matplotlib, save figures to `reports/figures/`, and use markdown cells with LaTeX.

1. **Title page:** project title, ATSA team names, course, date.
2. **Problem statement and motivation:** why short-term demand matters to airlines and airports, and why fixed smoothing fails under shocks.
3. **Data description:** sources, coverage, the imputed days, summary statistics.
4. **EDA:**
   - all 7 airports over time,
   - the day-of-week pattern,
   - the COVID collapse and recovery,
   - an hourly-profile heatmap.
5. **Decomposition:**
   - STL on the daily ALL7 series (period 7),
   - classical multiplicative decomposition on AirPassengers.
6. **Stationarity:**
   - ADF and KPSS on raw, log and differenced series, in a results table with interpretation,
   - ACF/PACF plots.
7. **Theory:** SES → Holt → damped Holt → ARR → AR-Holt, with all equations, and why each step is added.
8. **Implementation check:** from-scratch Holt vs statsmodels, side by side.
9. **Fitted parameters and their interpretation:** calm α vs shock α, φ, β.
10. **α_t and the tracking signal through 2019–2025 (the key plot):**
    - shade the shock regimes,
    - mark the outliers,
    - show the event table.
11. **Residual diagnostics:** Ljung–Box test, histogram, Q–Q plot, ACF of residuals.
12. **Results (from `results/`):**
    - Exp A, B, C tables,
    - MAPE-by-horizon plots,
    - Diebold–Mariano table,
    - interval coverage,
    - the ablation study.
13. **Application demo:** a staffing heatmap for one airport for the next 7 days.
14. **Conclusions, limitations and future work.**
    - Limitations: no explicit holiday effects; parameters are not refitted online.
    - Future work: holiday regressors, a hierarchical airport → national model, online re-estimation.
15. **References:** Holt 1957; Gardner & McKenzie 1985; Trigg & Leach 1967; Taylor 2004; Gelper, Fried & Croux 2010; Hyndman & Athanasopoulos (FPP3); Diebold & Mariano 1995; Harvey, Leybourne & Newbold 1997.

**Execute it** with `jupyter nbconvert --to notebook --execute --inplace`, and also export `notebooks/ATSA_Analysis.html`.

## 6. Phase 5 — Streamlit app "AeroCast" (`app/streamlit_app.py`)

**Look and feel:**

- Polished and professional, with a consistent colour palette.
- Plotly charts that read well in light and dark themes.
- `st.set_page_config(layout="wide", page_title="AeroCast")`.

**Performance:** it must run smoothly on a 1 GB-RAM EC2 instance.

- Cache data loading and model fitting (`st.cache_data` / `st.cache_resource`).
- Read backtest results from `results/` instead of recomputing them.

### Sidebar

- **Dataset:**
  - any of the 7 airports,
  - "All 7 hubs",
  - AirPassengers (monthly),
  - Australia (annual),
  - "Upload CSV" (two columns, date + value; auto-detect the frequency with `pd.infer_freq`; validate and show friendly errors).
- **Model:** default AR-Holt.
- **Horizon:** daily 1–28 (default 14), monthly 1–24, annual 1–10.
- **Training start date.**

### Tabs

1. **📈 Forecast**
   - KPI tiles: last actual, next-period forecast, horizon total, current regime (Calm/Shock) with today's α_t.
   - Chart: the last ~180 points, the forecast, and 80/95% interval bands.
   - Forecast table with CSV download.
2. **🧭 How the model adapts**
   - α_t and the tracking signal over the full history, with shaded shock regimes and outlier markers.
   - The event table.
   - A short plain-English explanation.
3. **🏁 Model comparison**
   - Leaderboard from `results/` for the chosen dataset and experiment (A/B/C): MAE, RMSE, MAPE, MASE, coverage, Winkler, with the best value highlighted.
   - MAPE-by-horizon chart.
   - Diebold–Mariano table.
   - For uploaded data, run a quick live backtest on the last 20% of the data.
4. **💥 Shock simulator**
   - The user picks a shock type (sudden drop / gradual decline / sudden surge / one-day outlier), magnitude %, duration and start.
   - Apply it to a recent stretch of the series and run Holt vs AR-Holt with their fitted parameters.
   - Show actual vs each model's one-step and 7-step forecasts, and α_t.
   - Report "days until error is back within 10%" for each model.
   - This is the demo that makes "adaptive" obvious in a presentation.
5. **🛫 Capacity planner**
   - **Airports:** a staffing heatmap (day × hour → lanes needed) for the forecast window at the chosen quantile, plus a peak-hours table. Assumption inputs are editable.
   - **Other series:** the flight/seat planner.
6. **ℹ️ About**
   - Method summary with `st.latex` equations.
   - Data sources and credits.
   - The team for the current `PROJECT_COURSE`.
   - GitHub link.

**Footer:** project title + team (from config).

Run the app locally and click through every tab. Use headless runs or screenshots if you can. Fix every error and warning you see.

## 7. Phase 6 — Docker and deploy docs (VCC)

- **`Dockerfile`:**
  - Base `python:3.11-slim` (or 3.12).
  - Non-root user.
  - `pip install --no-cache-dir -r requirements.txt`.
  - Copy `airdemand/ app/ data/ results/ .streamlit/`.
  - `ENV PROJECT_COURSE=vcc`.
  - `EXPOSE 8501`.
  - HEALTHCHECK using python urllib against `http://localhost:8501/_stcore/health` (slim images have no curl).
  - CMD: `streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true`.
- **`.dockerignore`:** exclude raw/, .venv/, notebooks/, tests/, .git/.
- **`docker-compose.yml`:** port `80:8501`, `restart: unless-stopped`, `PROJECT_COURSE=vcc`.
- **`.streamlit/config.toml`:** headless, theme, `maxUploadSize` 10.
- If Docker is installed on my laptop, build the image and run it to confirm the health check works. If it isn't, skip this and say so.
- **`deploy/DEPLOY.md`:** manual EC2 deployment steps:
  1. launch a free-tier instance (Amazon Linux 2023),
  2. security group: 22 from my IP, 80 from anywhere,
  3. install docker + git, add 2 GB swap,
  4. git clone, `docker build`, `docker run -p 80:8501`,
  5. verify,
  6. how to update,
  7. teardown and cost notes,
  8. a simple architecture diagram in Mermaid (User → Internet → Security Group → EC2 → Docker → Streamlit).
- **`deploy/CLAUDE_CODE_DEPLOY_PROMPT.md`:** save the Appendix text below into it verbatim. Do NOT run it.

## 8. Phase 7 — Documentation

**`README.md`:**

- Title and one-paragraph pitch.
- "What makes this project different" (5 bullets).
- Screenshots section (placeholders I'll fill).
- Quick start: local venv and Docker.
- Results table, generated from `results/metrics_summary.csv`.
- Method overview with the AR-Holt equations.
- Project structure.
- Data sources and credits.
- Teams per course.
- License (MIT).

**`VIVA_NOTES.md`:**

- What every file does, in plain English.
- The AR-Holt intuition in simple words, with a worked numeric example of one update step.
- 20 likely viva questions with short answers, for both ATSA and VCC. For example:
  - Why remove seasonality before Holt?
  - What is a tracking signal?
  - Why damp the trend?
  - Why conformal intervals?
  - What does the DM test show?
  - Why Docker?
  - Why EC2, and how does the security group protect it?

## 9. Phase 8 — Git and GitHub

1. `git init`, sensible `.gitignore` (raw/, .venv/, __pycache__, .ipynb_checkpoints, *.pem, deploy/aws-resources.env), initial commit with a clear message.
2. **Ask me before creating the remote.**
   - Suggest a PUBLIC repo `air-demand-forecasting` under my GitHub account (username: ritipshah), so EC2 can clone it without credentials.
   - Use `gh repo create` if the GitHub CLI is installed and logged in.
   - Otherwise, give me the exact clicks and commands.
3. Push, and print the repo URL. I'll paste that URL into the deploy prompt.

## 10. Finish

End with:

- a short summary,
- the commands to run the app locally,
- the final results table,
- anything you couldn't do and why.

---

## Appendix — text for `deploy/CLAUDE_CODE_DEPLOY_PROMPT.md` (save verbatim, do not run)

~~~text
You are helping me deploy a university project to AWS EC2. I'm a student in a
Virtualization & Cloud Computing (VCC) course, so explain each step in 1–2 plain
sentences as you go — I have to present this deployment.

PROJECT
- Name: "Short-Term Air Passenger Demand Forecasting via Adaptive Holt's Linear Trend Method"
- A Python Streamlit dashboard, containerised with Docker.
- Repo: <PASTE GITHUB REPO URL HERE>
- Contract the repo follows: Dockerfile at repo root; app listens on port 8501;
  health check at /_stcore/health returns "ok"; env var PROJECT_COURSE=vcc makes
  the app show our VCC team (Riti Shah, Pradhi Kanani, Dhruv Parekh).
- If the repo has deploy/DEPLOY.md, read it first and prefer its instructions.

GROUND RULES
- Region: ap-south-1 (Mumbai) unless I say otherwise.
- Stay inside the AWS Free Tier. Before creating ANYTHING that may cost money
  (instance type, Elastic IP / public IPv4, EBS above free size), tell me the
  likely cost and wait for my yes.
- Never ask me to paste AWS access keys or secrets into chat. If credentials are
  missing, tell me to run `aws configure` myself.
- Never commit .pem files, keys or resource-ID files to git; add them to .gitignore.
- Detect my OS (Windows / macOS / Linux) and adapt commands (e.g. .pem permissions).
- Tag every resource Project=air-demand-forecasting.

PART 1 — INFRASTRUCTURE
1. Check prerequisites: `aws --version` (v2), `aws sts get-caller-identity`, git.
2. Pick a free-tier instance type using
   `aws ec2 describe-instance-types --filters Name=free-tier-eligible,Values=true`
   (prefer t3.micro or t2.micro, x86_64). Show me what you found.
3. Get the latest Amazon Linux 2023 AMI from SSM parameter
   /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64.
4. Create a key pair "airdemand-key", save the .pem locally with correct permissions.
5. Create security group "airdemand-sg": SSH (22) only from my current public IP
   (https://checkip.amazonaws.com), HTTP (80) from 0.0.0.0/0.
6. Launch ONE instance with a 20 GB gp3 root volume and user-data that:
   installs docker + git, enables docker, adds ec2-user to the docker group,
   and creates a 2 GB swap file (1 GB RAM is tight for building the image).
7. Wait until status checks pass, then SSH in and confirm `docker --version` works.
8. Save all resource IDs (instance, SG, key name, public IP, AMI) to
   deploy/aws-resources.env (gitignored) and print a summary table.

PART 2 — DEPLOY THE APP
1. On the instance: git clone the repo into ~/app.
2. `docker build -t airdemand:latest .`
3. `docker run -d --name airdemand --restart unless-stopped -p 80:8501 -e PROJECT_COURSE=vcc airdemand:latest`
4. Verify: `curl http://<public-ip>/_stcore/health` returns "ok", and give me the
   http:// URL to open in a browser.
5. Create deploy/redeploy.sh (SSH in → git pull → rebuild → replace container)
   so we can push updates with one command.

PART 3 — FOR OUR REPORT + SAFETY
1. Write deploy/DEPLOYMENT_LOG.md: architecture (User → Internet → Security Group
   → EC2 → Docker container → Streamlit), every step with the command used,
   resource IDs (no secrets), and a checklist of screenshots to take
   (EC2 console, security group rules, `docker ps`, the live app).
2. Create deploy/teardown.sh that stops/terminates the instance and deletes the
   SG, key pair and any Elastic IP — do NOT run it, just create it.
3. Offer (ask first) to set up an AWS Budget alert at $1 to my email, and
   optionally a GitHub Actions workflow that redeploys on every push to main.

Start with Part 1, step 1. Stop and ask me whenever a decision costs money.
~~~
