# AeroCast — Short-Term Air Passenger Demand Forecasting via Adaptive Holt's Linear Trend Method

AeroCast forecasts daily passenger demand at seven major U.S. airports (and two classic textbook
series) with **AR-Holt**, an *adaptive robust* version of Holt's linear trend method. In calm
periods it behaves like a steady damped Holt model. When demand shifts persistently, as it did
during the COVID-19 collapse, a tracking signal raises its smoothing parameter so it re-levels
fast. One-off shocks such as Thanksgiving or a snowstorm are clipped so they don't distort the
level. The project includes a from-scratch model library, an honest backtest against seven
benchmarks, an analysis notebook and a Streamlit dashboard. The dashboard turns the forecasts
into security-lane and flight plans, and is containerised with Docker for AWS EC2.

## What makes this project different

* **Adaptive *and* robust in one model.** AR-Holt combines four published ideas: Trigg–Leach
  tracking signal, Taylor's smooth-transition smoothing, Gelper–Fried–Croux robust cleaning, and
  the Gardner–McKenzie damped trend. It also has a regime switch that tells *outliers* (clip them)
  apart from *shifts* (follow them).
* **Real, messy, public data.** Hourly TSA checkpoint counts for 7 hubs, 2019–2025, with COVID,
  holidays and storms left in on purpose.
* **Everything from scratch and verified.** SES, Holt, damped Holt, Trigg–Leach and AR-Holt are
  written in numpy. Holt matches statsmodels to machine precision, and AR-Holt provably nests
  damped Holt (both are unit tests).
* **Honest evaluation.** Rolling-origin backtests from every daily origin, MASE, interval coverage,
  Winkler scores, Diebold–Mariano tests with the HLN correction, and an ablation study. Results
  where AR-Holt loses are reported too.
* **Calibrated uncertainty and a real use.** Locally-adaptive conformal prediction intervals widen
  automatically after shocks, and feed a checkpoint staffing planner (lanes and officers per hour).

## Screenshots

> *Placeholders: add your own screenshots here.*
>
> `docs/screenshots/forecast.png`: Forecast tab
> `docs/screenshots/adapts.png`: How the model adapts
> `docs/screenshots/shock.png`: Shock simulator
> `docs/screenshots/capacity.png`: Capacity planner

## Quick start

### Local (Python 3.11 or 3.12)

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app/streamlit_app.py          # opens http://localhost:8501
```
Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
streamlit run app/streamlit_app.py
```

Tests, backtests and the notebook:
```bash
pytest                                       # all tests
python scripts/run_backtests.py              # re-creates results/ (about a minute)
python scripts/build_notebook.py             # re-creates the notebook from its source
jupyter nbconvert --to notebook --execute --inplace notebooks/ATSA_Analysis.ipynb
jupyter nbconvert --to html notebooks/ATSA_Analysis.ipynb
python scripts/build_docs.py                 # refreshes the numbers in README and VIVA_NOTES
```

The processed data in `data/` is committed. To rebuild it from the raw sources, follow the
download commands in *Data sources* below, then run `python scripts/prepare_data.py`.

### Docker

```bash
docker build -t airdemand:latest .
docker run --rm -p 8501:8501 -e PROJECT_COURSE=vcc airdemand:latest   # http://localhost:8501
# or: docker compose up -d --build                                      # http://localhost (port 80)
```
Health check: `curl http://localhost:8501/_stcore/health` returns `ok`. AWS EC2 steps are in
[deploy/DEPLOY.md](deploy/DEPLOY.md).

## Results

<!-- RESULTS:START -->
| Experiment | Naive | SeasonalNaive | SES | Holt | DampedHolt | TriggLeachHolt | HoltWinters | AR-Holt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A · normal year · All 7 hubs | 11.41 | 7.33 | 6.68 | 7.03 | 6.76 | 7.25 | **6.44** | 6.79 |
| A · normal year · mean of 7 airports | 13.56 | 9.54 | 8.93 | 9.21 | 8.93 | 9.85 | **8.61** | 8.90 |
| B · COVID · All 7 hubs | 19.35 | 19.23 | 19.73 | 25.21 | 19.78 | 20.74 | 18.94 | **15.41** |
| B · COVID · mean of 7 airports | 21.03 | 20.89 | 21.85 | 27.07 | 21.86 | 22.31 | 20.56 | **19.15** |
| C1 · AirPassengers (monthly) | 16.80 | 8.91 | 7.36 | 5.41 | 5.41 | 6.28 | **5.34** | 5.50 |
| C2 · Australia (annual) | 12.89 | 12.89 | 12.89 | 8.71 | 8.71 | 9.72 | **7.05** | 9.49 |

Mean MAPE (%) over all horizons; **bold** = best in row. Generated from `results/metrics_summary.csv` by `scripts/build_docs.py`.
<!-- RESULTS:END -->

<!-- KEYNUMBERS:START -->
* **COVID stress test (All 7 hubs):** AR-Holt MAPE 15.4% vs classic Holt 25.2% (39% lower) and damped Holt 19.8%; best model at 4 of 7 airports.
* **Normal year (All 7 hubs):** AR-Holt 6.79% vs Holt–Winters 6.44% — the seasonal benchmark wins in calm times (AR-Holt best at 2 of 7 airports).
* **Ablation (COVID):** damped Holt 19.8% → robust only 18.7% → adaptive only 19.8% → full AR-Holt 15.4%: the two ingredients only pay off together.
<!-- KEYNUMBERS:END -->

In short: AR-Holt is **insurance against shocks**. It is the most accurate method during the COVID
collapse and about as good as damped Holt in normal times, where the fully seasonal Holt–Winters
benchmark is best. Full tables, Diebold–Mariano tests, coverage and the ablation study are in
`notebooks/ATSA_Analysis.ipynb` (section 12) and in the app's *Model comparison* tab.

## Method overview

1. **Seasonal adjustment.** Multiplicative weekday (or month) indices: the median ratio to a centred
   moving average, normalised to mean 1. They are removed before Holt and put back afterwards.
2. **AR-Holt** on the adjusted series $x_t$, at every step:

$$\hat x_t=\ell_{t-1}+\phi b_{t-1},\qquad e_t=x_t-\hat x_t$$

$$\tilde e_t=\begin{cases}e_t & |TS_{t-1}|\ge\tau \text{ (shock)}\\ \sigma_{t-1}\,\mathrm{clip}(e_t/\sigma_{t-1},-k,k) & \text{otherwise (calm)}\end{cases}$$

$$\alpha_t=\mathrm{expit}\left(a+b\,|TS_{t-1}|\right),\qquad \ell_t=\hat x_t+\alpha_t\tilde e_t,\qquad b_t=\phi b_{t-1}+\alpha_t\beta\tilde e_t$$

$$\sigma_t=\lambda|\tilde e_t|+(1-\lambda)\sigma_{t-1},\qquad TS_t=\frac{E_t}{M_t},\quad E_t=\gamma\tilde e_t+(1-\gamma)E_{t-1},\quad M_t=\gamma|\tilde e_t|+(1-\gamma)M_{t-1}$$

$$\hat x_{n+h}=\ell_n+(\phi+\phi^2+\dots+\phi^h)\,b_n$$

   The fitted parameters are $a$, $b\ge0$, $\beta$ and $\phi\in[0.8,1]$, found by minimising
   one-step SSE with Nelder–Mead from 3 starting points. The fixed hyperparameters are
   $\gamma=0.15$, $k=2.5$, $\lambda=0.1$ and $\tau=0.5$, and they were not tuned on test data.
   With $b=0$ and $k=\infty$, AR-Holt is exactly damped Holt.
3. **Prediction intervals.** These are split-conformal: empirical quantiles of the past h-step
   errors divided by $\sigma_t$, then scaled by the current $\sigma_n$.
4. **Benchmarks.** Naive, seasonal naive, SES, Holt, damped Holt, Trigg–Leach adaptive Holt, and
   statsmodels Holt–Winters (damped trend × multiplicative seasonality).

## Project structure

```
airdemand/            the library: all modelling logic
  config.py           teams, paths, defaults, hyperparameters
  data.py             built-in datasets, CSV upload parsing
  seasonal.py         multiplicative seasonal adjustment
  models.py           8 models behind one fit / forecast interface
  intervals.py        locally-adaptive conformal intervals
  evaluation.py       rolling-origin backtest, metrics, Diebold–Mariano
  events.py           outlier vs regime-shift event table
  capacity.py         staffing and flight planners
app/streamlit_app.py  the AeroCast dashboard
scripts/              prepare_data, run_backtests, build_notebook, build_docs
data/                 processed CSVs (committed)
results/              backtest outputs (read by the app)
notebooks/            ATSA_Analysis.ipynb (+ HTML export)
reports/figures/      figures saved by the notebook
tests/                pytest suite
deploy/               EC2 guide and the deployment prompt
Dockerfile, docker-compose.yml, .streamlit/config.toml
```

## Data sources and credits

* **TSA checkpoint throughput.** U.S. Transportation Security Administration, FOIA Electronic
  Reading Room (U.S. government work, public domain). Parsed to CSV by the open-source project
  [mikelor/TsaThroughput](https://github.com/mikelor/TsaThroughput). The data covers 2018-12-30 to
  2025-05-31 for ATL, DFW, DEN, ORD, LAX, JFK and SEA. The same 8 days are missing for every
  airport; they are imputed from the same weekday one week before and after, and flagged.
* **AirPassengers.** Box & Jenkins monthly international airline passengers, 1949–1960, from the R
  `datasets` package.
* **Australian air passengers.** Annual, 1970–2016, from Hyndman & Athanasopoulos, *Forecasting:
  Principles and Practice* (3rd ed.), R package `fpp3`.
* The R datasets come via [vincentarelbundock/Rdatasets](https://github.com/vincentarelbundock/Rdatasets).

Re-downloading the raw data (a partial checkout, which takes a few minutes):
```bash
mkdir raw && cd raw
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/mikelor/TsaThroughput
cd TsaThroughput
git -c gc.auto=0 checkout HEAD -- data/processed/tsa/throughput/TsaThroughput.{ATL,DFW,DEN,ORD,LAX,JFK,SEA}.csv
cd ..
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/vincentarelbundock/Rdatasets
cd Rdatasets
git -c gc.auto=0 checkout HEAD -- csv/datasets/AirPassengers.csv csv/fpp3/aus_airpassengers.csv
cd ../..
python scripts/prepare_data.py
```

## Teams

| Course | Team | Deliverable |
|---|---|---|
| Applied Time Series Analysis (ATSA) | Riti Shah, Pradhi Kanani, Shreeya | the analysis notebook |
| Virtualization & Cloud Computing (VCC) | Riti Shah, Pradhi Kanani, Dhruv Parekh | the Dockerised app on AWS EC2 |

The environment variable `PROJECT_COURSE` (`vcc` or `atsa`, default `vcc`) chooses which team the
app's footer and About tab show.

## License

MIT. See [LICENSE](LICENSE). The TSA data is public domain; the R datasets keep their original terms.
