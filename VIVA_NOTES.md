# Viva notes — AeroCast / AR-Holt

Plain-English notes for defending the project in the ATSA and VCC vivas.

## 1. What every file does

| File | In plain English |
|---|---|
| `airdemand/config.py` | One place for settings: both teams, file paths, seasonal periods, AR-Holt's fixed hyperparameters, capacity-planner assumptions. `PROJECT_COURSE` picks the team shown in the app. |
| `airdemand/data.py` | Loads the four processed datasets as tidy time series. It also reads a user's uploaded CSV, detects whether it is daily/weekly/monthly/quarterly/annual, and returns friendly error messages. |
| `airdemand/seasonal.py` | Removes the weekday (or monthly) pattern before Holt and puts it back afterwards. Each index is the *median* ratio of the data to a centred moving average. |
| `airdemand/models.py` | All 8 models with the same `fit` / `forecast` interface. SES, Holt, damped Holt, Trigg–Leach and AR-Holt are written from scratch; Holt-Winters wraps statsmodels. `MODELS` is the registry the app and scripts loop over. |
| `airdemand/intervals.py` | Prediction intervals. It checks how wrong each model was in the past (scaled by its error level at the time) and uses those empirical quantiles. |
| `airdemand/evaluation.py` | The backtest: fit once, roll forward through the test period, forecast from every day. Also the metrics (MAE, RMSE, MAPE, sMAPE, MASE, coverage, Winkler) and the Diebold–Mariano test. |
| `airdemand/events.py` | Turns AR-Holt's internals into a table of "regime shifts" (long shock periods) and "outliers" (clipped one-off days), with a near-holiday flag. |
| `airdemand/capacity.py` | Forecast → resources. Passengers per hour → security lanes → officers; demand → seats → flights. |
| `app/streamlit_app.py` | The AeroCast dashboard. It has six tabs and only displays results; all the maths lives in `airdemand/`. |
| `scripts/prepare_data.py` | Raw TSA hourly files → daily totals per airport (missing days imputed and flagged), ALL7, and hourly profiles. Also processes the two classic datasets. |
| `scripts/run_backtests.py` | Runs experiments A, B, C1, C2 and the ablation, and writes the CSVs in `results/`. |
| `scripts/build_notebook.py` | The source of the ATSA notebook (it writes `notebooks/ATSA_Analysis.ipynb`). |
| `scripts/build_docs.py` | Writes every result number into README and these notes, so no number is typed by hand. |
| `tests/` | pytest checks covering the seasonal round trip, Holt vs statsmodels, nesting, no look-ahead, ordered intervals, the level-shift response, and the capacity maths. |
| `Dockerfile`, `.dockerignore`, `docker-compose.yml` | How to package the app as a container (non-root user, health check, port 8501). |
| `.streamlit/config.toml` | Streamlit settings: headless mode, theme, 10 MB upload limit. |
| `deploy/DEPLOY.md` | Step-by-step manual EC2 deployment. `deploy/CLAUDE_CODE_DEPLOY_PROMPT.md` is the prompt for the automated version. |

## 2. AR-Holt in simple words

Imagine a forecaster who keeps two numbers in their head: **today's normal demand** (the level)
and **how fast it is changing** (the trend). Each morning they compare yesterday's forecast with
what actually happened and nudge both numbers towards the truth. **α** is how big the nudge is.

* If the misses go **up, down, up, down**, that's just noise. The forecaster stays calm and makes
  small nudges.
* If the misses are **always in the same direction**, something has really changed. The forecaster
  should make big nudges and catch up fast.
* If there is **one huge miss** (Thanksgiving), it's probably a one-off. The forecaster shouldn't
  rebuild their whole picture because of it.

AR-Holt automates exactly this:

* The **tracking signal** TS = (average error) / (average size of error). It is near 0 for noise and
  near ±1 when the misses are one-sided.
* **α** rises smoothly with |TS|, from the calm value to the shock value.
* While calm, any error bigger than **k = 2.5 typical errors** is cut down to 2.5 (Huber clipping) and
  flagged as an outlier.
* Once |TS| passes **τ = 0.5**, the model decides "this is a real shift". It stops clipping and uses
  full errors with a bigger α.
* The **damped trend** (φ < 1) stops the model from extrapolating a crash forever.
* The **error scale σ** tracks how big errors currently are. It sets the clipping threshold and the
  width of the prediction intervals.

**Honest trade-off:** on the first 2–3 days of an *instant* jump, AR-Holt can't tell an outlier from
a shift, so it clips and waits; Holt reacts immediately. After those days AR-Holt catches up much
faster. For *gradual* collapses (like COVID), AR-Holt wins clearly. This is shown in notebook §10.4.

### Worked example: one real update step

<!-- WORKED:START -->
Real numbers from the All-7-hubs model (fitted on the full history), for
**Wednesday 15 February 2023** — a day the model flagged as an outlier. All values are on the
seasonally adjusted scale (passengers ÷ that weekday's index).

Fitted parameters: a = -0.985, b = 0.693, β = 0.0101, φ = 0.934;
fixed k = 2.5, λ = 0.1, τ = 0.5.

| Step | Formula | Value |
|---|---|---|
| State the evening before | ℓ, b, σ, TS | ℓ = 407,714, b = 17.7, σ = 16,209, TS = +0.163 |
| 1. One-step forecast | x̂ = ℓ + φ·b | 407,714 + 0.934 × 17.7 = **407,731** |
| 2. Actual (adjusted) and error | e = x − x̂ | 463,424 − 407,731 = **55,694** |
| 3. Regime | \|TS\| = 0.163 vs τ = 0.5 | **calm** |
| 4. How unusual? | z = e / σ | 55,694 / 16,209 = **+3.44** (beyond ±2.5 → outlier) |
| 5. Clip the error | ẽ = σ·clip(z, −k, k) | 16,209 × +2.50 = **40,523** |
| 6. Step size | α = expit(a + b·\|TS\|) | expit(-0.985 + 0.693 × 0.163) = **0.295** |
| 7. New level | ℓ = x̂ + α·ẽ | 407,731 + 0.295 × 40,523 = **419,682** |
| 8. New trend | b = φ·b + α·β·ẽ | **137.6** |
| 9. New error scale | σ = λ\|ẽ\| + (1−λ)σ | **18,641** |

**Why it matters:** without clipping (plain damped Holt with the same α) the level would have moved to
424,157 instead of 419,682 — a 1.1% jump caused by a
single unusual day, which would have spoiled the next days' forecasts.
<!-- WORKED:END -->

### Headline results (generated)

<!-- KEYNUMBERS:START -->
* **COVID stress test (All 7 hubs):** AR-Holt MAPE 15.4% vs classic Holt 25.2% (39% lower) and damped Holt 19.8%; best model at 4 of 7 airports.
* **Normal year (All 7 hubs):** AR-Holt 6.79% vs Holt–Winters 6.44% — the seasonal benchmark wins in calm times (AR-Holt best at 2 of 7 airports).
* **Ablation (COVID):** damped Holt 19.8% → robust only 18.7% → adaptive only 19.8% → full AR-Holt 15.4%: the two ingredients only pay off together.
<!-- KEYNUMBERS:END -->

## 3. Likely viva questions

### ATSA (time series)

1. **Why remove seasonality before Holt?**
   Holt's method only has a level and a trend; it has no term for "Fridays are busy". If we fed it
   raw daily data, it would read every Friday peak as a jump in level. Dividing by multiplicative
   weekday indices gives a series with only level, trend and noise. We forecast that series and
   then multiply the weekday pattern back in.

2. **Why multiplicative and why the median?**
   The weekday swing is proportional to the level: during COVID it shrank along with traffic, so a
   ratio fits better than a fixed difference. The *median* ratio ignores a few extreme days, such as
   a holiday falling on a Thursday, where a mean would be pulled by them.

3. **What is a tracking signal?**
   TS = E/M, where E is the exponentially smoothed error and M the smoothed absolute error, so
   TS ∈ [−1, 1]. Random errors cancel in E, giving TS ≈ 0. Persistent one-sided errors give
   |TS| → 1, which means the forecast is biased because the level has moved.

4. **Why damp the trend?**
   An undamped Holt projects today's slope indefinitely. After a collapse it forecasts a further
   collapse, and forecasts can even go negative. φ < 1 makes the trend fade (φ + φ² + … converges),
   which is empirically much more robust. Gardner & McKenzie's damped trend is one of the strongest
   simple benchmarks there is.

5. **What does AR-Holt add over Trigg–Leach ARR?**
   ARR sets α = |TS| directly, which is very noisy, and its TS is inflated by single outliers.
   AR-Holt uses a smooth logistic function with fitted a and b (Taylor 2004), computes TS from
   *cleaned* errors so one holiday can't trigger it, and adds robust clipping and a damped trend.

6. **Why is a, the calm-α parameter, unconstrained when the brief said a, b ≥ 0?**
   a ≥ 0 would force calm α ≥ 0.5, a jumpy model in normal times, which defeats the purpose of a
   calm regime. Only b ≥ 0 is needed: it guarantees α can only increase when the forecasts are
   biased.

7. **How are parameters estimated?**
   By minimising the sum of squared one-step-ahead errors after a burn-in of 2m points, using
   Nelder–Mead from 3 starting points. Parameters are mapped through logit/softplus so the
   constraints always hold. The four hyperparameters γ, k, λ, τ are fixed a priori. A sensitivity
   check on a validation window *inside* the training data shows results barely move.

8. **What is the nesting test and why does it matter?**
   With b = 0 (α constant) and k = ∞ (no clipping), AR-Holt's equations reduce exactly to damped
   Holt in error-correction form. A test confirms this to machine precision. It proves the
   implementation is correct and that AR-Holt is a strict generalisation of damped Holt.

9. **How do you know your from-scratch Holt is right?**
   With identical parameters and initial states, its fitted values and forecasts match statsmodels'
   `Holt` to about 1e-14. With each optimising its own parameters, the forecasts differ by a tiny
   fraction of a percent (notebook §8).

10. **What is the backtest protocol, and why no refitting?**
    Parameters are fitted once on the training span. The model then runs through the test period
    with parameters fixed, forecasting h = 1..H from every daily origin. This mimics a model deployed
    and left alone; it is fast, and it cannot leak test data. A unit test checks that changing the
    future does not change past forecasts (no look-ahead).

11. **Why MASE as well as MAPE?**
    MAPE is easy to read but explodes when actuals are near zero (COVID) and penalises over- and
    under-forecasts unequally. MASE divides MAE by the in-sample seasonal-naive MAE. It is
    scale-free, and a value below 1 means better than "same day last week".

12. **What does the Diebold–Mariano test show?**
    Whether the difference in forecast loss between two models is statistically significant rather
    than luck. The loss difference d_t is tested for zero mean. We use a Newey–West variance with lag
    h−1 (h-step errors overlap) and the Harvey–Leybourne–Newbold small-sample correction with a t
    distribution. A negative statistic means AR-Holt is better. Result: AR-Holt is significantly
    better at h=1 during COVID, and Holt–Winters is significantly better in the normal year.

13. **Why conformal prediction intervals?**
    The residuals are heavy-tailed and autocorrelated: Jarque–Bera and Ljung–Box reject (notebook §11).
    Normal-theory intervals would be wrong. Conformal intervals use the model's own empirical error
    quantiles, so they need no distribution assumption. Dividing by σ_t makes them *locally adaptive*:
    after a shock σ_t grows and the intervals widen automatically.

14. **What do the ADF and KPSS tests tell you?**
    ADF tests H0 = unit root; KPSS tests H0 = stationarity. Using both avoids relying on one test's
    low power. Levels are non-stationary (trend plus the COVID break); first differences are
    stationary. That supports a trend model like Holt, which forecasts the change through b_t.

15. **What did the ablation show?**
    In the COVID test, neither clipping alone nor adaptive α alone helps much. "Adaptive only" even
    collapses back to damped Holt, because the optimiser sets b ≈ 0: without clipping, a large α would
    chase pre-COVID holiday outliers. Only the combination works, so the two ingredients are
    complementary.

16. **Where does AR-Holt lose, and why?**
    In the normal year Holt–Winters wins. It updates the weekday pattern every day, while our seasonal
    indices are fixed from the last 26 training weeks. On short annual data (Australia) there are no
    shocks to adapt to and few errors to calibrate intervals with. We report these results as they are.

### VCC (cloud and containers)

17. **Why Docker?**
    The container bundles the exact Python version, pinned libraries, code, data and results into one
    image, so "it works on my laptop" becomes "it works anywhere Docker runs". It is reproducible,
    isolated from the host, quick to start, and easy to update (rebuild and replace). It runs as a
    non-root user, and a HEALTHCHECK lets Docker see whether the app is alive.

18. **Container vs virtual machine?**
    A VM virtualises hardware and runs a full guest OS. A container shares the host kernel and isolates
    only processes, filesystem and network (namespaces + cgroups). That makes it much lighter: seconds
    to start, megabytes not gigabytes. Here we use both: an EC2 VM hosts the Docker container.

19. **Why EC2, and how does the security group protect it?**
    EC2 gives us a full Linux server we control, inside the free tier, which suits a long-running web
    app with Docker. The security group is a stateful virtual firewall in front of the instance: only
    port 80 (HTTP) is open to the world, SSH (22) is open only to our own IP, and everything else is
    denied. Port 8501 is never exposed; Docker maps 80 → 8501 inside the host.

20. **How does the app fit in 1 GB of RAM, and how is it monitored and updated?**
    Backtests are pre-computed and only *read* by the app. Data loading and model fitting are cached,
    the image contains only the runtime libraries (no Jupyter or matplotlib), and 2 GB of swap covers
    the image build. The container restarts automatically (`--restart unless-stopped`) and reports
    health at `/_stcore/health`. Updating means `git pull`, rebuild, and replacing the container
    (deploy/DEPLOY.md §6).

### Bonus quick answers

* **Why the 7 hubs and ALL7?** Each hub has its own shocks, such as storms at DFW, DEN and JFK. The
  total smooths out local noise but keeps national shocks like COVID.
* **What is imputed and how?** The same 8 days are missing for every airport. Each is filled with the
  average of the same weekday one week before and after, and flagged `imputed = 1`.
* **P80 in the staffing planner?** Staffing to the 80th percentile of demand means queues should be
  acceptable on about 4 days out of 5. It is interpolated between the median and the upper end of the
  80% interval.
