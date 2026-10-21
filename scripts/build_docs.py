"""Fill the generated sections of README.md and VIVA_NOTES.md.

Every number in those files comes from here (golden rule: never type a
result by hand). Generated sections sit between markers such as
<!-- RESULTS:START --> and <!-- RESULTS:END --> and are replaced on each run:

    python scripts/build_docs.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from airdemand import config  # noqa: E402
from airdemand.data import load_series  # noqa: E402
from airdemand.models import AdaptiveRobustHolt  # noqa: E402

ORDER = config.MODEL_ORDER


def replace_block(path: Path, tag: str, body: str) -> None:
    text = path.read_text()
    pattern = re.compile(rf"(<!-- {tag}:START -->\n).*?(<!-- {tag}:END -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"marker {tag} not found in {path.name}")
    path.write_text(pattern.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text))


def results_table() -> str:
    s = pd.read_csv(config.RESULTS_DIR / "metrics_summary.csv")
    rows = []
    specs = [("A", "ALL7", "A · normal year · All 7 hubs"),
             ("A", "airports", "A · normal year · mean of 7 airports"),
             ("B", "ALL7", "B · COVID · All 7 hubs"),
             ("B", "airports", "B · COVID · mean of 7 airports"),
             ("C1", "AirPassengers", "C1 · AirPassengers (monthly)"),
             ("C2", "Australia", "C2 · Australia (annual)")]
    for exp, series, label in specs:
        d = s[s.experiment == exp]
        if series == "airports":
            v = d[d.series.isin(config.AIRPORTS)].groupby("model").MAPE.mean()
        else:
            v = d[d.series == series].set_index("model").MAPE
        v = v.reindex(ORDER)
        best = v.idxmin()
        cells = [f"**{v[m]:.2f}**" if m == best else f"{v[m]:.2f}" for m in ORDER]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    head = "| Experiment | " + " | ".join(ORDER) + " |\n|---|" + "---:|" * len(ORDER)
    note = ("\n\nMean MAPE (%) over all horizons; **bold** = best in row. Generated from "
            "`results/metrics_summary.csv` by `scripts/build_docs.py`.")
    return head + "\n" + "\n".join(rows) + note


def key_numbers() -> str:
    s = pd.read_csv(config.RESULTS_DIR / "metrics_summary.csv")
    abl = pd.read_csv(config.RESULTS_DIR / "ablation.csv").set_index(["experiment", "variant"]).MAPE

    def mape(e, ser, m):
        return s[(s.experiment == e) & (s.series == ser) & (s.model == m)].MAPE.iloc[0]

    def wins(e):
        d = s[(s.experiment == e) & s.series.isin(config.AIRPORTS)]
        best = d.loc[d.groupby("series").MAPE.idxmin()]
        return int((best.model == "AR-Holt").sum())

    b_ar, b_holt, b_dh = mape("B", "ALL7", "AR-Holt"), mape("B", "ALL7", "Holt"), mape("B", "ALL7", "DampedHolt")
    a_ar, a_hw = mape("A", "ALL7", "AR-Holt"), mape("A", "ALL7", "HoltWinters")
    return "\n".join([
        f"* **COVID stress test (All 7 hubs):** AR-Holt MAPE {b_ar:.1f}% vs classic Holt {b_holt:.1f}% "
        f"({100 * (1 - b_ar / b_holt):.0f}% lower) and damped Holt {b_dh:.1f}%; best model at "
        f"{wins('B')} of 7 airports.",
        f"* **Normal year (All 7 hubs):** AR-Holt {a_ar:.2f}% vs Holt–Winters {a_hw:.2f}% — the seasonal "
        f"benchmark wins in calm times (AR-Holt best at {wins('A')} of 7 airports).",
        f"* **Ablation (COVID):** damped Holt {abl[('B', 'DampedHolt (neither)')]:.1f}% → robust only "
        f"{abl[('B', 'AR-Holt, b=0 (robust only)')]:.1f}% → adaptive only "
        f"{abl[('B', 'AR-Holt, k=inf (adaptive only)')]:.1f}% → full AR-Holt "
        f"{abl[('B', 'AR-Holt (full)')]:.1f}%: the two ingredients only pay off together.",
    ])


def worked_example() -> str:
    """One real AR-Holt update step, with every intermediate number."""
    y, _ = load_series("ALL7")
    m = AdaptiveRobustHolt().fit(y, freq="D")
    c, p = m.components_, m.params_
    # A calm-regime day whose error was clipped (an outlier), in a recent year.
    cand = c[(c.is_outlier) & (c.index.year >= 2023)]
    day = cand.index[0]
    t = c.index.get_loc(day)
    prev = c.iloc[t - 1]
    l_prev, b_prev, s_prev, ts_prev = prev.level, prev.trend, prev.sigma, prev.ts
    x_t = c.adjusted.iloc[t]
    fhat = l_prev + p["phi"] * b_prev
    e = x_t - fhat
    z = e / s_prev
    et = s_prev * np.clip(z, -p["k"], p["k"])
    alpha = expit(p["a"] + p["b"] * abs(ts_prev))
    level = fhat + alpha * et
    trend = p["phi"] * b_prev + alpha * p["beta"] * et
    sigma = p["lam"] * abs(et) + (1 - p["lam"]) * s_prev
    assert np.isclose(level, c.level.iloc[t]) and np.isclose(sigma, c.sigma.iloc[t])
    alpha_plain = alpha
    level_unclipped = fhat + alpha_plain * e
    f = lambda v: f"{v:,.0f}"  # noqa: E731
    return f"""Real numbers from the All-7-hubs model (fitted on the full history), for
**{day.strftime('%A %d %B %Y')}** — a day the model flagged as an outlier. All values are on the
seasonally adjusted scale (passengers ÷ that weekday's index).

Fitted parameters: a = {p['a']:.3f}, b = {p['b']:.3f}, β = {p['beta']:.4f}, φ = {p['phi']:.3f};
fixed k = {p['k']}, λ = {p['lam']}, τ = {p['tau']}.

| Step | Formula | Value |
|---|---|---|
| State the evening before | ℓ, b, σ, TS | ℓ = {f(l_prev)}, b = {b_prev:,.1f}, σ = {f(s_prev)}, TS = {ts_prev:+.3f} |
| 1. One-step forecast | x̂ = ℓ + φ·b | {f(l_prev)} + {p['phi']:.3f} × {b_prev:,.1f} = **{f(fhat)}** |
| 2. Actual (adjusted) and error | e = x − x̂ | {f(x_t)} − {f(fhat)} = **{f(e)}** |
| 3. Regime | \\|TS\\| = {abs(ts_prev):.3f} vs τ = {p['tau']} | **{'shock' if abs(ts_prev) >= p['tau'] else 'calm'}** |
| 4. How unusual? | z = e / σ | {f(e)} / {f(s_prev)} = **{z:+.2f}** (beyond ±{p['k']} → outlier) |
| 5. Clip the error | ẽ = σ·clip(z, −k, k) | {f(s_prev)} × {np.clip(z, -p['k'], p['k']):+.2f} = **{f(et)}** |
| 6. Step size | α = expit(a + b·\\|TS\\|) | expit({p['a']:.3f} + {p['b']:.3f} × {abs(ts_prev):.3f}) = **{alpha:.3f}** |
| 7. New level | ℓ = x̂ + α·ẽ | {f(fhat)} + {alpha:.3f} × {f(et)} = **{f(level)}** |
| 8. New trend | b = φ·b + α·β·ẽ | **{trend:,.1f}** |
| 9. New error scale | σ = λ\\|ẽ\\| + (1−λ)σ | **{f(sigma)}** |

**Why it matters:** without clipping (plain damped Holt with the same α) the level would have moved to
{f(level_unclipped)} instead of {f(level)} — a {abs(level_unclipped - level) / level:.1%} jump caused by a
single unusual day, which would have spoiled the next days' forecasts."""


def main() -> None:
    readme, viva = ROOT / "README.md", ROOT / "VIVA_NOTES.md"
    replace_block(readme, "RESULTS", results_table())
    replace_block(readme, "KEYNUMBERS", key_numbers())
    replace_block(viva, "KEYNUMBERS", key_numbers())
    replace_block(viva, "WORKED", worked_example())
    print("README.md and VIVA_NOTES.md updated")


if __name__ == "__main__":
    main()
