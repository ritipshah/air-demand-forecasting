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


SPECS = [("A", "ALL7", "A · normal year · All 7 hubs"),
         ("A", "airports", "A · normal year · mean of 7 airports"),
         ("B", "ALL7", "B · COVID · All 7 hubs"),
         ("B", "airports", "B · COVID · mean of 7 airports"),
         ("C1", "AirPassengers", "C1 · AirPassengers (monthly)"),
         ("C2", "Australia", "C2 · Australia (annual)")]


def load(version: str) -> pd.DataFrame:
    folder = config.RESULTS_DIR if version == "rolling" else config.RESULTS_V1_DIR
    return pd.read_csv(folder / "metrics_summary.csv")


def mape_row(s: pd.DataFrame, exp: str, series: str) -> pd.Series:
    d = s[s.experiment == exp]
    if series == "airports":
        return d[d.series.isin(config.AIRPORTS)].groupby("model").MAPE.mean().reindex(ORDER)
    return d[d.series == series].set_index("model").MAPE.reindex(ORDER)


def results_table(version: str) -> str:
    s = load(version)
    rows = []
    for exp, series, label in SPECS:
        v = mape_row(s, exp, series)
        best = v.idxmin()
        cells = [f"**{v[m]:.2f}**" if m == best else f"{v[m]:.2f}" for m in ORDER]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    head = "| Experiment | " + " | ".join(ORDER) + " |\n|---|" + "---:|" * len(ORDER)
    return head + "\n" + "\n".join(rows)


def results_block() -> str:
    folder_v1 = config.RESULTS_V1_DIR.relative_to(ROOT)
    return (
        "**v2 — rolling seasonal indices (default).** The weekday indices are re-estimated at every "
        "forecast origin from the 26 weeks before it (monthly: all data up to the origin).\n\n"
        + results_table("rolling")
        + "\n\n**v1 — fixed seasonal indices** (estimated once on the training span; kept in "
        f"`{folder_v1}/`).\n\n"
        + results_table("fixed")
        + "\n\nMean MAPE (%) over all horizons; **bold** = best in row. Naive, SeasonalNaive and "
        "Holt–Winters work on the raw series, so they are identical in both versions. Generated from "
        "`results/` by `scripts/build_docs.py`.\n\n"
        + version_changes()
    )


def wins(s: pd.DataFrame, exp: str, model: str = "AR-Holt") -> int:
    d = s[(s.experiment == exp) & s.series.isin(config.AIRPORTS)]
    best = d.loc[d.groupby("series").MAPE.idxmin()]
    return int((best.model == model).sum())


def version_changes() -> str:
    s1, s2 = load("fixed"), load("rolling")
    adj = ["SES", "Holt", "DampedHolt", "TriggLeachHolt", "AR-Holt"]
    changes = []
    for exp in ("A", "B"):
        for series in ("ALL7", "airports"):
            d = mape_row(s2, exp, series)[adj] - mape_row(s1, exp, series)[adj]
            changes.extend(d.tolist())
    changes = np.array(changes)
    gap1 = mape_row(s1, "A", "ALL7")["AR-Holt"] - mape_row(s1, "A", "ALL7")["HoltWinters"]
    gap2 = mape_row(s2, "A", "ALL7")["AR-Holt"] - mape_row(s2, "A", "ALL7")["HoltWinters"]
    cov = {v: t[(t.experiment == "B") & (t.series == "ALL7") & (t.model == "AR-Holt")].iloc[0]
           for v, t in (("v1", s1), ("v2", s2))}
    dm = pd.read_csv(config.RESULTS_DIR / "dm_tests.csv")
    r = dm[(dm.experiment == "A") & (dm.series == "ALL7") & (dm.vs == "HoltWinters") & (dm.h == 1)
           & (dm.loss == "squared")].iloc[0]
    c1 = {v: mape_row(t, "C1", "AirPassengers") for v, t in (("v1", s1), ("v2", s2))}
    return "\n".join([
        "**What changed from v1 to v2 (computed):**",
        "",
        f"* The seasonally adjusted models got more accurate in {int((changes < 0).sum())} of {len(changes)} "
        f"daily comparisons (5 models × 2 experiments × 2 series groups). The median change was "
        f"{np.median(changes):+.2f} MAPE points, with larger gains during COVID.",
        f"* During COVID, AR-Holt is best at {wins(s2, 'B')} of 7 airports with rolling indices, against "
        f"{wins(s1, 'B')} of 7 with fixed indices.",
        f"* **Normal year:** Holt–Winters still wins. On All 7 hubs, AR-Holt trails it by {gap1:.2f} points "
        f"(v1) and {gap2:.2f} points (v2), so rolling indices close {100 * (1 - gap2 / gap1):.0f}% of the "
        f"gap. The DM test (h=1) still finds \"{r.verdict}\" (p = {r.p_value:.3f}). Fixed indices "
        "therefore explain only part of Holt–Winters' advantage.",
        f"* **Worse in v2:** during COVID, AR-Holt's 95% intervals on All 7 hubs cover "
        f"{cov['v2'].cov95:.1%} of outcomes, down from {cov['v1'].cov95:.1%} in v1. Its 95% Winkler score "
        f"rises from {cov['v1'].winkler95:,.0f} to {cov['v2'].winkler95:,.0f}, because the intervals are "
        "narrower.",
        f"* On AirPassengers (an expanding window for monthly data), AR-Holt goes from "
        f"{c1['v1']['AR-Holt']:.2f}% to {c1['v2']['AR-Holt']:.2f}%, and the best model is now "
        f"{c1['v2'].idxmin()} (v1: {c1['v1'].idxmin()}). The annual series has no seasonality and is "
        "unchanged.",
    ])


def key_numbers() -> str:
    s = load("rolling")
    s1 = load("fixed")
    abl = pd.read_csv(config.RESULTS_DIR / "ablation.csv").set_index(["experiment", "variant"]).MAPE

    def mape(t, e, ser, m):
        return t[(t.experiment == e) & (t.series == ser) & (t.model == m)].MAPE.iloc[0]

    b_ar, b_holt, b_dh = mape(s, "B", "ALL7", "AR-Holt"), mape(s, "B", "ALL7", "Holt"), mape(s, "B", "ALL7", "DampedHolt")
    a_ar, a_hw = mape(s, "A", "ALL7", "AR-Holt"), mape(s, "A", "ALL7", "HoltWinters")
    return "\n".join([
        "Default version (v2, rolling seasonal indices):",
        "",
        f"* **COVID stress test (All 7 hubs):** AR-Holt MAPE {b_ar:.1f}%, against {b_holt:.1f}% for "
        f"classic Holt ({100 * (1 - b_ar / b_holt):.0f}% lower) and {b_dh:.1f}% for damped Holt. It is the "
        f"best model at {wins(s, 'B')} of 7 airports.",
        f"* **Normal year (All 7 hubs):** AR-Holt {a_ar:.2f}% against {a_hw:.2f}% for Holt–Winters. The "
        f"seasonal benchmark wins in calm times; AR-Holt is best at {wins(s, 'A')} of 7 airports.",
        f"* **Ablation (COVID):** damped Holt {abl[('B', 'DampedHolt (neither)')]:.1f}%, robust only "
        f"{abl[('B', 'AR-Holt, b=0 (robust only)')]:.1f}%, adaptive only "
        f"{abl[('B', 'AR-Holt, k=inf (adaptive only)')]:.1f}%, full AR-Holt "
        f"{abl[('B', 'AR-Holt (full)')]:.1f}%. The two ingredients only pay off together.",
        f"* **Compared with v1 (fixed indices):** AR-Holt on All 7 hubs was "
        f"{mape(s1, 'A', 'ALL7', 'AR-Holt'):.2f}% (normal year) and {mape(s1, 'B', 'ALL7', 'AR-Holt'):.1f}% "
        f"(COVID) in v1, and is {a_ar:.2f}% and {b_ar:.1f}% in v2.",
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
    replace_block(readme, "RESULTS", results_block())
    replace_block(readme, "KEYNUMBERS", key_numbers())
    replace_block(viva, "KEYNUMBERS", key_numbers())
    replace_block(viva, "WORKED", worked_example())
    replace_block(viva, "VERSIONS", version_changes())
    print("README.md and VIVA_NOTES.md updated")


if __name__ == "__main__":
    main()
