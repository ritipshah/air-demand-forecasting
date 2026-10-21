"""Run every experiment and write the results the app and notebook read.

Run from the repository root (each version takes a minute or two):

    python scripts/run_backtests.py                    # v2: rolling seasonal indices -> results/
    python scripts/run_backtests.py --seasonal fixed   # v1: fixed indices -> results/v1_fixed_indices/

Experiments
-----------
A  "Normal operations"  each airport + ALL7   train start..2024-05-31, test to 2025-05-31, H=14
B  "COVID stress test"  each airport + ALL7   train start..2020-02-29, test to 2020-12-31, H=7
C1 classic              AirPassengers          train 1949-01..1957-12, test 1958-01..1960-12, H=12
C2 classic              Australia annual       train 1970..2006, test 2007..2016, H=5

Outputs (results/, or results/v1_fixed_indices/ for --seasonal fixed)
-----------------------------------------------------------------------
metrics_by_horizon.csv  one row per experiment, series, model, h
metrics_summary.csv     the same, averaged over horizons
dm_tests.csv            Diebold–Mariano: AR-Holt vs every other model, h=1 and 7
ablation.csv            which AR-Holt ingredient helps where (ALL7, A and B)
ar_holt_params.csv      fitted AR-Holt parameters per experiment and series
"""

import argparse
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from airdemand import config  # noqa: E402
from airdemand.data import load_series, season_period  # noqa: E402
from airdemand.evaluation import (dm_table, mase_scale, metrics_by_horizon,  # noqa: E402
                                  run_backtest, summarise)
from airdemand.models import MODELS, AdaptiveRobustHolt, DampedHolt  # noqa: E402

warnings.filterwarnings("ignore")

EXPERIMENTS = {
    "A": {"label": "Normal operations", "series": config.AIRPORTS + [config.ALL7],
          "train_end": "2024-05-31", "test_end": "2025-05-31", "H": 14},
    "B": {"label": "COVID stress test", "series": config.AIRPORTS + [config.ALL7],
          "train_end": "2020-02-29", "test_end": "2020-12-31", "H": 7},
    "C1": {"label": "Classic monthly (AirPassengers)", "series": ["AirPassengers"],
           "train_end": "1957-12-01", "test_end": "1960-12-01", "H": 12},
    "C2": {"label": "Classic annual (Australia)", "series": ["Australia"],
           "train_end": "2006-01-01", "test_end": "2016-01-01", "H": 5},
}



def ablation(seasonal: str) -> dict:
    return {
        "DampedHolt (neither)": partial(DampedHolt, seasonal=seasonal),
        "AR-Holt, b=0 (robust only)": partial(AdaptiveRobustHolt, fix_b=0.0, seasonal=seasonal),
        "AR-Holt, k=inf (adaptive only)": partial(AdaptiveRobustHolt, k=np.inf, seasonal=seasonal),
        "AR-Holt (full)": partial(AdaptiveRobustHolt, seasonal=seasonal),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seasonal", choices=["rolling", "fixed"], default=config.SEASONAL_MODE,
                        help="seasonal indices: rolling (v2, default) or fixed (v1)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default depends on --seasonal)")
    args = parser.parse_args()
    seasonal = args.seasonal
    out_dir = args.out or (config.RESULTS_DIR if seasonal == "rolling" else config.RESULTS_V1_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"seasonal indices: {config.SEASONAL_MODES[seasonal]} -> {out_dir}")
    by_h_all, dm_all, abl_all, params_all = [], [], [], []
    t_start = time.time()
    for exp, spec in EXPERIMENTS.items():
        for name in spec["series"]:
            t0 = time.time()
            y, freq = load_series(name)
            y_train = y[:spec["train_end"]]
            scale = mase_scale(y_train, season_period(freq))
            keys = {"experiment": exp, "series": name}

            fc, fitted = run_backtest(y, freq, spec["train_end"], spec["test_end"], spec["H"],
                                      seasonal=seasonal)
            by_h = metrics_by_horizon(fc, scale)
            by_h_all.append(by_h.assign(**keys))

            horizons = sorted({1, min(7, spec["H"])})
            dm = dm_table(fc, config.PROPOSED, horizons)
            dm_all.append(dm.assign(**keys))

            p = fitted[config.PROPOSED].params_
            params_all.append({**keys, "n_train": len(y_train),
                               **{k: p[k] for k in ("a", "b", "beta", "phi", "alpha_calm", "alpha_shock")}})

            if name == config.ALL7 and exp in ("A", "B"):
                fc_abl, _ = run_backtest(y, freq, spec["train_end"], spec["test_end"], spec["H"],
                                         models=ablation(seasonal))
                abl = metrics_by_horizon(fc_abl, scale)
                abl_all.append(abl.assign(**keys))

            mape = by_h.groupby("model")["MAPE"].mean()
            print(f"{exp:3s} {name:14s} {time.time() - t0:5.1f}s  "
                  f"AR-Holt MAPE {mape[config.PROPOSED]:6.2f}  best: {mape.idxmin()} {mape.min():.2f}")

    order = {m: i for i, m in enumerate(config.MODEL_ORDER)}
    lead = ["experiment", "series", "model"]
    by_h = pd.concat(by_h_all, ignore_index=True)
    by_h = by_h[lead + [c for c in by_h.columns if c not in lead]]
    by_h = by_h.sort_values(["experiment", "series", "model", "h"],
                            key=lambda s: s.map(order) if s.name == "model" else s)
    by_h.to_csv(out_dir / "metrics_by_horizon.csv", index=False, float_format="%.6g")

    summary = summarise(by_h, ["experiment", "series"])
    summary = summary.sort_values(["experiment", "series", "model"],
                                  key=lambda s: s.map(order) if s.name == "model" else s)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False, float_format="%.6g")

    dm = pd.concat(dm_all, ignore_index=True)
    dm = dm[["experiment", "series", "h", "loss", "model", "vs", "n", "mean_loss_diff",
             "dm_stat", "p_value", "verdict"]]
    dm.to_csv(out_dir / "dm_tests.csv", index=False, float_format="%.6g")

    abl = pd.concat(abl_all, ignore_index=True)
    abl = abl.rename(columns={"model": "variant"})
    abl_sum = abl.groupby(["experiment", "series", "variant"], sort=False)[
        ["MAE", "RMSE", "MAPE", "MASE", "cov80", "cov95", "winkler95"]].mean().reset_index()
    for h in (1, 7):
        abl_sum[f"MAPE_h{h}"] = abl[abl["h"] == h].set_index(["experiment", "variant"])["MAPE"] \
            .reindex(pd.MultiIndex.from_frame(abl_sum[["experiment", "variant"]])).to_numpy()
    abl_sum.to_csv(out_dir / "ablation.csv", index=False, float_format="%.6g")

    pd.DataFrame(params_all).to_csv(out_dir / "ar_holt_params.csv", index=False,
                                    float_format="%.6g")
    print(f"done in {time.time() - t_start:.0f}s; wrote {out_dir}")


if __name__ == "__main__":
    main()
