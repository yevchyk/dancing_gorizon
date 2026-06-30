"""Honest walk-forward for fast_v3: each test day scored by a model trained ONLY
on prior anchors (no leakage). Produces out-of-sample scores across several days
so the strategy probes can be re-run on a multi-day, regime-mixed holdout.

  python -m src.run_fast_v3_walkforward --test-days 4 --iterations 300
  then: python -m src.run_fast_v3_unicorn  (point it at holdout_scores_wf.parquet)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .fast import config as FC
from .fast.curve import FastCurve
from .run_fast_v3 import HORIZONS_V3, MAX_HORIZON, V3_ANALYSIS, V3_DATASET, WEIGHT_FLOOR, _fit

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-days", type=int, default=4)
    ap.add_argument("--days", default="", help="explicit comma-sep test days (overrides --test-days)")
    ap.add_argument("--tag", default="", help="output filename suffix, e.g. _bear")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--min-dense", type=int, default=5000, help="min anchors/day to be a test day")
    args = ap.parse_args()

    ds = pd.read_parquet(V3_DATASET).sort_values("anchor_time").reset_index(drop=True)
    ds["at"] = pd.to_datetime(ds["anchor_time"], utc=True)
    ds["day"] = ds["at"].dt.strftime("%Y-%m-%d")
    counts = ds["day"].value_counts()
    dense = sorted(d for d, c in counts.items() if c >= args.min_dense)
    if args.days:
        test_days = [d.strip() for d in args.days.split(",")]
    else:
        test_days = dense[-args.test_days:]
    print(f"dataset {len(ds)} rows; testing {test_days}\n")

    curve = FastCurve(FC.CURVE_POINTS, FC.CURVE_MIN_STEP_MIN, FC.CURVE_MAX_DEPTH_MIN, FC.CURVE_SEGMENTS)
    embargo = pd.Timedelta(minutes=MAX_HORIZON)
    frames = []
    for d in test_days:
        day_start = pd.Timestamp(d, tz="UTC")
        test = ds[ds["day"] == d]
        train = ds[ds["at"] < day_start - embargo]
        print(f"[{d}] train={len(train)} test={len(test)}", flush=True)
        if len(train) < 10000 or len(test) == 0:
            print("  skip (too little train)"); continue
        sc = test[["symbol", "anchor_time", "day"]].copy()
        for m, lab, lb in HORIZONS_V3:
            cols = curve.columns_for_lookback(lb)
            Xtr, Xte = train[cols], test[cols]
            ret = train[f"ret_{lab}"].to_numpy()
            sc[f"real_ret_{lab}"] = test[f"ret_{lab}"].to_numpy()
            sc[f"real_mfe_{lab}"] = test[f"mfe_{lab}"].to_numpy()
            sc[f"real_mae_{lab}"] = test[f"mae_{lab}"].to_numpy()
            for side in ("up", "down"):
                if side == "up":
                    y = (ret > FC.TARGET_EDGE).astype(int); prof = np.maximum(ret, 0.0)
                else:
                    y = (ret < -FC.TARGET_EDGE).astype(int); prof = np.maximum(-ret, 0.0)
                pm = prof[prof > 0].mean() if (prof > 0).any() else 1.0
                w = np.where(prof > 0, prof / pm, WEIGHT_FLOOR).astype("float64")
                model = _fit(Xtr, y, w, args.iterations, args.depth)
                sc[f"p_{side}_{lab}"] = model.predict_proba(Xte)[:, 1]
            print(f"  {lab} done", flush=True)
        frames.append(sc)

    wf = pd.concat(frames, ignore_index=True)
    out = V3_ANALYSIS / f"holdout_scores_wf{args.tag}.parquet"
    wf.to_parquet(out, index=False)
    print(f"\nwalk-forward scores: {len(wf)} rows, days={wf['day'].nunique()} -> {out}")
    # quick per-day long sanity: up_20m @ p>=0.85
    for d, g in wf.groupby("day"):
        p = g["p_up_20m"].to_numpy(); r = g["real_ret_20m"].to_numpy()
        fire = p >= 0.85
        if fire.sum():
            pnl = r[fire] - FC.EVAL_COST
            print(f"  {d}: up_20m p>=.85  n={int(fire.sum())} win={ (pnl>0).mean():.3f} avg={pnl.mean()*100:+.3f}%")


if __name__ == "__main__":
    main()
