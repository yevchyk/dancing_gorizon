"""Probability-percentile slices for fast_v3 holdout (no retrain — reads scores).

For each model, rank the holdout by predicted probability and look at the most
confident top X%: how many signals, win rate, avg PnL, and the avg favorable /
adverse excursion (TP/SL basis) at each strictness.

  python -m src.run_fast_v3_slices
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fast import config as FC
from .run_fast_v3 import HORIZONS_V3, V3_ANALYSIS

CUTS = (1, 2, 5, 10, 20, 100)  # top X% by probability


def _slice(p, real, mfe, mae, sign, q) -> dict:
    thr = np.quantile(p, 1 - q / 100.0)
    fire = p >= thr
    n = int(fire.sum())
    if n == 0:
        return {"n": 0}
    r = real[fire]
    pnl = sign * r - FC.EVAL_COST
    fav = (mfe[fire].mean() if sign > 0 else -mae[fire].mean()) * 100
    adv = (mae[fire].mean() if sign > 0 else -mfe[fire].mean()) * 100
    return {"n": n, "win": float((pnl > 0).mean()), "avg": float(pnl.mean() * 100),
            "tp": float(fav), "sl": float(adv), "total": float(pnl.sum() * 100)}


def main() -> None:
    s = pd.read_parquet(V3_ANALYSIS / "holdout_scores.parquet")
    total_rows = len(s)
    days = s["day"].nunique() if "day" in s else 1
    print(f"holdout rows={total_rows}  days={days}  fee={FC.EVAL_COST*100:.2f}%\n")

    rows = []
    for cut in CUTS:
        n_universe = int(round(total_rows * cut / 100))
        print(f"=== top {cut}% by P  (~{n_universe} rows, ~{n_universe//max(days,1)}/day) ===")
        print(f"{'model':<10}{'n':>6}{'win':>7}{'avg%':>9}{'TP+':>8}{'SL-':>8}{'total%':>9}")
        for m, lab, _ in HORIZONS_V3:
            for side in ("up", "down"):
                p = s[f"p_{side}_{lab}"].to_numpy()
                real = s[f"real_ret_{lab}"].to_numpy()
                mfe = s[f"real_mfe_{lab}"].to_numpy()
                mae = s[f"real_mae_{lab}"].to_numpy()
                r = _slice(p, real, mfe, mae, 1.0 if side == "up" else -1.0, cut)
                if r["n"] == 0:
                    continue
                r.update({"cut%": cut, "model": f"{side}_{lab}"})
                rows.append(r)
                print(f"{side+'_'+lab:<10}{r['n']:>6}{r['win']:>7.3f}{r['avg']:>+9.4f}"
                      f"{r['tp']:>+8.3f}{r['sl']:>+8.3f}{r['total']:>+9.2f}")
        print()
    out = V3_ANALYSIS / "holdout_slices.csv"
    pd.DataFrame(rows)[["model", "cut%", "n", "win", "avg", "tp", "sl", "total"]].to_csv(out, index=False)
    print(f"slices -> {out}")


if __name__ == "__main__":
    main()
