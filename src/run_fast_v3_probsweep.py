"""fast_v3 holdout by ABSOLUTE probability threshold (how loud the model screams).

For each model: fire when p >= thr and report count / win / avg PnL / TP / SL.
This is the real operating knob (live engine uses fixed thresholds like 0.92),
NOT AUC. Reads saved holdout scores, no retrain.

  python -m src.run_fast_v3_probsweep
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .fast import config as FC
from .run_fast_v3 import HORIZONS_V3, V3_ANALYSIS

THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", action="store_true", help="use walk-forward (honest) scores")
    ap.add_argument("--up-only", action="store_true", help="skip down side")
    args = ap.parse_args()
    f = "holdout_scores_wf.parquet" if args.wf else "holdout_scores.parquet"
    s = pd.read_parquet(V3_ANALYSIS / f)
    days = s["day"].nunique() if "day" in s else 1
    print(f"holdout rows={len(s)}  days={days}  fee={FC.EVAL_COST*100:.2f}%")
    print("(n = signals fired at p>=thr; avg% = net PnL/trade; TP/SL = avg favorable/adverse move)\n")

    rows = []
    sides = ("up",) if args.up_only else ("up", "down")
    for m, lab, _ in HORIZONS_V3:
        for side in sides:
            p = s[f"p_{side}_{lab}"].to_numpy()
            real = s[f"real_ret_{lab}"].to_numpy()
            mfe = s[f"real_mfe_{lab}"].to_numpy()
            mae = s[f"real_mae_{lab}"].to_numpy()
            sign = 1.0 if side == "up" else -1.0
            print(f"{side}_{lab}   pmax={p.max():.3f}  p99={np.quantile(p,0.99):.3f}  "
                  f"p999={np.quantile(p,0.999):.3f}")
            print(f"   {'p>=':>5}{'n':>8}{'/day':>7}{'win':>8}{'avg%':>9}{'TP+':>8}{'SL-':>8}{'total%':>9}")
            for thr in THRESHOLDS:
                fire = p >= thr
                n = int(fire.sum())
                if n == 0:
                    print(f"   {thr:>5.2f}{0:>8}{'-':>7}")
                    continue
                r = real[fire]
                pnl = sign * r - FC.EVAL_COST
                tp = (mfe[fire].mean() if sign > 0 else -mae[fire].mean()) * 100
                sl = (mae[fire].mean() if sign > 0 else -mfe[fire].mean()) * 100
                win = float((pnl > 0).mean())
                avg = float(pnl.mean() * 100)
                tot = float(pnl.sum() * 100)
                print(f"   {thr:>5.2f}{n:>8}{n//max(days,1):>7}{win:>8.3f}{avg:>+9.4f}"
                      f"{tp:>+8.3f}{sl:>+8.3f}{tot:>+9.2f}")
                rows.append({"model": f"{side}_{lab}", "p_thr": thr, "n": n,
                             "per_day": n // max(days, 1), "win": win, "avg": avg,
                             "tp": tp, "sl": sl, "total": tot})
            print()
    pd.DataFrame(rows).to_csv(V3_ANALYSIS / "holdout_probsweep.csv", index=False)
    print(f"probsweep -> {V3_ANALYSIS/'holdout_probsweep.csv'}")


if __name__ == "__main__":
    main()
