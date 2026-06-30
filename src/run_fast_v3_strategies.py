"""fast_v3 strategy probes on the holdout scores (no retrain). Three directions:

  1) balance  : per-model probability threshold that MAXIMISES net PnL/day
                (frequency x quality, fees already netted).
  2) spread   : does ranking by (p_up - p_down) beat raw p_up on win rate?
  3) unicorn2 : clean agreement engine on the new models — fire LONG when >=N
                up-models scream (p>=thr) and 0 down-models do; measure at an exit horizon.

  python -m src.run_fast_v3_strategies
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .fast import config as FC
from .run_fast_v3 import HORIZONS_V3, V3_ANALYSIS

LABELS = [lab for _, lab, _ in HORIZONS_V3]
COST = FC.EVAL_COST


def _load():
    s = pd.read_parquet(V3_ANALYSIS / "holdout_scores.parquet")
    days = s["day"].nunique() if "day" in s else 1
    return s, days


# ---------- 1) BALANCE: max net PnL/day per model -----------------------------
def balance(s, days):
    print("=== 1) BALANCE — threshold that maximises net PnL/day (longs) ===")
    print(f"{'model':<9}{'best_p':>7}{'/day':>7}{'win':>7}{'avg%':>9}{'total/day':>11}{'TP+':>8}{'SL-':>8}")
    grid = np.round(np.arange(0.50, 0.971, 0.005), 3)
    for _, lab, _ in HORIZONS_V3:
        for side in ("up", "down"):
            p = s[f"p_{side}_{lab}"].to_numpy()
            real = s[f"real_ret_{lab}"].to_numpy()
            mfe = s[f"real_mfe_{lab}"].to_numpy()
            mae = s[f"real_mae_{lab}"].to_numpy()
            sign = 1.0 if side == "up" else -1.0
            best = None
            for thr in grid:
                fire = p >= thr
                n = int(fire.sum())
                if n < days * 2:
                    continue
                pnl = sign * real[fire] - COST
                tot_day = pnl.sum() / days
                if best is None or tot_day > best["tot_day"]:
                    best = {"thr": thr, "n": n, "win": float((pnl > 0).mean()),
                            "avg": float(pnl.mean() * 100), "tot_day": tot_day * 100,
                            "tp": (mfe[fire].mean() if sign > 0 else -mae[fire].mean()) * 100,
                            "sl": (mae[fire].mean() if sign > 0 else -mfe[fire].mean()) * 100}
            if best and side == "up":  # longs only in this bull window
                print(f"{side+'_'+lab:<9}{best['thr']:>7.3f}{best['n']//days:>7}{best['win']:>7.3f}"
                      f"{best['avg']:>+9.4f}{best['tot_day']:>+11.2f}{best['tp']:>+8.3f}{best['sl']:>+8.3f}")
    print()


# ---------- 2) SPREAD: p_up - p_down vs raw p_up ------------------------------
def spread(s, days):
    print("=== 2) SPREAD — fire longs by (p_up - p_down) vs by p_up alone, matched count ===")
    print(f"{'horizon':<8}{'N/day':>7}{'  byP_up: win  avg%':>22}{'  bySPREAD: win  avg%':>24}{'  win Δ':>8}")
    for _, lab, _ in HORIZONS_V3:
        pu = s[f"p_up_{lab}"].to_numpy()
        pd_ = s[f"p_down_{lab}"].to_numpy()
        real = s[f"real_ret_{lab}"].to_numpy()
        spr = pu - pd_
        N = max(int(len(s) * 0.01), days)        # top ~1%
        def stat(score):
            idx = np.argsort(score)[::-1][:N]
            pnl = real[idx] - COST
            return float((pnl > 0).mean()), float(pnl.mean() * 100)
        w1, a1 = stat(pu)
        w2, a2 = stat(spr)
        print(f"{lab:<8}{N//days:>7}{w1:>14.3f}{a1:>+8.4f}{w2:>16.3f}{a2:>+8.4f}{(w2-w1):>+8.3f}")
    print()


# ---------- 3) UNICORN-2: clean agreement engine ------------------------------
def unicorn2(s, days):
    print("=== 3) UNICORN-2 — fire LONG when >=N up-models agree (p>=thr) & 0 down-models ===")
    THR = 0.80
    up_hits = np.zeros(len(s), dtype=int)
    dn_hits = np.zeros(len(s), dtype=int)
    for lab in LABELS:
        up_hits += (s[f"p_up_{lab}"].to_numpy() >= THR).astype(int)
        dn_hits += (s[f"p_down_{lab}"].to_numpy() >= THR).astype(int)
    print(f"agreement threshold p>={THR}")
    for exit_lab in ("8m", "20m"):
        real = s[f"real_ret_{exit_lab}"].to_numpy()
        mfe = s[f"real_mfe_{exit_lab}"].to_numpy()
        mae = s[f"real_mae_{exit_lab}"].to_numpy()
        print(f"  exit@{exit_lab}:  {'N>=':>4}{'/day':>7}{'win':>8}{'avg%':>9}{'total/day':>11}{'TP+':>8}{'SL-':>8}")
        for N in (2, 3, 4):
            fire = (up_hits >= N) & (dn_hits == 0)
            n = int(fire.sum())
            if n == 0:
                print(f"        {N:>4}{0:>7}")
                continue
            pnl = real[fire] - COST
            print(f"        {N:>4}{n//max(days,1):>7}{float((pnl>0).mean()):>8.3f}"
                  f"{float(pnl.mean()*100):>+9.4f}{float(pnl.sum()/days*100):>+11.2f}"
                  f"{mfe[fire].mean()*100:>+8.3f}{mae[fire].mean()*100:>+8.3f}")
    print()


def main():
    s, days = _load()
    print(f"holdout rows={len(s)} days={days} fee={COST*100:.2f}%\n")
    balance(s, days)
    spread(s, days)
    unicorn2(s, days)


if __name__ == "__main__":
    main()
