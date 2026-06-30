"""Unicorn-2 grid: agreement strictness (N, threshold) x exit horizon.

Fire LONG when >=N up-models scream (p>=THR) and 0 down-models do. Then measure
realized PnL closing at each horizon -> answers "how strict?" and "close after how long?".

  python -m src.run_fast_v3_unicorn
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .fast import config as FC
from .run_fast_v3 import HORIZONS_V3, V3_ANALYSIS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LABELS = [lab for _, lab, _ in HORIZONS_V3]
COST = FC.EVAL_COST


def main():
    s = pd.read_parquet(V3_ANALYSIS / "holdout_scores.parquet")
    days = s["day"].nunique() if "day" in s else 1
    print(f"holdout rows={len(s)} days={days} fee={COST*100:.2f}%\n")

    for THR in (0.80, 0.85):
        up_hits = np.zeros(len(s), dtype=int)
        dn_hits = np.zeros(len(s), dtype=int)
        for lab in LABELS:
            up_hits += (s[f"p_up_{lab}"].to_numpy() >= THR).astype(int)
            dn_hits += (s[f"p_down_{lab}"].to_numpy() >= THR).astype(int)
        print(f"################  agreement p>={THR}  ################")
        for N in (3, 4, 5, 6):
            fire = (up_hits >= N) & (dn_hits == 0)
            n = int(fire.sum())
            print(f"\n  N>={N}  signals={n}  ({n//max(days,1)}/day)")
            if n == 0:
                continue
            print(f"    {'exit':>5}{'win':>8}{'avg%':>9}{'total/day':>11}{'TP+':>8}{'SL-':>8}")
            for exit_lab in LABELS:
                real = s[f"real_ret_{exit_lab}"].to_numpy()[fire]
                mfe = s[f"real_mfe_{exit_lab}"].to_numpy()[fire]
                mae = s[f"real_mae_{exit_lab}"].to_numpy()[fire]
                pnl = real - COST
                print(f"    {exit_lab:>5}{float((pnl>0).mean()):>8.3f}{float(pnl.mean()*100):>+9.4f}"
                      f"{float(pnl.sum()/days*100):>+11.2f}{mfe.mean()*100:>+8.3f}{mae.mean()*100:>+8.3f}")
        print()


if __name__ == "__main__":
    main()
