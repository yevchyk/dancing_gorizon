"""Two-engine A/B sim with $ sizing + leverage, on fast_v3 scores.

  verkh_v2   : FLAT POOL (long-only). Each up-model fires INDEPENDENTLY when it
               screams past its own threshold; exit at that model's own horizon.
               Union of all triggers = "smart zerg rush".  $10 x 3x.
  unicorn_v2 : AGREEMENT. Fire only when the WHOLE side screams together
               (>=N models agree, 0 opposite); exit 20m. Both sides.  $10 x 6x.

PnL$ = size_usd * leverage * (side*ret - cost). Run on 2-day holdout or --wf.

  python -m src.run_engines_v2_sim --wf
"""

from __future__ import annotations

import argparse
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

# verkh_v2 flat pool: tuned 2026-06-03 after live bleed on short horizons.
# up_1m dropped entirely (its move is smaller than real fee+slippage even at win 0.66).
VERKH_THRESH = {"2m": 0.95, "4m": 0.95, "8m": 0.92, "12m": 0.90, "20m": 0.85}
VERKH_SIZE, VERKH_LEV = 10, 3

# unicorn_v2 agreement
UNI_THR, UNI_N, UNI_EXIT = 0.85, 4, "20m"
UNI_SIZE, UNI_LEV = 10, 6


def verkh(s, days):
    print(f"--- verkh_v2  FLAT POOL long-only  (${VERKH_SIZE} x {VERKH_LEV}x = ${VERKH_SIZE*VERKH_LEV} notional) ---")
    print(f"    {'model':<8}{'p>=':>6}{'n':>7}{'/day':>6}{'win':>7}{'avg%':>9}{'$/day':>9}")
    all_usd, all_day = [], []
    for lab in LABELS:
        if lab not in VERKH_THRESH:
            continue
        thr = VERKH_THRESH[lab]
        p = s[f"p_up_{lab}"].to_numpy()
        fire = p >= thr
        n = int(fire.sum())
        if n == 0:
            print(f"    up_{lab:<5}{thr:>6.2f}{0:>7}"); continue
        pct = s[f"real_ret_{lab}"].to_numpy()[fire] - COST
        usd = VERKH_SIZE * VERKH_LEV * pct
        all_usd.append(usd); all_day.append(s["day"].to_numpy()[fire])
        print(f"    up_{lab:<5}{thr:>6.2f}{n:>7}{n//max(days,1):>6}{(pct>0).mean():>7.3f}"
              f"{pct.mean()*100:>+9.4f}{usd.sum()/days:>+9.2f}")
    usd = np.concatenate(all_usd); day = np.concatenate(all_day)
    print(f"    {'UNION':<8}{'':>6}{len(usd):>7}{len(usd)//max(days,1):>6}{(usd>0).mean():>7.3f}"
          f"{'':>9}{usd.sum()/days:>+9.2f}")
    per = pd.DataFrame({"day": day, "usd": usd}).groupby("day")["usd"].sum()
    print(f"    total ${usd.sum():+.2f} over {days}d   per-day: "
          + "  ".join(f"{d[5:]}:{v:+.1f}" for d, v in per.items()))


def unicorn(s, days):
    up = np.zeros(len(s), int); dn = np.zeros(len(s), int)
    for lab in LABELS:
        up += (s[f"p_up_{lab}"].to_numpy() >= UNI_THR).astype(int)
        dn += (s[f"p_down_{lab}"].to_numpy() >= UNI_THR).astype(int)
    print(f"--- unicorn_v2  AGREEMENT >= {UNI_N} @ {UNI_THR}  exit {UNI_EXIT}  "
          f"(${UNI_SIZE} x {UNI_LEV}x = ${UNI_SIZE*UNI_LEV} notional, both sides) ---")
    print(f"    {'side':<7}{'n':>7}{'/day':>6}{'win':>7}{'avg%':>9}{'$/day':>9}")
    all_usd, all_day = [], []
    for side, fire, sign in (("long", (up >= UNI_N) & (dn == 0), 1.0),
                             ("short", (dn >= UNI_N) & (up == 0), -1.0)):
        n = int(fire.sum())
        if n == 0:
            print(f"    {side:<7}{0:>7}"); continue
        pct = sign * s[f"real_ret_{UNI_EXIT}"].to_numpy()[fire] - COST
        usd = UNI_SIZE * UNI_LEV * pct
        all_usd.append(usd); all_day.append(s["day"].to_numpy()[fire])
        print(f"    {side:<7}{n:>7}{n//max(days,1):>6}{(pct>0).mean():>7.3f}"
              f"{pct.mean()*100:>+9.4f}{usd.sum()/days:>+9.2f}")
    usd = np.concatenate(all_usd); day = np.concatenate(all_day)
    print(f"    {'TOTAL':<7}{len(usd):>7}{len(usd)//max(days,1):>6}{(usd>0).mean():>7.3f}"
          f"{'':>9}{usd.sum()/days:>+9.2f}")
    per = pd.DataFrame({"day": day, "usd": usd}).groupby("day")["usd"].sum()
    print(f"    total ${usd.sum():+.2f} over {days}d   per-day: "
          + "  ".join(f"{d[5:]}:{v:+.1f}" for d, v in per.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", action="store_true")
    ap.add_argument("--tag", default="", help="WF scores suffix, e.g. _bear")
    ap.add_argument("--last", type=int, default=0, help="keep only the last N days")
    ap.add_argument("--only", choices=["verkh", "unicorn", "both"], default="both")
    args = ap.parse_args()
    f = f"holdout_scores_wf{args.tag}.parquet" if args.wf else "holdout_scores.parquet"
    s = pd.read_parquet(V3_ANALYSIS / f)
    if args.last:
        keep = sorted(s["day"].unique())[-args.last:]
        s = s[s["day"].isin(keep)]
    days = s["day"].nunique()
    src = "WALK-FORWARD (honest)" if args.wf else "2-day holdout (bull)"
    print(f"=== engines v2  [{src}]  days={days} ({', '.join(sorted(s['day'].unique()))}) ===\n")
    if args.only in ("verkh", "both"):
        verkh(s, days)
        print()
    if args.only in ("unicorn", "both"):
        unicorn(s, days)


if __name__ == "__main__":
    main()
