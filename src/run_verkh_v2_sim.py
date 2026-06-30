"""verkh_v2 — long-only "smart zerg rush" engine on fast_v3 up-models.

Decision: go LONG when >=N up-models scream (p>=THR) and (optionally) 0 down-models do.
Exit: deadline at EXIT horizon, OR static TP/SL (the owner's exit idea: take profit at
+TP, stop at -SL, else close at the horizon).

Runs on the saved holdout scores (2-day) or the walk-forward scores (--wf, honest multi-day).

  python -m src.run_verkh_v2_sim                # 2-day holdout
  python -m src.run_verkh_v2_sim --wf           # walk-forward (honest)
  python -m src.run_verkh_v2_sim --wf --tp 1.0 --sl 0.6
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


def fire_mask(s, thr, n_agree, down_veto):
    up = np.zeros(len(s), dtype=int)
    dn = np.zeros(len(s), dtype=int)
    for lab in LABELS:
        up += (s[f"p_up_{lab}"].to_numpy() >= thr).astype(int)
        dn += (s[f"p_down_{lab}"].to_numpy() >= thr).astype(int)
    m = up >= n_agree
    if down_veto:
        m &= dn == 0
    return m


def exit_pnl(real_ret, mfe, mae, tp, sl):
    """Long PnL after cost. If tp/sl set, model a TP/SL exit (conservative: if the
    stop was breached we assume it filled, even if the peak was also reached)."""
    if tp is None and sl is None:
        return real_ret - COST
    out = real_ret.copy()
    if sl is not None:
        hit_sl = mae <= -sl / 100.0
        out = np.where(hit_sl, -sl / 100.0, out)
    if tp is not None:
        hit_tp = mfe >= tp / 100.0
        # only award TP where the stop was NOT breached (conservative ordering)
        award = hit_tp & ~(mae <= -(sl / 100.0)) if sl is not None else hit_tp
        out = np.where(award, tp / 100.0, out)
    return out - COST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", action="store_true", help="use walk-forward scores (honest)")
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--exit", default="20m", choices=LABELS)
    ap.add_argument("--tp", type=float, default=None, help="take-profit %% (else deadline close)")
    ap.add_argument("--sl", type=float, default=None, help="stop-loss %% (positive number)")
    ap.add_argument("--no-down-veto", action="store_true")
    args = ap.parse_args()

    f = "holdout_scores_wf.parquet" if args.wf else "holdout_scores.parquet"
    s = pd.read_parquet(V3_ANALYSIS / f)
    days = s["day"].nunique()
    src = "WALK-FORWARD" if args.wf else "2-day holdout"
    print(f"verkh_v2  [{src}]  rows={len(s)} days={days}  "
          f"rule: LONG if >={args.n} up>= {args.thr}{'' if args.no_down_veto else ' & 0 down'}  "
          f"exit={args.exit}  tp={args.tp} sl={args.sl}\n")

    m = fire_mask(s, args.thr, args.n, not args.no_down_veto)
    d = s[m]
    real = d[f"real_ret_{args.exit}"].to_numpy()
    mfe = d[f"real_mfe_{args.exit}"].to_numpy()
    mae = d[f"real_mae_{args.exit}"].to_numpy()
    pnl = exit_pnl(real, mfe, mae, args.tp, args.sl)

    print(f"{'metric':<16}{'value':>12}")
    print(f"{'signals':<16}{len(d):>12}")
    print(f"{'signals/day':<16}{len(d)//max(days,1):>12}")
    print(f"{'win rate':<16}{(pnl>0).mean():>12.3f}")
    print(f"{'avg %/trade':<16}{pnl.mean()*100:>+12.4f}")
    print(f"{'total %/day':<16}{pnl.sum()/days*100:>+12.2f}")
    print(f"{'median %':<16}{np.median(pnl)*100:>+12.4f}")
    print(f"{'best %':<16}{pnl.max()*100:>+12.3f}")
    print(f"{'worst %':<16}{pnl.min()*100:>+12.3f}")

    print(f"\nper-day:")
    print(f"{'day':<12}{'n':>6}{'win':>8}{'avg%':>9}{'total%':>9}")
    for day, g in d.assign(pnl=pnl).groupby("day"):
        p = g["pnl"].to_numpy()
        print(f"{day:<12}{len(g):>6}{(p>0).mean():>8.3f}{p.mean()*100:>+9.4f}{p.sum()*100:>+9.2f}")


if __name__ == "__main__":
    main()
