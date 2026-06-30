"""Per-day stats for the short-horizon bluechip model, one block per horizon.
Goal: see whether SHORTER horizon => smaller 'crash' (worst-day drawdown).

Selection = per-day top-K longs by p_up (the 'Rebound' style). Per day we report
n / win / avg% / $-day @ notional. Then a horizon-comparison summary sorted by
worst-day $ (the posos) so the smallest-drawdown horizon is obvious.

  python -m src.run_short_perday --tag bluechip_short --k 50 --days 7
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
COST = 0.0012
HORIZONS = ["4m", "8m", "12m", "18m", "24m", "32m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bluechip_short")
    ap.add_argument("--experiment", default="fast_bluechip")
    ap.add_argument("--k", type=int, default=50, help="top-k longs per day")
    ap.add_argument("--notional", type=float, default=30.0)
    ap.add_argument("--days", type=int, default=7, help="show last N days")
    ap.add_argument("--horizons", default=",".join(HORIZONS))
    args = ap.parse_args()
    horizons = args.horizons.split(",")
    N = args.notional

    path = f"outputs/analysis/{args.experiment}/{args.tag}/holdout_scores.parquet"
    cols = ["symbol", "anchor_time", "day"]
    for h in horizons:
        cols += [f"p_up_{h}", f"real_ret_{h}"]
    s = pd.read_parquet(path, columns=cols)
    all_days = sorted(s["day"].unique())
    last = all_days[-args.days:]
    print(f"file={path}")
    print(f"holdout has {len(all_days)} days: {all_days[0]} -> {all_days[-1]}")
    print(f"showing per-day top-{args.k} longs for the last {len(last)} days, ${N:.0f}/position, cost={COST*100:.2f}%\n")

    # build per-(day,horizon) metrics
    dollars = {}  # (day,h) -> $day
    win = {}      # (day,h) -> win
    avg = {}      # (day,h) -> avg%
    for h in horizons:
        s["rk"] = s.groupby("day")[f"p_up_{h}"].rank(ascending=False, method="first")
        top = s[s.rk <= args.k].copy()
        top["pnl"] = top[f"real_ret_{h}"] - COST
        for d, x in top.groupby("day"):
            p = x["pnl"].to_numpy()
            dollars[(d, h)] = p.sum() * N
            win[(d, h)] = (p > 0).mean()
            avg[(d, h)] = p.mean() * 100

    def matrix(title, data, fmt, totals=True):
        print(f"=== {title}  (top-{args.k} longs/day, ${N:.0f}/pos) ===")
        head = f"{'day':<12}" + "".join(f"{h:>9}" for h in horizons)
        print(head)
        for d in last:
            row = f"{d:<12}" + "".join(fmt(data.get((d, h), float('nan'))) for h in horizons)
            print(row)
        if totals:
            print("-" * len(head))
            tot = f"{'TOTAL':<12}" + "".join(
                f"{sum(data.get((d, h), 0.0) for d in last):>+9.0f}" for h in horizons)
            print(tot)
            mean = f"{'mean/day':<12}" + "".join(
                f"{np.mean([data[(d, h)] for d in last if (d, h) in data]):>+9.2f}" for h in horizons)
            print(mean)
            posos = f"{'posos':<12}" + "".join(
                f"{min(data.get((d, h), 0.0) for d in last):>+9.2f}" for h in horizons)
            print(posos)
            green = f"{'green':<12}" + "".join(
                f"{np.mean([data[(d, h)] > 0 for d in last if (d, h) in data]):>9.2f}" for h in horizons)
            print(green)
        print()

    matrix("$/DAY by day x horizon", dollars, lambda v: f"{v:>+9.2f}")
    matrix("WIN-RATE by day x horizon", win, lambda v: f"{v:>9.3f}", totals=False)
    matrix("AVG% by day x horizon", avg, lambda v: f"{v:>+9.3f}", totals=False)

    print("rows = each day; columns = horizons. posos = worst single-day $ per horizon.")


if __name__ == "__main__":
    main()
