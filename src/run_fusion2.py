"""Fusion round 2: fair leave-one-day-out stacking of B (direction) + C (market
listener), per-day breakdown, normal-days-only aggregate, and C-down crash-veto.
A is dropped (corr 0.99 with B -> redundant).

  python -m src.run_fusion2 --horizon 32m
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
COST = 0.0012
BASE = "outputs/analysis/fast_bluechip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="32m")
    ap.add_argument("--k", type=int, default=50, help="top-k/day for matched compare")
    ap.add_argument("--veto", type=float, default=0.60, help="C-down crash-veto threshold")
    args = ap.parse_args()
    h = args.horizon

    b = pd.read_parquet(f"{BASE}/bluechip/holdout_scores.parquet",
                        columns=["symbol", "anchor_time", "day", f"p_up_{h}", f"real_ret_{h}"])
    c = pd.read_parquet(f"{BASE}/listener/holdout_scores.parquet",
                        columns=["symbol", "anchor_time", f"p_up_{h}", f"p_down_{h}"])
    m = b.merge(c, on=["symbol", "anchor_time"], suffixes=("_B", "_C"))
    m = m.rename(columns={f"real_ret_{h}": "ret"})
    day = m["day"].to_numpy(); ret = m["ret"].to_numpy()
    pB = m[f"p_up_{h}_B"].to_numpy(); pC = m[f"p_up_{h}_C"].to_numpy(); pCd = m[f"p_down_{h}"].to_numpy()
    days = sorted(set(day))
    print(f"horizon={h} rows={len(m)} days={days}")

    # market direction per day (label bull/crash)
    print("\nday        mean_ret%  type")
    crash = []
    for d in days:
        mr = ret[day == d].mean() * 100
        t = "CRASH" if mr < -0.05 else ("bull" if mr > 0.02 else "flat")
        if t == "CRASH":
            crash.append(d)
        print(f"  {d}  {mr:+7.3f}   {t}")

    # leave-one-day-out stacking on [pB, pC]
    X = np.column_stack([pB, pC]); y = (ret > COST).astype(int)
    stack = np.full(len(m), np.nan)
    for d in days:
        tr = day != d
        meta = LogisticRegression(max_iter=600).fit(X[tr], y[tr])
        stack[day == d] = meta.predict_proba(X[day == d])[:, 1]

    def topk_day(score, dd_mask):
        # per-day top-k by score within dd_mask
        df = pd.DataFrame({"s": score, "ret": ret, "day": day})[dd_mask]
        df["rk"] = df.groupby("day")["s"].rank(ascending=False, method="first")
        sel = df[df.rk <= args.k]
        pnl = sel["ret"].to_numpy() - COST
        return len(sel), float((pnl > 0).mean()), float(pnl.mean() * 100)

    allmask = np.ones(len(m), bool)
    normal = ~np.isin(day, crash)
    print(f"\n=== matched top-{args.k}/day: B vs STACK(B,C) ===")
    print(f"{'subset':<16}{'B win':>8}{'B avg%':>9}{'STACK win':>11}{'STACK avg%':>11}{'win Δ':>8}")
    for label, mask in [("ALL days", allmask), ("NORMAL only", normal), ("CRASH only", np.isin(day, crash))]:
        if mask.sum() == 0:
            continue
        nb, wb, ab = topk_day(pB, mask)
        ns, ws, as_ = topk_day(stack, mask)
        print(f"{label:<16}{wb:>8.3f}{ab:>+9.4f}{ws:>11.3f}{as_:>+11.4f}{ws-wb:>+8.3f}")

    # variation: C-down crash veto on B-longs
    print(f"\n=== C-down crash-veto (B>=0.85 long, skip if C.p_down>= {args.veto}) ===")
    for label, mask in [("ALL", allmask), ("NORMAL", normal), ("CRASH", np.isin(day, crash))]:
        fB = (pB >= 0.85) & mask
        fV = fB & (pCd < args.veto)
        for nm, f in [("B no veto", fB), ("B + C-down veto", fV)]:
            if f.sum() >= 3:
                pnl = ret[f] - COST
                print(f"  {label:<7}{nm:<18} n={int(f.sum()):<5} win={(pnl>0).mean():.3f} avg%={pnl.mean()*100:+.4f}")


if __name__ == "__main__":
    main()
