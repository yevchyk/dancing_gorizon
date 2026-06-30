"""Per-model review of the v4 engine on the sim grid: $ earned, signals,
stability, and win-rate by probability zone from the floor up.

Candidates = v4 rule (prob >= SIGNAL_FLOOR AND opp <= CLEAN_OPP_MAX). $10/trade.
This is RAW per-model signal quality (every qualifying signal, not the top-3/scan
the equity sim takes) -- so the engine's components can be judged individually.

Usage:
  python -m src.run_engine_review --floor 0.82 --size 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=C.SIGNAL_FLOOR)
    ap.add_argument("--clean-opp", type=float, default=C.CLEAN_OPP_MAX)
    ap.add_argument("--size", type=float, default=10.0)
    ap.add_argument("--slip", type=float, default=0.05)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0
    g = pd.read_parquet(C.DATASETS_DIR / "sim_grid.parquet")
    day = pd.to_datetime(g["time"]).dt.normalize()
    days = np.sort(day.unique())
    mid = days[len(days) // 2]

    rows = []
    for h in C.HORIZONS:
        lab = h.label
        ex, en = g[f"exit_{lab}"].to_numpy(), g["entry_price"].to_numpy()
        ok = np.isfinite(ex)
        ret = ex / en - 1.0
        for kind, side in (("up", 1), ("down", -1)):
            prob = g[f"p_up_{lab}"].to_numpy() if kind == "up" else g[f"p_down_{lab}"].to_numpy()
            opp = g[f"p_down_{lab}"].to_numpy() if kind == "up" else g[f"p_up_{lab}"].to_numpy()
            m = ok & (prob >= args.floor) & (opp <= args.clean_opp)
            rows.append(pd.DataFrame({"model": f"{kind}_{lab}", "prob": prob[m],
                                      "pnl": side * ret[m] - cost, "day": day.to_numpy()[m]}))
    c = pd.concat(rows, ignore_index=True)

    print(f"=== PER-MODEL REVIEW (floor>={args.floor}, opp<={args.clean_opp}, ${args.size}/trade) ===")
    print(f"  {'model':<9} {'n':>5} {'win':>5} {'avg%':>8} {'$':>8} "
          f"{'+days':>7} {'stable':>7}")
    summ = []
    for m, d in c.groupby("model"):
        daily = d.groupby("day")["pnl"].sum()
        pos_days = (daily > 0).mean()
        h1 = d[d.day <= mid]["pnl"].mean()
        h2 = d[d.day > mid]["pnl"].mean()
        stable = "yes" if (h1 > 0 and h2 > 0) else "no"
        tot = d.pnl.sum() * args.size
        print(f"  {m:<9} {len(d):>5} {(d.pnl>0).mean():>5.3f} {d.pnl.mean()*100:>+7.3f}% "
              f"{tot:>+7.2f} {pos_days:>6.0%} {stable:>7}")
        summ.append((m, tot))

    print(f"\n  TOTAL across models: ${sum(t for _, t in summ):+.2f} "
          f"(raw, all signals; equity sim caps at top-{C.CONF_TOP_PER_SCAN}/scan)")

    print(f"\n=== WIN-RATE by PROBABILITY ZONE (from floor {args.floor}) ===")
    print(f"  {'model':<9} " + "".join(f"{z:>14}" for z in
          ("[.82,.85)", "[.85,.90)", "[.90,1.0)")))
    for m, d in c.groupby("model"):
        cells = []
        for lo, hi in [(0.82, 0.85), (0.85, 0.90), (0.90, 1.01)]:
            z = d[(d.prob >= lo) & (d.prob < hi)]
            cells.append(f"{z.pnl.gt(0).mean():.2f}/{len(z)}" if len(z) >= 5 else "  -/-")
        print(f"  {m:<9} " + "".join(f"{x:>14}" for x in cells))


if __name__ == "__main__":
    main()
