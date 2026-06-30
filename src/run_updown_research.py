"""Deeper look at the up/down probability RELATIONSHIP on the sim grid.

  A. Among high-conf signals (p_dir>=thr), does the OPPOSITE side's level matter?
     (conflict: opp also high -> whipsaw?)
  B. SPREAD (p_dir - p_opp) as a signal vs raw p_dir.
  C. SUM (p_up + p_down) as a volatility/whipsaw indicator.
  D. Does a 'clean' filter (opp <= X) on top of p_dir>=thr improve win/PnL?

Usage:
  python -m src.run_updown_research --thr 0.85
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0


def long_short_table(g, cost):
    rows = []
    for h in C.HORIZONS:
        lab = h.label
        ex, en = g[f"exit_{lab}"].to_numpy(), g["entry_price"].to_numpy()
        ok = np.isfinite(ex)
        ret = ex / en - 1.0
        rows.append(pd.DataFrame({"side": "long", "dir": g[f"p_up_{lab}"].to_numpy(),
                                  "opp": g[f"p_down_{lab}"].to_numpy(),
                                  "pnl": ret - cost, "ok": ok}))
        rows.append(pd.DataFrame({"side": "short", "dir": g[f"p_down_{lab}"].to_numpy(),
                                  "opp": g[f"p_up_{lab}"].to_numpy(),
                                  "pnl": -ret - cost, "ok": ok}))
    c = pd.concat(rows, ignore_index=True)
    return c[c.ok]


def line(d):
    return f"n={len(d):>5} win={(d.pnl>0).mean():.3f} avg={d.pnl.mean()*100:+.4f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--slip", type=float, default=0.05)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0
    g = pd.read_parquet(C.DATASETS_DIR / "sim_grid.parquet")
    c = long_short_table(g, cost)

    print(f"=== A. p_dir>={args.thr}: does OPPOSITE side level matter? ===")
    hi = c[c.dir >= args.thr]
    for lo, h in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]:
        d = hi[(hi.opp >= lo) & (hi.opp < h)]
        if len(d) >= 20:
            print(f"  opp in [{lo:.1f},{h:.1f}): {line(d)}")

    print("\n=== B. SPREAD (p_dir - p_opp) zones ===")
    c["spread"] = c.dir - c.opp
    for lo, h in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        d = c[(c.spread >= lo) & (c.spread < h)]
        if len(d) >= 50:
            print(f"  spread [{lo:.1f},{h:.1f}): {line(d)}")

    print("\n=== C. SUM (p_up + p_down) -- whipsaw indicator (long side) ===")
    # use long rows only to avoid double count; sum is symmetric
    longs = c[c.side == "long"].copy()
    longs["psum"] = longs.dir + longs.opp
    for lo, h in [(0.6, 0.9), (0.9, 1.1), (1.1, 1.3), (1.3, 1.6)]:
        d = longs[(longs.psum >= lo) & (longs.psum < h)]
        if len(d) >= 50:
            print(f"  p_up+p_down [{lo:.1f},{h:.1f}): {line(d)}")

    print(f"\n=== D. CLEAN filter: p_dir>={args.thr} AND opp<=X ===")
    for x in [1.01, 0.5, 0.4, 0.3, 0.2]:
        d = c[(c.dir >= args.thr) & (c.opp <= x)]
        tag = "no filter" if x > 1 else f"opp<={x:.1f}"
        if len(d) >= 20:
            print(f"  {tag:<10}: {line(d)}")

    print(f"\n=== E. compare: raw p_dir>={args.thr}  vs  spread>={args.thr} ===")
    a = c[c.dir >= args.thr]
    b = c[c.spread >= args.thr]
    print(f"  p_dir>={args.thr}  : {line(a)}")
    print(f"  spread>={args.thr} : {line(b)}")


if __name__ == "__main__":
    main()
