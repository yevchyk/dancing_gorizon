"""Probability-zone report from the sim grid (ALL models, NO trust filter).
Realized PnL = fixed-horizon close, $10/trade.

  A. per-horizon overall (incl 5m/15m) -- how each horizon/side actually does
  B. probability zones (step 0.05): n trades, win, avg pnl, $ -- per zone
  C. UP/DOWN disagreement: where up says low & down high (and vice versa)
  D. calibration: predicted prob zone vs realized win-rate

Usage:
  python -m src.run_prob_zones --size 10 --step 0.05
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0


def candidates(g: pd.DataFrame, cost: float) -> pd.DataFrame:
    out = []
    for h in C.HORIZONS:
        lab = h.label
        ex = g[f"exit_{lab}"].to_numpy()
        for kind, side in (("up", 1), ("down", -1)):
            prob = (g[f"p_up_{lab}"] if kind == "up" else g[f"p_down_{lab}"]).to_numpy()
            pnl = side * (ex / g["entry_price"].to_numpy() - 1.0) - cost
            d = pd.DataFrame({"horizon": lab, "side": kind, "prob": prob, "pnl": pnl})
            out.append(d[np.isfinite(ex)])
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--slip", type=float, default=0.05)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0
    g = pd.read_parquet(C.DATASETS_DIR / "sim_grid.parquet")
    c = candidates(g, cost)

    def line(d):
        return (f"n={len(d):>6} win={ (d.pnl>0).mean():.3f} "
                f"avg={d.pnl.mean()*100:+.4f}% $={d.pnl.sum()*args.size:+8.2f}")

    print("=== A. PER-HORIZON x SIDE (prob>=0.50) ===")
    for h in C.HORIZONS:
        for kind in ("up", "down"):
            d = c[(c.horizon == h.label) & (c.side == kind) & (c.prob >= 0.50)]
            if len(d):
                print(f"  {kind}_{h.label:<3} {line(d)}")

    print(f"\n=== B. PROBABILITY ZONES (all models, step {args.step}) ===")
    edges = np.round(np.arange(0.40, 1.0001, args.step), 3)
    print(f"  {'zone':>12} {'n':>7} {'win':>6} {'avg_pnl':>9} {'$ (10/trade)':>12}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        d = c[(c.prob >= lo) & (c.prob < hi)]
        if len(d) >= 20:
            print(f"  [{lo:.2f},{hi:.2f}) {len(d):>7} {(d.pnl>0).mean():>6.3f} "
                  f"{d.pnl.mean()*100:>+8.4f}% {d.pnl.sum()*args.size:>+11.2f}")

    print("\n=== C. UP/DOWN DISAGREEMENT (per horizon) ===")
    for h in C.HORIZONS:
        lab = h.label
        pu, pdn = g[f"p_up_{lab}"].to_numpy(), g[f"p_down_{lab}"].to_numpy()
        ex, en = g[f"exit_{lab}"].to_numpy(), g["entry_price"].to_numpy()
        ok = np.isfinite(ex)
        ret = ex / en - 1.0
        def stat(mask, side_arr):
            m = mask & ok
            if m.sum() < 20:
                return "(few)"
            pnl = side_arr[m] * ret[m] - cost if hasattr(side_arr, "__len__") \
                else side_arr * ret[m] - cost
            return f"n={int(m.sum()):>5} win={(pnl>0).mean():.3f} avg={pnl.mean()*100:+.4f}%"
        up_hi = (pu >= 0.60) & (pdn <= 0.40)
        dn_hi = (pdn >= 0.60) & (pu <= 0.40)
        both = (pu >= 0.60) & (pdn >= 0.60)
        argmax_side = np.where(pu >= pdn, 1, -1)
        print(f"  {lab:>3}:")
        print(f"       up_hi & dn_lo -> LONG : {stat(up_hi, 1)}")
        print(f"       dn_hi & up_lo -> SHORT: {stat(dn_hi, -1)}")
        print(f"       both_hi -> argmax     : {stat(both, argmax_side)}")

    print(f"\n=== E. PER-MODEL probability zones (step {args.step}, $10/trade) ===")
    for h in C.HORIZONS:
        for kind in ("up", "down"):
            cm = c[(c.horizon == h.label) & (c.side == kind)]
            print(f"  --- {kind}_{h.label} ---")
            for lo, hi in zip(edges[:-1], edges[1:]):
                d = cm[(cm.prob >= lo) & (cm.prob < hi)]
                if len(d) >= 15:
                    print(f"    [{lo:.2f},{hi:.2f}) n={len(d):>5} win={(d.pnl>0).mean():.3f} "
                          f"avg={d.pnl.mean()*100:+.4f}% $={d.pnl.sum()*args.size:+8.2f}")

    print("\n=== D. CALIBRATION (predicted prob zone -> realized win) ===")
    for lo, hi in zip(edges[:-1], edges[1:]):
        d = c[(c.prob >= lo) & (c.prob < hi)]
        if len(d) >= 50:
            real_dir = (d.pnl + cost > 0)  # direction correct (before cost)
            print(f"  [{lo:.2f},{hi:.2f}) predicted~{(lo+hi)/2:.2f} -> "
                  f"realized_win={real_dir.mean():.3f} n={len(d)}")


if __name__ == "__main__":
    main()
