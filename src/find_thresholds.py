"""Per-model probability ACTIVATION threshold, found at the SAME 5-min frequency
we trade at (independent-anchor thresholds were too low for dense trading).

Temporal split of the sim grid: FIT on the first `fit_frac` of days, hold out the
rest. Optimal = LOWEST T (more signals) that, on the FIT period, satisfies:
  - n(prob>=T) >= min_signals
  - avg_pnl > 0  and  win_rate >= min_win
  - avg_pnl > 0 in BOTH halves of the fit period   (stable, "no flukes")
Saved to models/prob_thresholds.json; run_sim --thresholds-json tests them on the
held-out days.

Usage:
  python -m src.find_thresholds --fit-frac 0.6 --min-signals 150 --min-win 0.55
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-frac", type=float, default=0.6)
    ap.add_argument("--min-signals", type=int, default=150)
    ap.add_argument("--min-win", type=float, default=0.55)
    ap.add_argument("--slip", type=float, default=0.05)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0

    g = pd.read_parquet(C.DATASETS_DIR / "sim_grid.parquet")
    days = np.sort(pd.to_datetime(g["time"]).dt.normalize().unique())
    n_fit = max(2, int(len(days) * args.fit_frac))
    fit_days, test_days = set(days[:n_fit]), set(days[n_fit:])
    g["d"] = pd.to_datetime(g["time"]).dt.normalize()
    fit = g[g["d"].isin(fit_days)]
    half = np.sort(list(fit_days))[len(fit_days) // 2]
    print(f"fit days {len(fit_days)} ({pd.Timestamp(days[0]).date()}..), "
          f"test days {len(test_days)}\n")

    grid = np.round(np.arange(0.55, 0.931, 0.01), 2)
    result, table = {}, []
    for h in C.HORIZONS:
        lab = h.label
        ex, en = fit[f"exit_{lab}"].to_numpy(), fit["entry_price"].to_numpy()
        ok = np.isfinite(ex)
        ret = ex / en - 1.0
        d = fit["d"].to_numpy()
        for kind, side in (("up", 1), ("down", -1)):
            name = f"{kind}_{lab}"
            prob = fit[f"p_{kind}_{lab}"].to_numpy()
            pnl = side * ret - cost
            chosen = None
            for T in grid:
                m = ok & (prob >= T)
                n = int(m.sum())
                if n < args.min_signals:
                    break
                p = pnl[m]
                h1 = m & (d <= half)
                h2 = m & (d > half)
                if (p.mean() > 0 and (p > 0).mean() >= args.min_win
                        and h1.sum() >= 20 and h2.sum() >= 20
                        and pnl[h1].mean() > 0 and pnl[h2].mean() > 0):
                    chosen = (T, n, (p > 0).mean(), p.mean())
                    break
            result[name] = round(float(chosen[0]), 2) if chosen else None
            table.append({"model": name, "threshold": chosen[0] if chosen else None,
                          "n_fit": chosen[1] if chosen else 0,
                          "win": round(chosen[2], 3) if chosen else None,
                          "avg_pnl%": round(chosen[3] * 100, 4) if chosen else None})

    (C.MODELS_DIR / "prob_thresholds.json").write_text(json.dumps(result, indent=2))
    print(pd.DataFrame(table).to_string(index=False))
    print(f"\nactive: {sum(v is not None for v in result.values())}/{len(result)} "
          f"-> models/prob_thresholds.json")


if __name__ == "__main__":
    main()
