"""Hour-of-day analysis on the OOS sim grid (models never saw this window).
Builds the v4 clean+agree signals at every 5-min step and buckets realized PnL by
UTC hour -> honest answer to 'are some hours bad / trade at night?'.

Usage:
  python -m src.run_hour_grid
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
FLOOR, OPP, AGREE = 0.82, 0.30, 2
EXCL = {"down_1h", "down_2h"}


def side_signal(g, kind, cost):
    """Return (mask agree>=2, pnl) for one direction across horizons."""
    labs = [h.label for h in C.HORIZONS]
    sgn = 1 if kind == "up" else -1
    ex_en = {lab: (g[f"exit_{lab}"].to_numpy() / g["entry_price"].to_numpy() - 1.0) for lab in labs}
    spreads, cleans, pnls = [], [], []
    for lab in labs:
        if f"{kind}_{lab}" in EXCL:
            spreads.append(np.full(len(g), -np.inf)); cleans.append(np.zeros(len(g), bool))
            pnls.append(np.full(len(g), np.nan)); continue
        p = g[f"p_{kind}_{lab}"].to_numpy()
        opp = (g[f"p_down_{lab}"] if kind == "up" else g[f"p_up_{lab}"]).to_numpy()
        ok = np.isfinite(g[f"exit_{lab}"].to_numpy())
        clean = ok & (p >= FLOOR) & (opp <= OPP)
        cleans.append(clean)
        spreads.append(np.where(clean, p - opp, -np.inf))
        pnls.append(sgn * ex_en[lab] - cost)
    agree = np.sum(cleans, axis=0)
    best = np.argmax(np.array(spreads), axis=0)
    pnl_arr = np.array(pnls)
    best_pnl = pnl_arr[best, np.arange(len(g))]
    take = agree >= AGREE
    return take, best_pnl


def main() -> None:
    g = pd.read_parquet(C.DATASETS_DIR / "sim_grid.parquet")
    g["time"] = pd.to_datetime(g["time"], utc=True)
    cost = FEE + 0.0005
    rows = []
    for kind in ("up", "down"):
        take, pnl = side_signal(g, kind, cost)
        rows.append(pd.DataFrame({"hour": g["time"].dt.hour.to_numpy()[take],
                                  "pnl": pnl[take]}))
    df = pd.concat(rows, ignore_index=True).dropna()
    df["won"] = df.pnl > 0
    print(f"OOS clean signals: {len(df)}  (grid {g['time'].min().date()}..{g['time'].max().date()})\n")
    print(f"  {'hour':>4} {'n':>5} {'win':>6} {'avg_pnl':>9}")
    for hr, gg in df.groupby("hour"):
        if len(gg) >= 15:
            print(f"  {hr:>4} {len(gg):>5} {gg.won.mean():>6.3f} {gg.pnl.mean()*100:>+8.4f}%")
    night = df[(df.hour >= 20) | (df.hour < 8)]
    day = df[(df.hour >= 8) & (df.hour < 20)]
    print(f"\n  DAY  (08-20 UTC): n={len(day):>5} win={day.won.mean():.3f} avg={day.pnl.mean()*100:+.4f}%")
    print(f"  NIGHT(20-08 UTC): n={len(night):>5} win={night.won.mean():.3f} avg={night.pnl.mean()*100:+.4f}%")


if __name__ == "__main__":
    main()
