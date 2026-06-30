"""Reconcile the live v2_pair trades against the candle store + summarise real PnL.

For each closed trade: recompute what the candles say the entry/exit should be and
compare to the logged fill. If logged ~= candle-implied, execution is honest.
Also nets out the roundtrip fee (the logged pnl_pct is GROSS price move only).

  python -m src.run_v2_recon                       # latest live run
  python -m src.run_v2_recon live_20260603_131216
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore

HMIN = {"1m": 1, "2m": 2, "4m": 4, "8m": 8, "12m": 12, "20m": 20}
ROUNDTRIP_FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0 * 100  # in %


def run_dir() -> str:
    base = C.OUTPUTS_DIR / "trading_logs"
    runs = sorted(p.name for p in base.glob("live_*"))
    return sys.argv[1] if len(sys.argv) > 1 else runs[-1]


def main() -> None:
    rd = run_dir()
    path = C.OUTPUTS_DIR / "trading_logs" / rd / "trades.csv"
    t = pd.read_csv(path)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    closed = t[t["event"] == "deadline_close"].copy()
    opens = t[t["event"] == "open"]
    print(f"run={rd}  opens={len(opens)} closed={len(closed)} still_open={len(opens)-len(closed)}\n")

    # --- real PnL summary per engine (net = gross - roundtrip fee) ---
    print(f"=== REAL PnL (fee {ROUNDTRIP_FEE:.2f}% netted) ===")
    print(f"{'engine':<11}{'n':>4}{'win_net':>9}{'gross%':>9}{'net%':>9}{'net$':>9}")
    for eng, g in closed.groupby("engine"):
        gross = g["pnl_pct"].to_numpy()
        net = gross - ROUNDTRIP_FEE
        usd = (net / 100.0) * g["size_usd"].to_numpy()
        print(f"{eng:<11}{len(g):>4}{(net>0).mean():>9.3f}{gross.sum():>+9.3f}"
              f"{net.sum():>+9.3f}{usd.sum():>+9.2f}")
    allnet = closed["pnl_pct"].to_numpy() - ROUNDTRIP_FEE
    allusd = (allnet / 100.0) * closed["size_usd"].to_numpy()
    print(f"{'TOTAL':<11}{len(closed):>4}{(allnet>0).mean():>9.3f}"
          f"{closed['pnl_pct'].sum():>+9.3f}{allnet.sum():>+9.3f}{allusd.sum():>+9.2f}")

    # --- recon vs candles: did the logged FILLS match the market at those moments? ---
    store = CandleStore(C.CANDLES_DIR)
    print(f"\n=== RECON: logged fill price vs candle close at actual open/close time ===")
    print(f"{'symbol':<15}{'side':<6}{'h':>4}{'entry_slip%':>12}{'exit_slip%':>11}")
    eslip, xslip = [], []
    for r in closed.itertuples(index=False):
        # match this close to its open row (same symbol/model/entry_price) for the real open ts
        cand = opens[(opens.symbol == r.symbol) & (opens.model == r.model)
                     & (np.isclose(opens.entry_price, r.entry_price))]
        if cand.empty:
            continue
        open_ts = cand["ts"].iloc[-1]
        c = store.load(r.symbol)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        try:
            ce = float(c.loc[:open_ts, "close"].iloc[-1])
            cx = float(c.loc[:r.ts, "close"].iloc[-1])
        except Exception:
            continue
        es = (r.entry_price / ce - 1.0) * 100
        xs = (r.exit_price / cx - 1.0) * 100
        eslip.append(es); xslip.append(xs)
        print(f"{r.symbol:<15}{r.side:<6}{r.horizon:>4}{es:>+12.3f}{xs:>+11.3f}")
    if eslip:
        e, x = np.array(eslip), np.array(xslip)
        print(f"\nfill vs candle: entry mean={e.mean():+.3f}% (|max|={np.abs(e).max():.3f}), "
              f"exit mean={x.mean():+.3f}% (|max|={np.abs(x).max():.3f})")
        print("small/zero => fills track the market; large => real slippage vs candle close")


if __name__ == "__main__":
    main()
