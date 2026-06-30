"""Full live P&L reconstruction + direction-inversion audit.

Closed trades use the logged fill. The positions that were still OPEN when the
engine was killed (and closed by hand) are marked to the latest candle so the
REAL total is visible. Also checks whether 'long' actually aligned with the move
(rules out a sign/direction inversion).

  python -m src.run_v2_full_pnl
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore

HMIN = {"1m": 1, "2m": 2, "4m": 4, "8m": 8, "12m": 12, "20m": 20}
FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0 * 100  # roundtrip %, taker both sides ignored extra


def main():
    base = C.OUTPUTS_DIR / "trading_logs"
    rd = sys.argv[1] if len(sys.argv) > 1 else sorted(p.name for p in base.glob("live_*"))[-1]
    t = pd.read_csv(base / rd / "trades.csv")
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    opens = t[t.event == "open"]
    closes = t[t.event != "open"]
    store = CandleStore(C.CANDLES_DIR)

    closed_keys = set(zip(closes.symbol, closes.model, closes.entry_price))

    print(f"run={rd}  roundtrip_fee={FEE:.2f}%\n")

    # --- assemble every position: closed (logged) + still-open (mark to last candle) ---
    recs = []
    for r in closes.itertuples(index=False):
        recs.append(dict(symbol=r.symbol, engine=r.engine, side=r.side, h=r.horizon,
                         size=r.size_usd, gross=r.pnl_pct, state="closed"))
    for r in opens.itertuples(index=False):
        if (r.symbol, r.model, r.entry_price) in closed_keys:
            continue
        c = store.load(r.symbol)
        mark = np.nan
        if c is not None and not c.empty:
            last = float(c.sort_index()["close"].iloc[-1])
            sgn = 1.0 if r.side == "long" else -1.0
            mark = sgn * (last / r.entry_price - 1.0) * 100
        recs.append(dict(symbol=r.symbol, engine=r.engine, side=r.side, h=r.horizon,
                         size=r.size_usd, gross=mark, state="open->marked"))
    d = pd.DataFrame(recs)
    d["net"] = d["gross"] - FEE
    d["usd"] = d["net"] / 100.0 * d["size"]

    print("=== FULL P&L (closed + hand-closed opens, fee netted) ===")
    for state, g in d.groupby("state"):
        print(f"  {state:<14} n={len(g):<3} win={ (g.net>0).mean():.3f}  net%={g.net.sum():+.3f}  net$={g.usd.sum():+.2f}")
    print(f"  {'ALL':<14} n={len(d):<3} win={(d.net>0).mean():.3f}  net%={d.net.sum():+.3f}  net$={d.usd.sum():+.2f}")

    print("\n=== by horizon (where the money went) ===")
    print(f"{'h':>4}{'n':>5}{'win':>7}{'gross%':>9}{'net$':>9}")
    for h in ("1m", "2m", "4m", "8m", "12m", "20m"):
        g = d[d.h == h]
        if len(g):
            print(f"{h:>4}{len(g):>5}{(g.net>0).mean():>7.3f}{g.gross.sum():>+9.3f}{g.usd.sum():>+9.2f}")

    # --- inversion audit: did 'long' align with the realized move? ---
    print("\n=== direction audit (gross sign = did our side win before fees) ===")
    longs = d[d.side == "long"]; shorts = d[d.side == "short"]
    print(f"  longs : n={len(longs)} moved-our-way={ (longs.gross>0).mean():.3f}  mean_gross={longs.gross.mean():+.3f}%")
    if len(shorts):
        print(f"  shorts: n={len(shorts)} moved-our-way={(shorts.gross>0).mean():.3f}  mean_gross={shorts.gross.mean():+.3f}%")
    print("  (if 'moved-our-way' << 0.5 for BOTH sides, suspect inversion; if ~0.5 it's just a hard window)")


if __name__ == "__main__":
    main()
