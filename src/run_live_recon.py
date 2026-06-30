"""Reconcile live trades against a fresh re-inference: did the live engine open
real signals, and is the logged PnL true?

For every live OPEN we rebuild the 320-col curve + score the 8 fast_v2 models at
that open's scan anchor (= open_ts floored to 2 min) and recompute the worthy
agreement. The open is "confirmed" if the agreement and side match what the
engine claims (pulse00 => >=3 agree; pulse => >=2 agree, clean, same side).
For every CLOSE we recompute the realized 10m return from candles and compare it
to the logged pnl_pct.

Run: python -m src.run_live_recon <run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .trading.fast_combo_engine import FastComboEngine, WORTHY
from .trading.timeutil import index_to_ns

EXIT_MIN = {"2m": 2, "5m": 5, "8m": 8, "10m": 10}


def agreement_at(eng, store, sym, anchor):
    """(up_count, down_count) of worthy models at one anchor for one symbol."""
    c = store.load(sym)
    if c is None or c.empty:
        return None
    c = c.sort_index()
    feats, valid = eng.curve.build_matrix(
        index_to_ns(c.index), c["close"].to_numpy("float64"),
        np.array([int(pd.Timestamp(anchor).value)], dtype="int64"))
    if not bool(valid[0]):
        return None
    X = pd.DataFrame(feats[[0]], columns=eng.columns)
    up = dn = 0
    for full, (mname, _sn, side, base) in WORTHY.items():
        model, cols = eng._models[mname]
        p = float(model.predict_proba(X[cols])[:, 1])
        if p >= base:
            if side == 1:
                up += 1
            else:
                dn += 1
    return up, dn


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(
        (FC.FAST_ANALYSIS_DIR.parent.parent / "trading_logs").glob("live_*"),
        key=lambda p: p.stat().st_mtime)
    t = pd.read_csv(run / "trades.csv")
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    op = t[t["event"] == "open"].copy()
    cl = t[t["event"].str.contains("close")].copy()
    eng = FastComboEngine("pulse00")
    store = CandleStore(C.CANDLES_DIR)

    print(f"run={run.name}  opens={len(op)}  closes={len(cl)}")

    # ---- Check 1: were the opened signals real? ----
    rows = []
    for r in op.itertuples(index=False):
        anchor = r.ts.floor("2min")
        ag = agreement_at(eng, store, r.symbol, anchor)
        if ag is None:
            rows.append({"engine": r.engine, "ok": None}); continue
        up, dn = ag
        want = 3 if r.engine == "pulse00" else 2
        side_cnt = up if r.side == "long" else dn
        opp = dn if r.side == "long" else up
        ok = side_cnt >= want and opp == 0
        rows.append({"engine": r.engine, "symbol": r.symbol, "side": r.side,
                     "anchor": anchor, "up": up, "down": dn, "need": want, "ok": ok})
    rec = pd.DataFrame(rows)
    print("\n=== CHECK 1: opened signals confirmed by re-inference ===")
    for engn, g in rec.groupby("engine"):
        gg = g[g["ok"].notna()]
        print(f"  {engn:8s} confirmed {int(gg['ok'].sum())}/{len(gg)} "
              f"({gg['ok'].mean()*100:.0f}%)")
    bad = rec[(rec["ok"] == False)]
    if len(bad):
        print("  unconfirmed (engine said signal, re-inference disagrees):")
        print(bad[["engine", "symbol", "side", "up", "down", "need"]].head(10).to_string(index=False))
    else:
        print("  every confirmable open matched the live agreement -> execution is faithful")

    # ---- Check 2: is the logged PnL true? (recompute 10m return from candles) ----
    print("\n=== CHECK 2: logged PnL vs candle-recomputed 10m return ===")
    cl2 = cl.dropna(subset=["entry_price"]).copy()
    cl2["entry_price"] = pd.to_numeric(cl2["entry_price"], errors="coerce")
    cl2["pnl_pct"] = pd.to_numeric(cl2["pnl_pct"], errors="coerce")
    diffs = []
    for r in cl2.itertuples(index=False):
        c = store.load(r.symbol)
        if c is None or c.empty or not np.isfinite(r.entry_price) or r.entry_price <= 0:
            continue
        c = c.sort_index()
        # the close ts is ~deadline+confirm; recompute price at the 10m deadline
        # measured from the open anchor. Approx anchor = close_ts - 10m, floored.
        dl = r.ts.floor("2min")
        px = c["close"].reindex(c.index[c.index <= dl]).iloc[-1] if (c.index <= dl).any() else None
        if px is None:
            continue
        sgn = 1 if r.side == "long" else -1
        recomputed = sgn * (float(px) / r.entry_price - 1.0) * 100
        if np.isfinite(r.pnl_pct):
            diffs.append(abs(recomputed - r.pnl_pct))
    if diffs:
        d = np.array(diffs)
        print(f"  {len(d)} closes checked; |logged - recomputed| median={np.median(d):.3f}pp "
              f"mean={d.mean():.3f}pp max={d.max():.3f}pp")
        print("  (small diffs = OCO/timing; large = mismatch)")

    # ---- Live realized summary ----
    print("\n=== LIVE realized (from log) ===")
    clp = cl.copy(); clp["pnl_pct"] = pd.to_numeric(clp["pnl_pct"], errors="coerce")
    clp["size_usd"] = pd.to_numeric(clp["size_usd"], errors="coerce").fillna(0)
    clp["usd"] = clp["size_usd"] * clp["pnl_pct"] / 100
    for engn, g in clp.groupby("engine"):
        gg = g.dropna(subset=["pnl_pct"])
        print(f"  {engn:8s} n={len(gg):2d} win={(gg.pnl_pct>0).mean():.3f} "
              f"sumPnl%={gg.pnl_pct.sum():+.2f} $={g.usd.sum():+.2f}")
    print(f"  TOTAL $ realized = {clp['usd'].sum():+.3f}  (open still: {len(op)-len(cl)})")


if __name__ == "__main__":
    main()
