"""Replay the verkh flat-pool over a past time window using the live models +
candle store, for OLD vs NEW thresholds. Answers: would the new config have bled
on the same 40 minutes the live engine bled?

Pure model-signal sim (fills at candle close, fixed cost) -- no live slippage/caps.

  python -m src.run_verkh_window_sim --start "2026-06-03 10:12" --minutes 50
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .trading.fast_v3_engine import FastV3Engine, V3_DATASET, V3_LABELS
from .trading.timeutil import index_to_ns

HMIN = {"1m": 1, "2m": 2, "4m": 4, "8m": 8, "12m": 12, "20m": 20}
NS_MIN = 60_000_000_000
COST = FC.EVAL_COST

OLD = {"1m": 0.95, "2m": 0.93, "4m": 0.92, "8m": 0.90, "12m": 0.88, "20m": 0.85}
NEW = {"2m": 0.95, "4m": 0.95, "8m": 0.92, "12m": 0.90, "20m": 0.90}
# keep up_20m at its sweet spot 0.85 (the breadwinner), prune weak short legs only
CORRECTED = {"4m": 0.95, "8m": 0.88, "12m": 0.88, "20m": 0.85}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-03 10:12")
    ap.add_argument("--minutes", type=int, default=50)
    ap.add_argument("--cadence", type=int, default=2)
    args = ap.parse_args()

    eng = FastV3Engine("verkh_v2")
    store = CandleStore(C.CANDLES_DIR)
    watch = list(pd.read_parquet(V3_DATASET, columns=["symbol"])["symbol"].unique())

    cache = {}
    for sym in watch:
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        cache[sym] = (index_to_ns(c.index), c["close"].to_numpy("float64"))
    last_ns = max(ts[-1] for ts, _ in cache.values())

    start = pd.Timestamp(args.start, tz="UTC")
    anchors = pd.date_range(start, start + pd.Timedelta(minutes=args.minutes),
                            freq=f"{args.cadence}min")
    print(f"window {anchors[0]} -> {anchors[-1]}  ({len(anchors)} anchors)  symbols={len(cache)}\n")

    # collect signals per up-model: (prob, pnl_realized) keyed by label
    sig = {lab: [] for lab in V3_LABELS}
    for a in anchors:
        a_ns = int(a.value)
        syms, rows = [], []
        for sym, (ts, close) in cache.items():
            f, valid = eng.curve.build_matrix(ts, close, np.array([a_ns], dtype="int64"))
            if not bool(valid[0]):
                continue
            syms.append(sym)
            rows.append(f[0])
        if not rows:
            continue
        X = pd.DataFrame(rows, index=syms, columns=eng.columns)
        for lab in V3_LABELS:
            model, cols = eng._models[f"up_{lab}"]
            p = model.predict_proba(X[cols])[:, 1]
            h = HMIN[lab]
            for i, sym in enumerate(syms):
                ts, close = cache[sym]
                ei = int(np.searchsorted(ts, a_ns, side="right")) - 1
                xj = int(np.searchsorted(ts, a_ns + h * NS_MIN, side="right")) - 1
                if ei < 0 or xj <= ei or ts[xj] > last_ns:
                    continue
                pnl = close[xj] / close[ei] - 1.0 - COST
                sig[lab].append((float(p[i]), float(pnl)))

    def report(name, thr):
        print(f"--- verkh {name} ($30 notional) ---")
        print(f"    {'model':<8}{'p>=':>6}{'n':>6}{'win':>7}{'avg%':>9}{'$tot':>9}")
        tot_usd = 0.0
        tot_n = wins = 0
        for lab in V3_LABELS:
            if lab not in thr:
                continue
            fired = [(p, pnl) for p, pnl in sig[lab] if p >= thr[lab]]
            if not fired:
                print(f"    up_{lab:<5}{thr[lab]:>6.2f}{0:>6}")
                continue
            pnls = np.array([pnl for _p, pnl in fired])
            usd = pnls.sum() * 30.0
            tot_usd += usd; tot_n += len(pnls); wins += int((pnls > 0).sum())
            print(f"    up_{lab:<5}{thr[lab]:>6.2f}{len(pnls):>6}{(pnls>0).mean():>7.3f}"
                  f"{pnls.mean()*100:>+9.4f}{usd:>+9.2f}")
        win = wins / tot_n if tot_n else 0.0
        print(f"    {'TOTAL':<8}{'':>6}{tot_n:>6}{win:>7.3f}{'':>9}{tot_usd:>+9.2f}\n")

    report("OLD (live config)", OLD)
    report("NEW (raised 20m -> 0.90)", NEW)
    report("CORRECTED (keep 20m 0.85, drop 1m/2m)", CORRECTED)


if __name__ == "__main__":
    main()
