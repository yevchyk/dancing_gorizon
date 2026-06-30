"""Do the DOWN models have a short edge in the current (crash) regime?

Re-scores the 6 down models over the last N hours and, per model, sweeps the
probability threshold for a SHORT taken at that model's own horizon.
short pnl = -(close_exit/close_entry - 1) - cost.

  python -m src.run_down_window_sweep --hours 12
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .trading.fast_v3_engine import FastV3Engine, V3_DATASET, V3_LABELS
from .trading.timeutil import index_to_ns

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HMIN = {"1m": 1, "2m": 2, "4m": 4, "8m": 8, "12m": 12, "20m": 20}
NS_MIN = 60_000_000_000
COST = FC.EVAL_COST
THRS = (0.70, 0.80, 0.85, 0.90, 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12)
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
    end = pd.Timestamp(last_ns, tz="UTC")
    anchors = pd.date_range(end - pd.Timedelta(hours=args.hours), end, freq=f"{args.cadence}min")
    print(f"window {anchors[0]} -> {anchors[-1]}  ({len(anchors)} anchors, {args.hours}h)  SHORT side\n", flush=True)

    coll = {lab: [] for lab in V3_LABELS}  # (prob, short_pnl)
    for a in anchors:
        a_ns = int(a.value)
        syms, rows = [], []
        for sym, (ts, close) in cache.items():
            f, valid = eng.curve.build_matrix(ts, close, np.array([a_ns], dtype="int64"))
            if bool(valid[0]):
                syms.append(sym); rows.append(f[0])
        if not rows:
            continue
        X = pd.DataFrame(rows, index=syms, columns=eng.columns)
        for lab in V3_LABELS:
            model, cols = eng._models[f"down_{lab}"]
            p = model.predict_proba(X[cols])[:, 1]
            h = HMIN[lab]
            for i, sym in enumerate(syms):
                ts, close = cache[sym]
                ei = int(np.searchsorted(ts, a_ns, side="right")) - 1
                xj = int(np.searchsorted(ts, a_ns + h * NS_MIN, side="right")) - 1
                if ei < 0 or xj <= ei or ts[xj] > last_ns:
                    continue
                short_pnl = -(close[xj] / close[ei] - 1.0) - COST
                coll[lab].append((float(p[i]), short_pnl))

    print(f"{'model':<9}{'p>=':>6}{'n':>7}{'/hr':>6}{'win':>7}{'avg%':>9}{'$tot(30n)':>11}")
    for lab in V3_LABELS:
        arr = coll[lab]
        if not arr:
            continue
        p = np.array([x[0] for x in arr]); pnl = np.array([x[1] for x in arr])
        for thr in THRS:
            m = p >= thr
            n = int(m.sum())
            if n == 0:
                continue
            pk = pnl[m]
            print(f"down_{lab:<4}{thr:>6.2f}{n:>7}{n/args.hours:>6.1f}{(pk>0).mean():>7.3f}"
                  f"{pk.mean()*100:>+9.4f}{pk.sum()*30:>+11.2f}")
        print()


if __name__ == "__main__":
    main()
