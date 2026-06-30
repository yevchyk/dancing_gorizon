"""Produce HONEST out-of-sample scores for days AFTER the training window (>06-01),
using the final fast_v3 models + the live candle store. Output matches the
holdout_scores_wf schema so run_engines_v2_sim can consume it.

  python -m src.run_score_future --days 2026-06-02,2026-06-03 --tag _future
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .trading.fast_v3_engine import FastV3Engine, V3_DATASET, V3_LABELS
from .trading.timeutil import index_to_ns
from .run_fast_v3 import V3_ANALYSIS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HMIN = {"1m": 1, "2m": 2, "4m": 4, "8m": 8, "12m": 12, "20m": 20}
NS_MIN = 60_000_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True, help="comma-sep, e.g. 2026-06-02,2026-06-03")
    ap.add_argument("--tag", default="_future")
    ap.add_argument("--cadence", type=int, default=2)
    args = ap.parse_args()

    eng = FastV3Engine("verkh_v2")  # loads all 12 up/down models + curve
    store = CandleStore(C.CANDLES_DIR)
    watch = list(pd.read_parquet(V3_DATASET, columns=["symbol"])["symbol"].unique())
    cache = {}
    for sym in watch:
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        cache[sym] = (index_to_ns(c.index), c["close"].to_numpy("float64"),
                      c["high"].to_numpy("float64"), c["low"].to_numpy("float64"))
    last_ns = max(v[0][-1] for v in cache.values())

    out_rows = []
    for day in args.days.split(","):
        day = day.strip()
        d0 = pd.Timestamp(day, tz="UTC")
        anchors = pd.date_range(d0, d0 + pd.Timedelta(hours=24), freq=f"{args.cadence}min")
        anchors = [a for a in anchors if int(a.value) + 20 * NS_MIN <= last_ns]
        print(f"[{day}] {len(anchors)} anchors", flush=True)
        for a in anchors:
            a_ns = int(a.value)
            syms, rows = [], []
            for sym, (ts, close, high, low) in cache.items():
                f, valid = eng.curve.build_matrix(ts, close, np.array([a_ns], dtype="int64"))
                if bool(valid[0]):
                    syms.append(sym); rows.append(f[0])
            if not rows:
                continue
            X = pd.DataFrame(rows, index=syms, columns=eng.columns)
            probs = {}
            for lab in V3_LABELS:
                for side in ("up", "down"):
                    model, cols = eng._models[f"{side}_{lab}"]
                    probs[f"p_{side}_{lab}"] = model.predict_proba(X[cols])[:, 1]
            # realized ret/mfe/mae per horizon, from candles (dataset _targets convention)
            real = {f"{k}_{lab}": np.full(len(syms), np.nan) for lab in V3_LABELS for k in ("ret", "mfe", "mae")}
            for j, sym in enumerate(syms):
                ts, close, high, low = cache[sym]
                ei = int(np.searchsorted(ts, a_ns, side="right")) - 1
                if ei < 0:
                    continue
                entry = close[ei]
                for lab in V3_LABELS:
                    fj = int(np.searchsorted(ts, a_ns + HMIN[lab] * NS_MIN, side="right"))
                    if fj <= ei + 1:
                        continue
                    real[f"ret_{lab}"][j] = close[fj - 1] / entry - 1.0
                    real[f"mfe_{lab}"][j] = high[ei + 1:fj].max() / entry - 1.0
                    real[f"mae_{lab}"][j] = low[ei + 1:fj].min() / entry - 1.0
            rec = {"symbol": syms, "anchor_time": [a] * len(syms), "day": [day] * len(syms)}
            for k, v in probs.items():
                rec[k] = v
            for k, v in real.items():
                rec[f"real_{k}"] = v
            out_rows.append(pd.DataFrame(rec))

    df = pd.concat(out_rows, ignore_index=True)
    # keep only rows with all targets realized
    tcols = [f"real_ret_{lab}" for lab in V3_LABELS]
    df = df.dropna(subset=tcols)
    out = V3_ANALYSIS / f"holdout_scores_wf{args.tag}.parquet"
    df.to_parquet(out, index=False)
    print(f"\nout-of-sample scores: {len(df)} rows, days={df['day'].nunique()} -> {out}")


if __name__ == "__main__":
    main()
