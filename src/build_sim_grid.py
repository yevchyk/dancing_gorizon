"""Build a 5-minute signal grid for the last N days on the top-liquid coins, so
the event-driven simulator can replay 'scan every 5 min' realistically.

For each coin at each 5-min timestamp: curve features -> production model scores
(p_up/p_down/pred_mfe/pred_mae per horizon) + the realized EXIT price at
anchor+horizon for each horizon (for fixed-horizon-close PnL). Saved to
data/datasets/sim_grid.parquet.

Usage:
  python -m src.build_sim_grid --days 10 --coins 40 --step 5
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .features import CurveBuilder
from .training.horizon_slicer import HorizonSlicer
from .trading.timeutil import index_to_ns, anchors_to_ns, NS_PER_MIN

REG = C.MODELS_DIR / "reg"
DIRP = C.MODELS_DIR / "dir_prob"


def top_liquid(store: CandleStore, n: int) -> list[str]:
    bl = set(C.BLACKLIST_SYMBOLS)
    vols = []
    for s in store.symbols():
        if s in bl:
            continue
        c = store.load(s)
        if c is None or c.empty:
            continue
        tail = c.iloc[-1440:]
        vols.append((s, float((tail["close"] * tail["volume"]).sum())))
    vols.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in vols[:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--coins", type=int, default=40)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()

    store = CandleStore(C.CANDLES_DIR)
    curve = CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    slicer = HorizonSlicer(curve)
    models = {}
    for h in C.HORIZONS:
        models[h.label] = {
            "cols": slicer.columns_for(h),
            "up": joblib.load(DIRP / f"up_{h.label}.joblib"),
            "down": joblib.load(DIRP / f"down_{h.label}.joblib"),
            "mfe": joblib.load(REG / f"mfe_{h.label}.joblib"),
            "mae": joblib.load(REG / f"mae_{h.label}.joblib"),
        }
    syms = top_liquid(store, args.coins)
    now = pd.Timestamp.now(tz="UTC").floor(f"{args.step}min")
    start = now - pd.Timedelta(days=args.days)
    end = now - pd.Timedelta(minutes=C.HORIZONS[-1].minutes)   # need lookahead for 2h exit
    grid = pd.date_range(start, end, freq=f"{args.step}min")
    print(f"coins={len(syms)} grid={len(grid)} steps ({start.date()}..{end.date()})")

    out = []
    for ci, sym in enumerate(syms, 1):
        candles = store.load(sym)
        if candles is None or candles.empty:
            continue
        ts = index_to_ns(candles.index)
        close = candles["close"].to_numpy(float)
        rows = []
        for a in grid:
            cv = curve.build(candles, a)
            if cv is None:
                continue
            ei = int(np.searchsorted(ts, a.value, side="right")) - 1
            if ei < 0:
                continue
            rows.append({"symbol": sym, "time": a, "entry_price": close[ei], **cv})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        anc = anchors_to_ns(df["time"])   # nanoseconds (match ts)
        # exit price at anchor+H per horizon
        for h in C.HORIZONS:
            ex_ns = anc + h.minutes * NS_PER_MIN
            idx = np.searchsorted(ts, ex_ns, side="right") - 1
            df[f"exit_{h.label}"] = np.where(idx >= 0, close[np.clip(idx, 0, len(close) - 1)], np.nan)
            X = df[models[h.label]["cols"]]
            df[f"p_up_{h.label}"] = models[h.label]["up"].predict_proba(X)[:, 1]
            df[f"p_down_{h.label}"] = models[h.label]["down"].predict_proba(X)[:, 1]
            df[f"mfe_{h.label}"] = models[h.label]["mfe"].predict(X)
            df[f"mae_{h.label}"] = models[h.label]["mae"].predict(X)
        keep = ["symbol", "time", "entry_price"] + [
            f"{p}_{h.label}" for h in C.HORIZONS
            for p in ("p_up", "p_down", "mfe", "mae", "exit")]
        out.append(df[keep])
        if ci % 5 == 0 or ci == len(syms):
            print(f"  {ci}/{len(syms)} coins", flush=True)

    grid_df = pd.concat(out, ignore_index=True)
    path = C.DATASETS_DIR / "sim_grid.parquet"
    grid_df.to_parquet(path, index=False)
    print(f"sim grid: {len(grid_df)} rows -> {path}")


if __name__ == "__main__":
    main()
