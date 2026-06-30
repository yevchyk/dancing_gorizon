"""Build a custom crisis dataset from specific date windows.

Takes anchors ONLY from:
  - bad days (05-27, 06-02 morning crash)
  - good days for balance (05-30, 05-21)
Plus a fresh holdout = last N hours.

Run: python -m src.run_build_crisis_custom
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .fast.candles import load_1m
from .fast.curve import FastCurve
from .fast.dataset import _targets, _to_ns
from .trading.timeutil import index_to_ns


EXPERIMENT = os.environ.get("ML_FAST_EXPERIMENT", "latest_crisis")
BTC_ON = os.environ.get("ML_FAST_BTC", "0") == "1"

# Crisis (bad) training windows + balance (good) windows
TRAIN_WINDOWS = [
    # bad days — crisis/chop
    ("bad",  "2026-05-27 00:00", "2026-05-28 00:00"),
    ("bad",  "2026-06-02 00:00", "2026-06-02 16:50"),
    # good days — balance so model doesn't go all-short
    ("good", "2026-05-30 00:00", "2026-05-31 00:00"),
    ("good", "2026-05-21 00:00", "2026-05-22 00:00"),
]
HOLDOUT_HOURS = 3.0
STEP_MIN = 2
TRAIN_ANCHORS_PER_SYM = 2000  # random per window per symbol


def build() -> None:
    out_dir = C.DATA_DIR / EXPERIMENT / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = CandleStore(C.CANDLES_DIR)
    curve = FastCurve(FC.CURVE_POINTS, FC.CURVE_MIN_STEP_MIN,
                      FC.CURVE_MAX_DEPTH_MIN, FC.CURVE_SEGMENTS)

    # BTC context
    btc = None
    btc_curve = None
    btc_cols: list[str] = []
    if BTC_ON:
        btc_raw = store.load(FC.BTC_SYMBOL)
        if btc_raw is not None:
            btc_raw = btc_raw.sort_index()
            btc_curve = FastCurve(0, FC.CURVE_MIN_STEP_MIN, FC.CURVE_MAX_DEPTH_MIN,
                                  offsets_min=FC.BTC_OFFSETS_MIN)
            btc = (_to_ns(btc_raw.index), btc_raw["close"].to_numpy("float64"), btc_curve)
            btc_cols = FC.btc_columns()
            print(f"BTC ON: {len(btc_cols)} cols")

    now = pd.Timestamp.now(tz="UTC").floor("1min")
    hold_start = now - pd.Timedelta(hours=HOLDOUT_HOURS)
    hold_end = now - pd.Timedelta(minutes=FC.HORIZONS[-1].minutes + 2)

    import json
    syms_path = C.DATA_DIR / "fast_v2" / "datasets" / "symbols.json"
    syms = json.loads(syms_path.read_text())[:120]
    syms = [s for s in syms if s not in set(C.BLACKLIST_SYMBOLS)]
    print(f"symbols={len(syms)}  hold={hold_start:%m-%d %H:%M}->{hold_end:%m-%d %H:%M} UTC")

    rng = np.random.default_rng(42)
    chunks: list[pd.DataFrame] = []
    kept = 0

    for i, sym in enumerate(syms, 1):
        tgt = load_1m(sym)
        feat = store.load(sym)
        if tgt is None or tgt.empty or feat is None or feat.empty:
            continue
        tgt = tgt.sort_index(); feat = feat.sort_index()
        tgt_ns = _to_ns(tgt.index)
        feat_ns = index_to_ns(feat.index)
        feat_cl = feat["close"].to_numpy("float64")
        hi = tgt["high"].to_numpy("float64")
        lo = tgt["low"].to_numpy("float64")
        cl = tgt["close"].to_numpy("float64")

        all_anchors: list[pd.Timestamp] = []
        splits: list[str] = []

        # --- train windows (random sample per window) ---
        for tag, ws, we in TRAIN_WINDOWS:
            w_s = pd.Timestamp(ws, tz="UTC"); w_e = pd.Timestamp(we, tz="UTC")
            idx = tgt.index[(tgt.index >= w_s) & (tgt.index < w_e)]
            if len(idx) == 0:
                continue
            n = min(TRAIN_ANCHORS_PER_SYM, len(idx))
            picks = np.sort(rng.choice(len(idx), size=n, replace=False))
            all_anchors.extend(idx[picks])
            splits.extend(["train"] * n)

        # --- holdout (dense 2m grid) ---
        hold_grid = pd.date_range(hold_start.ceil(f"{STEP_MIN}min"),
                                  hold_end.floor(f"{STEP_MIN}min"),
                                  freq=f"{STEP_MIN}min")
        hold_grid = hold_grid[hold_grid <= tgt.index.max()]
        all_anchors.extend(hold_grid)
        splits.extend(["holdout"] * len(hold_grid))

        if not all_anchors:
            continue

        anchors = pd.DatetimeIndex(all_anchors)
        split_arr = np.array(splits, dtype=object)
        ans_ns = anchors.as_unit("ns").asi8

        feats, fv = curve.build_matrix(feat_ns, feat_cl, ans_ns)
        tgt_d = _targets(tgt_ns, hi, lo, cl, ans_ns)
        valid = fv.copy()
        for v in tgt_d.values():
            valid &= np.isfinite(v)

        btc_feats = None
        if btc is not None and btc_curve is not None:
            btc_ns, btc_cl, bc = btc
            btc_feats, bv = bc.build_matrix(btc_ns, btc_cl, ans_ns)
            valid &= bv

        if valid.sum() == 0:
            continue

        data = {
            "symbol": np.array([sym] * len(anchors), dtype=object)[valid],
            "anchor_time": anchors[valid],
            "split": split_arr[valid],
        }
        for j, col in enumerate(curve.columns()):
            data[col] = feats[valid, j]
        for col, vals in tgt_d.items():
            data[col] = vals[valid]
        if btc_feats is not None:
            for j, col in enumerate(btc_cols):
                data[col] = btc_feats[valid, j]

        chunks.append(pd.DataFrame(data))
        kept += 1
        if i % 20 == 0 or i == len(syms):
            print(f"  {i}/{len(syms)} kept={kept}", flush=True)

    master = pd.concat(chunks, ignore_index=True)
    by_split = master.groupby("split")["anchor_time"].agg(["min", "max", "count"])
    print(f"\ndataset: {len(master)} rows, {len(master.columns)} cols")
    print(by_split.to_string())
    out = out_dir / "master.parquet"
    master.to_parquet(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    build()
