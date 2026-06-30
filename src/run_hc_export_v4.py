"""Export a V4 (1-min-horizon) model's scored legs to the browser builder.

Reads the model's own dataset shards (features + realized ret are already there),
scores the up/down ensemble, and writes reports/sim_explorer/data.js as ONE sim
so you can play with filters in the explorer immediately.

  python -m src.run_hc_export_v4 --model-dir models/min1_2to120 \
      --dataset-dir data/hc_min1_2to120/dataset --name "min1 2-120" --hours 240
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .hc import config as HC
from .hc.costs import cost_pct
from .hc.data import load_dataset
from .markets import is_equity
from .run_hc_band_calib import feature_cols, load_folds

OUT = C.ROOT / "reports" / "sim_explorer" / "data.js"
COST = 0.75


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--name", default="min1 2-120")
    ap.add_argument("--hours", type=float, default=240.0)
    ap.add_argument("--floor", type=float, default=0.55)
    args = ap.parse_args()

    feat = feature_cols(args.model_dir)
    folds = load_folds(args.model_dir)
    need = list(dict.fromkeys(feat + ["symbol", "base_time", "horizon_minutes", "ret_pct"]))
    df = load_dataset(args.dataset_dir, columns=need)
    df["base_time"] = pd.to_datetime(df["base_time"], utc=True)
    edge = df["base_time"].max()
    start = edge - pd.Timedelta(hours=args.hours)
    df = df[df["base_time"] >= start].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no rows in window")

    X = df[feat]
    up = np.mean([u.predict_proba(X)[:, 1] for u, _ in folds], axis=0)
    dn = np.mean([d.predict_proba(X)[:, 1] for _, d in folds], axis=0)
    long = up >= dn
    p_dir = np.where(long, up, dn)
    p_opp = np.where(long, dn, up)
    ret = df["ret_pct"].to_numpy()
    signed_ret = np.where(long, ret, -ret)
    side = np.where(long, 1, -1)

    keep = p_dir >= float(args.floor)
    sub = df[keep]
    p_dir, p_opp, signed_ret, side = p_dir[keep], p_opp[keep], signed_ret[keep], side[keep]
    syms = sub["symbol"].to_numpy()
    # Fix 2: per-instrument round-trip cost (was flat COST). Cache one cost/symbol
    # from its recent bar-range so thin/equity names carry their real slippage.
    store = CandleStore(C.CANDLES_DIR)
    cost_cache: dict[str, float] = {}
    def _cost(sym: str) -> float:
        if sym not in cost_cache:
            cost_cache[sym] = cost_pct(sym, candles=store.load(sym))
        return cost_cache[sym]
    net = signed_ret - np.array([_cost(str(s)) for s in syms])
    tmin = (pd.to_datetime(sub["base_time"], utc=True).astype("int64") // 60_000_000_000).to_numpy()
    hz = sub["horizon_minutes"].to_numpy()

    # merge into existing data.js so the old models stay available
    syms_arr, sims = _load_existing()
    idx_of = {s: i for i, s in enumerate(syms_arr)}

    def sym_idx(s: str) -> int:
        if s not in idx_of:
            idx_of[s] = len(syms_arr)
            syms_arr.append(s)
        return idx_of[s]

    legs = []
    for i in range(len(sub)):
        s = str(syms[i])
        legs.append([sym_idx(s), int(tmin[i]), int(hz[i]), int(side[i]),
                     round(float(p_dir[i]), 4), round(float(p_opp[i]), 4),
                     round(float(net[i]), 3), 1 if is_equity(s) else 0])
    sims[args.name] = {"legs": legs}

    horizons = sorted(int(x) for x in np.unique(hz))
    hours = max(1.0, (edge - start).total_seconds() / 3600.0)
    meta = {"cost_pct": COST, "cost_model": "per-instrument (hc.costs)",
            "window_start": str(start), "window_end": str(edge),
            "hours": hours, "horizons": horizons, "floor": args.floor, "tz": "Europe/Kiev"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("window.SIM_META=" + json.dumps(meta) + ";\n")
        f.write("window.SYMS=" + json.dumps(syms_arr) + ";\n")
        f.write("window.SIMS=" + json.dumps(sims, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT}  sims={list(sims)} new='{args.name}' legs={len(legs)} "
          f"syms={len(syms_arr)} horizons={len(horizons)} window={hours:.0f}h size={OUT.stat().st_size//1024}KB")
    try:
        from .run_hc_export_meta import main as _inject_meta
        _inject_meta()  # (re)add window.LIQUID + window.COST so the explorer keeps them
    except Exception as e:
        print(f"meta inject skipped: {e}")


def _load_existing() -> tuple[list, dict]:
    """Parse existing data.js -> (SYMS list, SIMS dict). Empty if missing/bad."""
    if not OUT.exists():
        return [], {}
    try:
        txt = OUT.read_text(encoding="utf-8")
        syms, sims = [], {}
        for line in txt.splitlines():
            line = line.strip().rstrip(";")
            if line.startswith("window.SYMS="):
                syms = json.loads(line[len("window.SYMS="):])
            elif line.startswith("window.SIMS="):
                sims = json.loads(line[len("window.SIMS="):])
        return list(syms), dict(sims)
    except Exception:
        return [], {}


if __name__ == "__main__":
    main()
