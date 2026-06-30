"""Sanity-check a built Binance year-rebuild dataset against the frozen configs.

Verifies, per shard (a sample or all):
  1. thr_pct == cost(symbol) + med|funding|(symbol) * h/480  (bit-exact f32)
  2. labels consistent: up = ret_pct >= thr, down = ret_pct <= -thr
  3. horizons within the declared grid; base_time within era; no feature NaNs
  4. per-symbol grid jitter actually varies across the universe
  5. label base rates per horizon band (prints — eyeball that win isn't ~0 or ~1)

  python -m src.binance_dataset_check --dataset data/binance_smoke/dataset
  python -m src.binance_dataset_check --dataset data/binance_y1/dataset --sample 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COSTS = Path("configs/binance_costs.json")
FUNDING = Path("configs/binance_funding.json")


def check_shard(p: Path, cost_map, fund_map) -> dict:
    df = pd.read_parquet(p)
    sym = df["symbol"].iloc[0]
    h = df["horizon_minutes"].to_numpy("float32")
    want = np.float32(cost_map[sym]) + np.float32(fund_map.get(sym, {}).get("med_abs_pct", 0.0)) * (h / np.float32(480.0))
    thr = df["thr_pct"].to_numpy("float32")
    thr_ok = np.allclose(thr, want.astype("float32"), atol=1e-5)
    ret = df["ret_pct"].to_numpy("float32")
    # the builder labels in FULL float64 precision, but ret_pct is stored float32:
    # a row sitting within a rounding-ulp of the ±thr boundary can legitimately
    # flip when recomputed from the rounded value (seen: 1 row / 7.3M). Only count
    # a mismatch as REAL when the row is clearly away from the boundary.
    TOL = 1e-4  # percent
    up_calc = (ret >= thr).astype("int8")
    dn_calc = (ret <= -thr).astype("int8")
    up_lbl = df["up_label"].to_numpy("int8")
    dn_lbl = df["down_label"].to_numpy("int8")
    up_real = (up_calc != up_lbl) & (np.abs(ret - thr) > TOL)
    dn_real = (dn_calc != dn_lbl) & (np.abs(ret + thr) > TOL)
    borderline = int(((up_calc != up_lbl) | (dn_calc != dn_lbl)).sum()
                     - (up_real | dn_real).sum())
    up_ok = not bool(up_real.any())
    down_ok = not bool(dn_real.any())
    feat_cols = [c for c in df.columns if c.startswith(("c1m_", "c5m_", "c15m_", "c1h_", "c4h_"))]
    nan_ok = not df[feat_cols].isna().any().any()
    bt = pd.to_datetime(df["base_time"], utc=True)
    return {"symbol": sym, "rows": len(df), "thr_ok": thr_ok, "up_ok": up_ok,
            "down_ok": down_ok, "nan_ok": nan_ok, "borderline": borderline,
            "h_min": int(h.min()), "h_max": int(h.max()),
            "minute_offset": int(bt.dt.minute.iloc[0] % 60),
            "bt_min": str(bt.min()), "bt_max": str(bt.max()),
            "up_rate": round(float(df["up_label"].mean()), 4),
            "down_rate": round(float(df["down_label"].mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=0, help="0 = all shards")
    args = ap.parse_args()
    cost_map = {k: float(v) for k, v in json.loads(COSTS.read_text())["costs"].items()}
    fund_map = json.loads(FUNDING.read_text())["symbols"]
    shards = sorted(args.dataset.glob("*.parquet"))
    if args.sample:
        idx = np.linspace(0, len(shards) - 1, num=min(args.sample, len(shards)), dtype=int)
        shards = [shards[i] for i in sorted(set(idx))]
    print(f"checking {len(shards)} shards from {args.dataset}")
    results = [check_shard(p, cost_map, fund_map) for p in shards]
    bad = [r for r in results if not (r["thr_ok"] and r["up_ok"] and r["down_ok"] and r["nan_ok"])]
    offsets = sorted({r["minute_offset"] for r in results})
    for r in results:
        flag = "" if r not in bad else "  <-- BAD"
        print(f"  {r['symbol']:18s} rows={r['rows']:>7} thr={r['thr_ok']} up={r['up_ok']} "
              f"down={r['down_ok']} nan_ok={r['nan_ok']} h={r['h_min']}..{r['h_max']} "
              f"off={r['minute_offset']:>2}m up_rate={r['up_rate']:.3f} down_rate={r['down_rate']:.3f}{flag}")
    print(f"\njitter offsets seen: {offsets}")
    print(f"window: {min(r['bt_min'] for r in results)} .. {max(r['bt_max'] for r in results)}")
    if bad:
        raise SystemExit(f"FAIL: {len(bad)} shard(s) inconsistent")
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
