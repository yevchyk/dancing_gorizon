"""Build the Binance year-rebuild dataset (BINANCE_PLAN.md §5).

v4 pipeline (1-min targets, c1m+5m/15m/1h/4h curves + time feats) pointed at the
Binance store, with the two honest-label changes vs the OKX builds:
  * per-symbol threshold = rt_cost_pct (configs/binance_costs.json)
                           + med|funding| * h/480 (configs/binance_funding.json)
    — NOT the flat THRESHOLD_GRID_PCT curve;
  * per-symbol grid jitter so a 60-min stride doesn't pin every snapshot
    universe-wide to the same minute-of-hour.

Horizons: full 30..480 by 5 (91 values); every snapshot gets anchors (30,120,480)
plus --random-count drawn from the rest, so the whole grid is covered while rows
stay bounded (~9-10M for ~180 syms x 365d x stride 60 x 6 horizons; 32GB box).

Shards are per-symbol parquets — resume-able (existing shards are skipped), so
the build can be killed/restarted freely.

  python -m src.run_binance_dataset                       # full build
  python -m src.run_binance_dataset --limit-symbols 2 --days 3 \
      --out-dir data/binance_smoke/dataset                # smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .hc import config as HC
from .hc import schema_v3 as S3
from .markets import REGISTRY, Store
from . import config as C

BINANCE_CANDLES = Path("data/binance/candles")
BINANCE_ERA_START = pd.Timestamp("2025-06-01T00:00:00Z")

# --- point the HC pipeline at the Binance store BEFORE importing the builders.
# STORE_KEY is read at call time by _load_raw(); HC_ERA_START clamps _fine_start
# (the OKX value 2025-09-27 would silently cut 3.5 months of the Binance year).
REGISTRY["binance_feature"] = Store(
    "binance_feature", "crypto", "feature", "1m",
    C.ROOT / BINANCE_CANDLES, C.ROOT / "configs" / "binance_train_universe.json",
    "Binance USDT-M 365d 1m store for the year-rebuild (BINANCE_PLAN.md).")
HC.STORE_KEY = "binance_feature"
HC.HC_ERA_START = BINANCE_ERA_START

from .hc.data import SymbolBuildStats, prepare_btc_frames, stable_seed  # noqa: E402
from .hc.data_v4 import build_symbol_frame_v4  # noqa: E402

UNIVERSE = Path("configs/binance_train_universe.json")
COSTS = Path("configs/binance_costs.json")
FUNDING = Path("configs/binance_funding.json")
ANCHORS = (30, 120, 480)

# --- multiprocessing workers (feature build is a pure-python per-anchor loop ->
# processes, not threads). Heavy state is built lazily ONCE per worker process.
_W_BTC = None
_W_THR = None


def _worker_build(task: tuple) -> dict:
    global _W_BTC, _W_THR
    (sym, out_dir, anchors, candidates, random_count, stride_min, days, entry_delay) = task
    if _W_BTC is None:
        _W_BTC = prepare_btc_frames()
    if _W_THR is None:
        cost_map = {k: float(v) for k, v in json.loads(COSTS.read_text())["costs"].items()}
        fund_map = json.loads(FUNDING.read_text())["symbols"]
        _W_THR = make_threshold_fn(cost_map, fund_map)
    offset = (stable_seed(sym, 7) % (stride_min // 5)) * 5 if stride_min >= 10 else 0
    df, stat = build_symbol_frame_v4(
        sym, btc_frames=_W_BTC, anchors=anchors, candidates=candidates,
        random_count=random_count, stride_min=stride_min, days=days,
        entry_delay_min=entry_delay, threshold_fn=_W_THR, grid_offset_min=offset)
    if len(df):
        df.to_parquet(Path(out_dir) / f"{sym}.parquet", index=False)
    return stat.__dict__


def make_threshold_fn(cost_map: dict[str, float], fund_map: dict[str, dict]):
    """thr%(symbol, h) = round-trip cost + expected adverse funding over the hold.
    Missing symbol = hard error: the frozen universe must be fully covered."""
    def fn(symbol: str, h: np.ndarray) -> np.ndarray:
        cost = cost_map[symbol]
        fund = fund_map.get(symbol, {}).get("med_abs_pct", 0.0)
        return (cost + fund * (h.astype("float32") / 480.0)).astype("float32")
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data/binance_y1/dataset"))
    ap.add_argument("--universe", type=Path, default=UNIVERSE)
    ap.add_argument("--stride-min", type=int, default=60)
    ap.add_argument("--days", type=int, default=None, help="default: full series")
    ap.add_argument("--hmin", type=int, default=30)
    ap.add_argument("--hmax", type=int, default=480)
    ap.add_argument("--hstep", type=int, default=5)
    ap.add_argument("--random-count", type=int, default=3,
                    help="random horizons per snapshot on top of anchors (30,120,480)")
    ap.add_argument("--entry-delay-min", type=int, default=HC.EXEC_ENTRY_DELAY_MIN)
    ap.add_argument("--limit-symbols", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel symbol builders (8 ~= 6-8x faster on this box)")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    syms = json.loads(a.universe.read_text())["symbols"]
    if a.limit_symbols:
        syms = syms[:a.limit_symbols]
    cost_map = {k: float(v) for k, v in json.loads(COSTS.read_text())["costs"].items()}
    fund_map = json.loads(FUNDING.read_text())["symbols"]
    missing_cost = [s for s in syms if s not in cost_map]
    if missing_cost:
        raise SystemExit(f"{len(missing_cost)} universe symbols lack a trusted cost "
                         f"(rerun src.binance_costs): {missing_cost[:8]}")
    thr_fn = make_threshold_fn(cost_map, fund_map)
    candidates = tuple(range(a.hmin, a.hmax + 1, a.hstep))
    anchors = tuple(x for x in ANCHORS if a.hmin <= x <= a.hmax) or (candidates[0],)
    n_per_snap = len(anchors) + a.random_count

    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.fresh:
        for p in a.out_dir.glob("*.parquet"):
            p.unlink()
    print(f"binance_y1 dataset -> {a.out_dir}\n"
          f"  syms={len(syms)} stride={a.stride_min}m(+jitter) days={a.days or 'all'} "
          f"horizons={a.hmin}..{a.hmax}/{a.hstep} ({len(candidates)} grid, {n_per_snap}/snap) "
          f"feat_cols={len(S3.FEATURE_COLUMNS_V3)}\n"
          f"  labels: per-symbol cost+funding (e.g. BTC thr@480m="
          f"{thr_fn('BTC_USDT_SWAP', np.array([480]))[0]:.3f}%)", flush=True)

    stats, total, t0 = [], 0, time.time()
    todo = []
    for sym in syms:
        shard = a.out_dir / f"{sym}.parquet"
        if shard.exists() and not a.fresh:
            rows = len(pd.read_parquet(shard, columns=["symbol"]))
            total += rows
            stats.append(SymbolBuildStats(sym, "cached", rows=rows).__dict__)
        else:
            todo.append(sym)
    print(f"  cached={len(stats)} todo={len(todo)}", flush=True)

    tasks = [(sym, str(a.out_dir), anchors, candidates, a.random_count,
              a.stride_min, a.days, a.entry_delay_min) for sym in todo]
    if a.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_worker_build, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                st = fut.result()
                total += st.get("rows", 0)
                stats.append(st)
                if i % 5 == 0 or i == len(tasks):
                    el = (time.time() - t0) / 60
                    print(f"  {i}/{len(tasks)} rows={total}  ({el:.1f}m, ~{el/max(i,1)*len(tasks):.0f}m total)",
                          flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            st = _worker_build(t)
            total += st.get("rows", 0)
            stats.append(st)
            if i % 5 == 0 or i == len(tasks):
                el = (time.time() - t0) / 60
                print(f"  {i}/{len(tasks)} rows={total}  ({el:.1f}m, ~{el/max(i,1)*len(tasks):.0f}m total)",
                      flush=True)

    names = json.dumps(S3.FEATURE_COLUMNS_V3, indent=2)
    (a.out_dir / "feature_names.json").write_text(names, encoding="utf-8")
    (a.out_dir.parent / "feature_names.json").write_text(names, encoding="utf-8")
    ok_stats = [s for s in stats if s.get("first_time")]
    summary = {
        "out_dir": str(a.out_dir), "schema": "v4_binance_y1", "store": "binance_feature",
        "feature_columns": len(S3.FEATURE_COLUMNS_V3),
        "labels": "per-symbol rt_cost + med_abs_funding*h/480 (configs/binance_costs.json + binance_funding.json)",
        "anchors": list(anchors), "h_grid": [a.hmin, a.hmax, a.hstep],
        "random_count": a.random_count, "stride_min": a.stride_min,
        "grid_jitter": "per-symbol 0..55m (stable_seed(sym,7))",
        "days": a.days, "entry_delay_min": int(a.entry_delay_min),
        "symbols_requested": len(syms),
        "shards": len(sorted(a.out_dir.glob("*.parquet"))), "rows": int(total),
        "valid_base_time_min": min((s["first_time"] for s in ok_stats), default=None),
        "valid_base_time_max": max((s["last_time"] for s in ok_stats), default=None),
        "stats": stats,
    }
    (a.out_dir.parent / "dataset_summary.json").write_text(json.dumps(summary, indent=2),
                                                           encoding="utf-8")
    print(f"DONE rows={total} shards={summary['shards']} "
          f"window {summary['valid_base_time_min']}..{summary['valid_base_time_max']}", flush=True)


if __name__ == "__main__":
    main()
