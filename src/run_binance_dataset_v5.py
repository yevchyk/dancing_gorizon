"""Build the v5 (regime-block) Binance dataset — BINANCE_V5_PLAN §1.

Same grid discipline as v4 build (stride 60m + per-symbol jitter, per-symbol
cost+funding thresholds, per-symbol shards, resume-able) with the v5 changes:
  * horizons 30..320 by 5 (user: "далі воно бачить погано"), anchors (30,120,320)
  * +18 regime columns (market frame pre-pass cached once, workers reuse it)

  python -m src.run_binance_dataset_v5 --workers 8
  python -m src.run_binance_dataset_v5 --limit-symbols 2 --days 5 \
      --out-dir data/binance_smoke_v5/dataset --fresh     # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .hc import config as HC
from .hc import schema_v5 as S5
from .markets import REGISTRY, Store
from . import config as C

BINANCE_CANDLES = Path("data/binance/candles")
BINANCE_ERA_START = pd.Timestamp("2025-06-01T00:00:00Z")

REGISTRY.setdefault("binance_feature", Store(
    "binance_feature", "crypto", "feature", "1m",
    C.ROOT / BINANCE_CANDLES, C.ROOT / "configs" / "binance_train_universe.json",
    "Binance USDT-M 365d 1m store for the year-rebuild (BINANCE_PLAN.md)."))
HC.STORE_KEY = "binance_feature"
HC.HC_ERA_START = BINANCE_ERA_START

from .hc.data import SymbolBuildStats, prepare_btc_frames, stable_seed  # noqa: E402
from .hc.data_v5 import build_market_frame, build_symbol_frame_v5  # noqa: E402
from .run_binance_dataset import make_threshold_fn  # noqa: E402

UNIVERSE = Path("configs/binance_train_universe.json")
TRADE_UNIVERSE = Path("configs/binance_universe_trade.json")  # breadth = frozen 153
COSTS = Path("configs/binance_costs.json")
FUNDING = Path("configs/binance_funding.json")
MARKET_CACHE = Path("data/binance_y1_v5/market_frame.parquet")
ANCHORS = (30, 120, 320)

_W_BTC = None
_W_THR = None
_W_MKT = None


def _worker_build(task: tuple) -> dict:
    global _W_BTC, _W_THR, _W_MKT
    (sym, out_dir, anchors, candidates, random_count, stride_min, days, entry_delay) = task
    if _W_BTC is None:
        _W_BTC = prepare_btc_frames()
    if _W_MKT is None:
        _W_MKT = pd.read_parquet(MARKET_CACHE)
    if _W_THR is None:
        cost_map = {k: float(v) for k, v in json.loads(COSTS.read_text())["costs"].items()}
        fund_map = json.loads(FUNDING.read_text())["symbols"]
        _W_THR = make_threshold_fn(cost_map, fund_map)
    offset = (stable_seed(sym, 7) % (stride_min // 5)) * 5 if stride_min >= 10 else 0
    df, stat = build_symbol_frame_v5(
        sym, btc_frames=_W_BTC, market=_W_MKT, anchors=anchors, candidates=candidates,
        random_count=random_count, stride_min=stride_min, days=days,
        entry_delay_min=entry_delay, threshold_fn=_W_THR, grid_offset_min=offset)
    if len(df):
        df.to_parquet(Path(out_dir) / f"{sym}.parquet", index=False)
    return stat.__dict__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data/binance_y1_v5/dataset"))
    ap.add_argument("--universe", type=Path, default=UNIVERSE)
    ap.add_argument("--stride-min", type=int, default=60)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--hmin", type=int, default=30)
    ap.add_argument("--hmax", type=int, default=320)
    ap.add_argument("--hstep", type=int, default=5)
    ap.add_argument("--random-count", type=int, default=3)
    ap.add_argument("--entry-delay-min", type=int, default=HC.EXEC_ENTRY_DELAY_MIN)
    ap.add_argument("--limit-symbols", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--fresh-market", action="store_true")
    a = ap.parse_args()

    syms = json.loads(a.universe.read_text())["symbols"]
    if a.limit_symbols:
        syms = syms[:a.limit_symbols]
    cost_map = {k: float(v) for k, v in json.loads(COSTS.read_text())["costs"].items()}
    missing = [s for s in syms if s not in cost_map]
    if missing:
        raise SystemExit(f"{len(missing)} symbols lack trusted cost: {missing[:8]}")
    candidates = tuple(range(a.hmin, a.hmax + 1, a.hstep))
    anchors = tuple(x for x in ANCHORS if a.hmin <= x <= a.hmax) or (candidates[0],)

    trade_syms = json.loads(TRADE_UNIVERSE.read_text())
    trade_syms = trade_syms.get("symbols", trade_syms) if isinstance(trade_syms, dict) else trade_syms
    t0 = time.time()
    print(f"v5 market frame (breadth over {len(trade_syms)} frozen trade syms) -> {MARKET_CACHE}", flush=True)
    mkt = build_market_frame(trade_syms, cache=MARKET_CACHE, fresh=a.fresh_market)
    print(f"  market frame: {len(mkt)} 5m bars, {mkt.index.min()} .. {mkt.index.max()} "
          f"({(time.time()-t0)/60:.1f}m)", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.fresh:
        for p in a.out_dir.glob("*.parquet"):
            p.unlink()
    print(f"binance_y1_v5 dataset -> {a.out_dir}\n"
          f"  syms={len(syms)} stride={a.stride_min}m(+jitter) days={a.days or 'all'} "
          f"horizons={a.hmin}..{a.hmax}/{a.hstep} ({len(candidates)} grid, "
          f"{len(anchors)+a.random_count}/snap) feat_cols={len(S5.FEATURE_COLUMNS_V5)}", flush=True)

    stats, total = [], 0
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
    t1 = time.time()
    if a.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_worker_build, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                st = fut.result()
                total += st.get("rows", 0)
                stats.append(st)
                if i % 10 == 0 or i == len(tasks):
                    el = (time.time() - t1) / 60
                    print(f"  {i}/{len(tasks)} rows={total} ({el:.1f}m)", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            st = _worker_build(t)
            total += st.get("rows", 0)
            stats.append(st)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} rows={total}", flush=True)

    names = json.dumps(S5.FEATURE_COLUMNS_V5, indent=2)
    (a.out_dir / "feature_names.json").write_text(names, encoding="utf-8")
    (a.out_dir.parent / "feature_names.json").write_text(names, encoding="utf-8")
    ok_stats = [s for s in stats if s.get("first_time")]
    summary = {
        "out_dir": str(a.out_dir), "schema": "v5_binance_y1", "store": "binance_feature",
        "feature_columns": len(S5.FEATURE_COLUMNS_V5),
        "regime_block": S5.REGIME_COLUMNS_V5,
        "labels": "per-symbol rt_cost + med_abs_funding*h/480",
        "anchors": list(anchors), "h_grid": [a.hmin, a.hmax, a.hstep],
        "random_count": a.random_count, "stride_min": a.stride_min,
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
