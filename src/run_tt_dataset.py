"""Build the ТТ curve dataset (Phase 0) — DANCING_TARAS.md / CLAUDE.md §13.

ONE row per (symbol, scan): MAXIMAL features (561) + the forward price CURVE as a
multi-output target (vol-normalized cumulative log-return, 1-min grid 1..h_max).
Binance-only. The last --holdout-days are RESERVED for the user's own test and are
never built into the dataset (cutoff = data-edge − holdout-days).

  # smoke (fast, no regime pre-pass):
  python -m src.run_tt_dataset --limit-symbols 2 --days 7 --no-regime \
      --out-dir data/tt_smoke/dataset --fresh
  # real build:
  python -m src.run_tt_dataset --workers 8 --fresh
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .hc import config as HC
from .markets import REGISTRY, Store
from . import config as C

BINANCE_CANDLES = Path("data/binance/candles")
BINANCE_ERA_START = pd.Timestamp("2025-06-01T00:00:00Z")

# Point the hc candle-prep at the Binance store BEFORE the builders run. Done at
# MODULE level so spawned workers (Windows spawn) re-apply it on import.
REGISTRY["binance_feature"] = Store(
    "binance_feature", "crypto", "feature", "1m",
    C.ROOT / BINANCE_CANDLES, C.ROOT / "configs" / "binance_train_universe.json",
    "Binance USDT-M 365d 1m store (ТТ curve dataset).")
HC.STORE_KEY = "binance_feature"
HC.HC_ERA_START = BINANCE_ERA_START

from .hc.data import SymbolBuildStats, prepare_btc_frames, stable_seed  # noqa: E402
from .hc.data_v5 import build_market_frame  # noqa: E402
from .tt import schema_tt as STT  # noqa: E402
from .tt.data_tt import build_symbol_curve_tt  # noqa: E402

UNIVERSE = Path("configs/binance_train_universe.json")
TRADE_UNIVERSE = Path("configs/binance_universe_trade.json")   # breadth panel (frozen)

_W_BTC = None
_W_MKT = None


def _worker(task: tuple) -> dict:
    global _W_BTC, _W_MKT
    (sym, out_dir, cutoff, h_max, step, n_points, stride_min, days,
     entry_delay, vol_window, with_regime, target_mode, market_cache) = task
    if _W_BTC is None:
        _W_BTC = prepare_btc_frames()
    if with_regime and _W_MKT is None:
        _W_MKT = pd.read_parquet(market_cache)
    offset = (stable_seed(sym, 7) % (stride_min // 5)) * 5 if stride_min >= 10 else 0
    df, stat = build_symbol_curve_tt(
        sym, btc_frames=_W_BTC, market=_W_MKT, cutoff=cutoff, h_max=h_max, step=step,
        n_points=n_points, stride_min=stride_min, days=days, entry_delay_min=entry_delay,
        grid_offset_min=offset, vol_window=vol_window, with_regime=with_regime, target_mode=target_mode)
    if len(df):
        df.to_parquet(Path(out_dir) / f"{sym}.parquet", index=False)
    return stat.__dict__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data/tt_curve/dataset"))
    ap.add_argument("--universe", type=Path, default=UNIVERSE)
    ap.add_argument("--stride-min", type=int, default=60)
    ap.add_argument("--days", type=int, default=None, help="lookback before cutoff; default full series")
    ap.add_argument("--h-max", type=int, default=STT.TT_HORIZON_MAX)
    ap.add_argument("--h-step", type=int, default=STT.TT_HORIZON_STEP)
    ap.add_argument("--n-points", type=int, default=STT.TT_N_POINTS)
    ap.add_argument("--entry-delay-min", type=int, default=HC.EXEC_ENTRY_DELAY_MIN)
    ap.add_argument("--vol-window", type=int, default=1440, help="1-min bars for the sigma normalizer")
    ap.add_argument("--holdout-days", type=int, default=4, help="RESERVED tail (user's test) — never built")
    ap.add_argument("--cutoff", default=None,
                    help="explicit ISO cutoff in UTC (overrides --holdout-days); use a clean day boundary "
                         "(Kyiv midnight = ...T21:00:00Z) so NO calendar day is half-train/half-test")
    ap.add_argument("--limit-symbols", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--fresh-market", action="store_true")
    ap.add_argument("--no-regime", action="store_true", help="DROP the 18 regime scalars from features (TT v2)")
    ap.add_argument("--target-mode", choices=["volnorm", "ratioA", "ratioB"], default="volnorm",
                    help="target curve: volnorm=vol-norm cumret (legacy); ratioA=price/entry (cumulative); ratioB=price/prev-min")
    ap.add_argument("--market-cache", type=Path, default=Path("data/tt_curve/market_frame.parquet"))
    a = ap.parse_args()

    with_regime = not a.no_regime
    syms = json.loads(a.universe.read_text())["symbols"]
    if a.limit_symbols:
        syms = syms[:a.limit_symbols]

    btc_frames = prepare_btc_frames()
    data_edge = btc_frames["5m"].index.max()
    cutoff = data_edge - pd.Timedelta(days=a.holdout_days)
    if a.cutoff:
        c = pd.Timestamp(a.cutoff)
        cutoff = c.tz_localize("UTC") if c.tzinfo is None else c.tz_convert("UTC")
    hold_days_eff = round((data_edge - cutoff).total_seconds() / 86400.0, 2)
    n_nodes = len(STT.target_horizons_tt(a.h_max, a.h_step))
    feat_cols = STT.feature_names_tt(a.n_points, include_regime=with_regime)
    print(f"ТТ curve dataset -> {a.out_dir}\n"
          f"  store=binance_feature edge={data_edge.isoformat()} "
          f"cutoff={cutoff.isoformat()} (holdout {hold_days_eff}d RESERVED)\n"
          f"  syms={len(syms)} stride={a.stride_min}m(+jitter) days={a.days or 'all'} "
          f"n_points={a.n_points} feats={len(feat_cols)} "
          f"target=1..{a.h_max}/{a.h_step} ({n_nodes} nodes) regime={'on' if with_regime else 'OFF'}",
          flush=True)

    if with_regime:
        trade = json.loads(TRADE_UNIVERSE.read_text())
        trade = trade.get("symbols", trade) if isinstance(trade, dict) else trade
        t0 = time.time()
        print(f"  market frame (breadth over {len(trade)} frozen trade syms) -> {a.market_cache}", flush=True)
        mkt = build_market_frame(trade, cache=a.market_cache, fresh=a.fresh_market)
        print(f"    {len(mkt)} 5m bars, {mkt.index.min()}..{mkt.index.max()} ({(time.time()-t0)/60:.1f}m)", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.fresh:
        for p in a.out_dir.glob("*.parquet"):
            p.unlink()

    stats: list[dict] = []
    total = 0
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

    tasks = [(sym, str(a.out_dir), cutoff, a.h_max, a.h_step, a.n_points, a.stride_min,
              a.days, a.entry_delay_min, a.vol_window, with_regime, a.target_mode, str(a.market_cache))
             for sym in todo]
    t1 = time.time()
    if a.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_worker, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                st = fut.result()
                total += st.get("rows", 0)
                stats.append(st)
                if i % 5 == 0 or i == len(tasks):
                    el = (time.time() - t1) / 60
                    print(f"  {i}/{len(tasks)} rows={total} ({el:.1f}m, ~{el/max(i,1)*len(tasks):.0f}m total)", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            st = _worker(t)
            total += st.get("rows", 0)
            stats.append(st)
            if i % 5 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} rows={total} ({(time.time()-t1)/60:.1f}m)", flush=True)

    (a.out_dir / "feature_names.json").write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")
    (a.out_dir.parent / "feature_names.json").write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")
    (a.out_dir / "target_names.json").write_text(
        json.dumps(STT.target_columns_tt(a.h_max, a.h_step), indent=2), encoding="utf-8")
    ok_stats = [s for s in stats if s.get("first_time")]
    summary = {
        "out_dir": str(a.out_dir), "schema": "tt_curve_binance", "store": "binance_feature",
        "feature_columns": len(feat_cols), "n_points": a.n_points, "regime": with_regime,
        "target_nodes": n_nodes, "h_max": a.h_max, "h_step": a.h_step,
        "entry_delay_min": int(a.entry_delay_min), "vol_window": a.vol_window,
        "stride_min": a.stride_min, "days": a.days, "target_mode": a.target_mode,
        "holdout_days": hold_days_eff, "holdout_days_arg": a.holdout_days,
        "data_edge": data_edge.isoformat(), "cutoff": cutoff.isoformat(),
        "symbols_requested": len(syms),
        "shards": len(sorted(a.out_dir.glob("*.parquet"))), "rows": int(total),
        "valid_base_time_min": min((s["first_time"] for s in ok_stats), default=None),
        "valid_base_time_max": max((s["last_time"] for s in ok_stats), default=None),
        "stats": stats,
    }
    (a.out_dir.parent / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"DONE rows={total} shards={summary['shards']} "
          f"window {summary['valid_base_time_min']}..{summary['valid_base_time_max']}", flush=True)


if __name__ == "__main__":
    main()
