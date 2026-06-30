"""Benchmark the OLD ml_predictor models on the SAME v2 holdout, with the SAME
target/stop PnL, so old vs new is a fair head-to-head.

Usage:
  python -m src.run_legacy_benchmark --group directional_p05
  python -m src.run_legacy_benchmark --group directional_p03 --max-per-symbol 40
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .legacy import OldFeatureBuilder, LegacyModelGroup
from .legacy.old_scorer import OLD_HORIZONS
from .trading.optimizer import _resolve_one
from .trading.timeutil import index_to_ns, anchors_to_ns

GRID = tuple(round(x, 2) for x in np.arange(0.40, 0.971, 0.01))


def _latest_scored() -> Path:
    runs = sorted(p for p in C.TEST_RESULTS_DIR.glob("run_*") if p.is_dir())
    if not runs:
        raise SystemExit("no test run found; run `python -m src.run_tests` first")
    return runs[-1] / "scored.parquet"


def _name_meta(name: str) -> tuple[str, int]:
    d, h = name.split("_")
    return ("long" if d == "up" else "short"), int(h.rstrip("m"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="directional_p05")
    p.add_argument("--scored", default=None)
    p.add_argument("--max-per-symbol", type=int, default=0, help="0 = all anchors")
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--select-min", type=float, default=0.60)
    p.add_argument("--stop-ratio", type=float, default=C.STOP_PCT_RATIO)
    args = p.parse_args()

    scored_path = Path(args.scored) if args.scored else _latest_scored()
    holdout = pd.read_parquet(scored_path)[["symbol", "anchor_time"]]
    store = CandleStore(C.CANDLES_DIR)
    btc = store.load("BTC_USDT_SWAP")
    builder = OldFeatureBuilder(btc)
    group = LegacyModelGroup(args.group)
    move = LegacyModelGroup.move_pct(args.group)
    print(f"group={args.group}  models={group.names}  move_pct={move}  "
          f"anchors={len(holdout)}")

    records: list[dict] = []
    for symbol, g in holdout.groupby("symbol"):
        candles = store.load(symbol)
        if candles is None:
            continue
        if args.max_per_symbol and len(g) > args.max_per_symbol:
            g = g.sample(args.max_per_symbol, random_state=C.RANDOM_STATE)
        anchors_ns = anchors_to_ns(g["anchor_time"])
        feat_rows = builder.build_rows(candles, anchors_ns)
        probs = group.score_rows(feat_rows)

        ts = index_to_ns(candles.index)
        high = candles["high"].to_numpy(float)
        low = candles["low"].to_numpy(float)
        close = candles["close"].to_numpy(float)
        for name in group.names:
            side, hmin = _name_meta(name)
            pcol = probs[f"prob_{name}"].to_numpy()
            for a_ns, pr in zip(anchors_ns, pcol):
                res = _resolve_one(ts, high, low, close, int(a_ns), side, move,
                                   hmin, args.stop_ratio, C.OKX_FEE_PER_SIDE)
                if res is None:
                    continue
                records.append({"model": name, "prob": float(pr),
                                "won": res[0], "pnl_pct": res[1]})

    df = pd.DataFrame(records)
    ts_now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = C.TRADING_LOGS_DIR / f"legacy_{args.group}_{ts_now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    optima = []
    for name, gg in df.groupby("model"):
        prob, won, pnl = gg["prob"].to_numpy(), gg["won"].to_numpy(), gg["pnl_pct"].to_numpy()
        take_all = float(pnl.mean())
        per = []
        for thr in GRID:
            m = prob >= thr
            n = int(m.sum())
            if n == 0:
                continue
            per.append({"model": name, "threshold": thr, "n_trades": n,
                        "win_rate": round(float(won[m].mean()), 4),
                        "avg_pnl_pct": round(float(pnl[m].mean()), 4),
                        "total_pnl_pct": round(float(pnl[m].sum()), 2)})
        elig = [r for r in per if r["n_trades"] >= args.min_trades
                and r["threshold"] >= args.select_min]
        if elig:
            best = max(elig, key=lambda r: r["avg_pnl_pct"])
            optima.append({**best, "baseline_pnl_pct": round(take_all, 4),
                           "edge_vs_drift": round(best["avg_pnl_pct"] - take_all, 4),
                           "tradeable": best["avg_pnl_pct"] > 0})
        pd.DataFrame(per).to_csv(out_dir / "threshold_sweep.csv",
                                 mode="a", header=not (out_dir / "threshold_sweep.csv").exists(),
                                 index=False)

    opt = pd.DataFrame(optima).sort_values("avg_pnl_pct", ascending=False)
    opt.to_csv(out_dir / "optimal_thresholds.csv", index=False)
    print("\n=== OLD MODELS — OPTIMAL THRESHOLD (selective zone, n>=%d) ===" % args.min_trades)
    print(opt[["model", "threshold", "n_trades", "win_rate", "avg_pnl_pct",
               "baseline_pnl_pct", "edge_vs_drift", "tradeable"]].to_string(index=False))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
