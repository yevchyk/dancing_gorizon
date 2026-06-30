"""Per-coin honesty check on the corrected PnL: for the tradeable models at
their optimal thresholds, which coins fired confident signals but LOST money
(deceptive) vs which were reliable.

Usage:
  python -m src.run_coin_analysis --scored <scored.parquet> --thresholds-csv <optimal.csv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config as C
from .database import CandleStore
from .training import ModelRegistry
from .trading.optimizer import _resolve_one, HORIZON_MIN
from .trading.timeutil import index_to_ns, anchors_to_ns
from .trading.thresholds import load_optimal_thresholds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", required=True)
    p.add_argument("--thresholds-csv", required=True)
    p.add_argument("--models", default="up_15m,up_30m,up_1h,down_5m")
    p.add_argument("--min-signals", type=int, default=5)
    args = p.parse_args()

    scored = pd.read_parquet(args.scored)
    registry = ModelRegistry.load_default()
    thr = load_optimal_thresholds(Path(args.thresholds_csv))
    models = [m for m in args.models.split(",") if m in registry.names]
    store = CandleStore(C.CANDLES_DIR)

    rows: list[dict] = []
    for symbol, g in scored.groupby("symbol"):
        candles = store.load(symbol)
        if candles is None:
            continue
        ts = index_to_ns(candles.index)
        high, low, close = (candles[c].to_numpy(float) for c in ("high", "low", "close"))
        anchors_ns = anchors_to_ns(g["anchor_time"])
        for name in models:
            spec = registry.spec(name)
            side = "long" if spec.direction == "up" else "short"
            move, hmin = spec.horizon.move_pct, HORIZON_MIN[spec.horizon.label]
            t = thr.get(name, 0.9)
            probs = g[f"prob_{name}"].to_numpy(float)
            for a_ns, pr in zip(anchors_ns, probs):
                if pr < t:
                    continue
                res = _resolve_one(ts, high, low, close, int(a_ns), side, move,
                                   hmin, C.STOP_PCT_RATIO, C.OKX_FEE_PER_SIDE)
                if res is None:
                    continue
                rows.append({"symbol": symbol, "model": name,
                             "won": res[0], "pnl_pct": res[1]})

    df = pd.DataFrame(rows)
    if df.empty:
        print("no signals")
        return

    per_coin = (df.groupby("symbol")
                  .agg(n_signals=("pnl_pct", "size"), win_rate=("won", "mean"),
                       avg_pnl=("pnl_pct", "mean"), total_pnl=("pnl_pct", "sum"))
                  .reset_index())
    per_coin = per_coin[per_coin["n_signals"] >= args.min_signals]
    for col in ("win_rate", "avg_pnl", "total_pnl"):
        per_coin[col] = per_coin[col].round(4)

    deceptive = per_coin.sort_values("avg_pnl").head(15)
    reliable = per_coin.sort_values("avg_pnl", ascending=False).head(15)

    out = C.OUTPUTS_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    per_coin.sort_values("avg_pnl").to_csv(out / "coin_pnl.csv", index=False)

    print(f"models={models}  coins with >= {args.min_signals} signals: {len(per_coin)}")
    print(f"overall: n={len(df)}  win={df['won'].mean():.3f}  avg_pnl={df['pnl_pct'].mean():.4f}%")
    print("\n=== MOST DECEPTIVE COINS (confident signals, worst PnL) ===")
    print(deceptive.to_string(index=False))
    print("\n=== MOST RELIABLE COINS (best PnL) ===")
    print(reliable.to_string(index=False))
    print(f"\nfull -> {out / 'coin_pnl.csv'}")


if __name__ == "__main__":
    main()
