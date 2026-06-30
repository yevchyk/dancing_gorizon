"""Per-day robustness check: for each tradeable model at its optimal threshold,
break realized PnL down by UTC day. Goal: see whether every day is green (or at
least not red) across models, instead of one big day carrying the average.

Usage:
  python -m src.run_daily_breakdown --scored <scored.parquet> --thresholds-csv <optimal.csv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
        days = pd.to_datetime(g["anchor_time"], utc=True).dt.strftime("%m-%d").to_numpy()
        for name in models:
            spec = registry.spec(name)
            side = "long" if spec.direction == "up" else "short"
            move, hmin = spec.horizon.move_pct, HORIZON_MIN[spec.horizon.label]
            t = thr.get(name, 0.9)
            probs = g[f"prob_{name}"].to_numpy(float)
            for a_ns, pr, day in zip(anchors_ns, probs, days):
                if pr < t:
                    continue
                res = _resolve_one(ts, high, low, close, int(a_ns), side, move,
                                   hmin, C.STOP_PCT_RATIO, C.OKX_FEE_PER_SIDE)
                if res is None:
                    continue
                rows.append({"day": day, "model": name,
                             "won": res[0], "pnl_pct": res[1]})

    df = pd.DataFrame(rows)
    if df.empty:
        print("no signals")
        return

    # avg PnL per (day, model)
    pnl = df.pivot_table(index="day", columns="model", values="pnl_pct", aggfunc="mean")
    n = df.pivot_table(index="day", columns="model", values="pnl_pct", aggfunc="size")
    pnl = pnl.reindex(columns=models)
    pnl["ALL"] = df.groupby("day")["pnl_pct"].mean()

    out = C.OUTPUTS_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pnl.round(3).to_csv(out / "daily_pnl.csv")
    n.to_csv(out / "daily_n.csv")

    pd.set_option("display.width", 200)
    print("=== AVG PnL %/trade per DAY x MODEL ===")
    print(pnl.round(3).to_string())
    print("\n=== n trades per DAY x MODEL ===")
    print(n.reindex(columns=models).fillna(0).astype(int).to_string())

    print("\n=== DAY CONSISTENCY ===")
    for m in models + ["ALL"]:
        if m not in pnl.columns:
            continue
        col = pnl[m].dropna()
        green = int((col > 0).sum())
        red = int((col < 0).sum())
        print(f"  {m:<8} green_days={green}  red_days={red}  "
              f"worst={col.min():+.3f}%  best={col.max():+.3f}%")
    print(f"\nfull -> {out / 'daily_pnl.csv'}")


if __name__ == "__main__":
    main()
