"""Sweep probability thresholds for all directional models and report the
optimal cutoff (max avg PnL per trade) using realistic target/stop PnL.

Usage:
  python -m src.run_optimize_thresholds
  python -m src.run_optimize_thresholds --stop-ratio 1.5 --min-trades 50
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from . import config as C
from .training import ModelRegistry
from .trading.optimizer import ThresholdOptimizer


def _latest_scored() -> Path:
    runs = sorted(p for p in C.TEST_RESULTS_DIR.glob("run_*") if p.is_dir())
    if not runs:
        raise SystemExit("no test run found; run `python -m src.run_tests` first")
    return runs[-1] / "scored.parquet"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", type=str, default=None)
    p.add_argument("--stop-ratio", type=float, default=C.STOP_PCT_RATIO)
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--floor", type=float, default=0.40)
    args = p.parse_args()

    scored_path = Path(args.scored) if args.scored else _latest_scored()
    scored = pd.read_parquet(scored_path)
    registry = ModelRegistry.load_default()
    print(f"scored={scored_path}  rows={len(scored)}  stop_ratio={args.stop_ratio}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = C.TRADING_LOGS_DIR / f"optimize_{ts}"
    opt = ThresholdOptimizer(registry, floor=args.floor, min_trades=args.min_trades,
                             stop_ratio=args.stop_ratio)
    sweep, optima = opt.run(scored, out_dir)

    print("\n=== OPTIMAL THRESHOLD PER MODEL (max avg PnL in selective zone, n>=%d) ===" % args.min_trades)
    print(optima[["model", "threshold", "n_trades", "win_rate", "avg_pnl_pct",
                  "baseline_pnl_pct", "edge_vs_drift", "tradeable"]].to_string(index=False))
    print(f"\nfull sweep -> {out_dir}")


if __name__ == "__main__":
    main()
