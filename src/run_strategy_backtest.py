"""Backtest the strategy (regime + horizon-agreement gates) vs the flat baseline
on a scored window, to judge whether the gates add value.

Usage:
  python -m src.run_strategy_backtest --scored outputs/test_results/run_XX/scored.parquet
  python -m src.run_strategy_backtest --agreement 2 --no-regime
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from . import config as C
from .training import ModelRegistry
from .trading import load_signal_thresholds
from .trading.thresholds import load_optimal_thresholds
from .trading.strategy import Strategy
from .trading.regime import RegimeDetector
from .trading.strategy_backtester import StrategyBacktester


def _latest_scored() -> Path:
    runs = sorted(p for p in C.TEST_RESULTS_DIR.glob("run_*") if p.is_dir())
    if not runs:
        raise SystemExit("no test run found")
    return runs[-1] / "scored.parquet"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", default=None)
    p.add_argument("--top-pct", type=float, default=5.0,
                   help="per-model threshold slice from block 5 (1/5/10/25)")
    p.add_argument("--thresholds-csv", default=None,
                   help="optimal_thresholds.csv from a ThresholdOptimizer run")
    p.add_argument("--agreement", type=int, default=C.AGREEMENT_MIN)
    p.add_argument("--no-regime", action="store_true")
    p.add_argument("--regime-lookback", type=int, default=C.REGIME_LOOKBACK_MIN)
    p.add_argument("--all-models", action="store_true",
                   help="trade every directional model, not just the allow-list")
    args = p.parse_args()

    scored_path = Path(args.scored) if args.scored else _latest_scored()
    scored = pd.read_parquet(scored_path)
    registry = ModelRegistry.load_default()
    thresholds = (load_optimal_thresholds(Path(args.thresholds_csv))
                  if args.thresholds_csv else load_signal_thresholds(top_pct=args.top_pct))
    tradeable = None if args.all_models else C.TRADEABLE_MODELS

    strategy = Strategy(registry, thresholds=thresholds, agreement_min=args.agreement,
                        use_regime=not args.no_regime, tradeable=tradeable)
    regime = RegimeDetector(lookback_min=args.regime_lookback)
    print(f"scored={scored_path}  rows={len(scored)}  agreement={args.agreement}  "
          f"regime={not args.no_regime}  tradeable={tradeable}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = C.TRADING_LOGS_DIR / f"strategy_{ts}"
    report = StrategyBacktester(registry, strategy, regime).run(scored, out_dir)

    pd.set_option("display.width", 200)
    print("\n=== FLAT vs STRATEGY ===")
    print(report.to_string(index=False))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
