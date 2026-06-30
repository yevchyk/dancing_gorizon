"""Paper-trade the directional strategy on the holdout and cross-check block 5.

Usage:
  python -m src.run_paper_backtest
  python -m src.run_paper_backtest --top-pct 5 --no-veto
  python -m src.run_paper_backtest --scored outputs/test_results/run_XX  (reuse a run)
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from . import config as C
from .training import ModelRegistry
from .trading import PaperBacktester, load_signal_thresholds


def _latest_scored() -> Path:
    runs = sorted(p for p in C.TEST_RESULTS_DIR.glob("run_*") if p.is_dir())
    if not runs:
        raise SystemExit("no test run found; run `python -m src.run_tests` first")
    return runs[-1] / "scored.parquet"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", type=str, default=None,
                   help="path to a scored.parquet (default: latest test run)")
    p.add_argument("--top-pct", type=float, default=1.0,
                   help="per-model threshold slice from block 5 (1/5/10/25)")
    p.add_argument("--no-veto", action="store_true", help="disable stability veto")
    args = p.parse_args()

    scored_path = Path(args.scored) if args.scored else _latest_scored()
    scored = pd.read_parquet(scored_path)
    registry = ModelRegistry.load_default()
    thresholds = load_signal_thresholds(top_pct=args.top_pct)
    print(f"scored={scored_path}  rows={len(scored)}  "
          f"tuned_thresholds={len(thresholds)} (top {args.top_pct}%)")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = C.TRADING_LOGS_DIR / f"paper_{ts}"
    bt = PaperBacktester(registry, thresholds=thresholds,
                         use_stability_veto=not args.no_veto)
    summary = bt.run(scored, out_dir)
    print(summary.to_string(index=False))
    print(f"\npaper backtest -> {out_dir}")


if __name__ == "__main__":
    main()
