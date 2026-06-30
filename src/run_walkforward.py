"""Build the next-version statistics the honest way: independent anchors +
walk-forward retraining. Does NOT touch the production models (live loop keeps
using those).

  1. build a master dataset of independent anchors (1/coin/day) over the span
  2. roll folds: retrain 15 models in-memory, score the later slice, resolve PnL
  3. report out-of-sample per-model win/PnL + threshold behaviour + daily consistency

Usage:
  python -m src.run_walkforward
  python -m src.run_walkforward --span-days 200 --train-days 90 --test-days 14 --folds 4
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .features import CurveBuilder
from .dataset import TargetBuilder, DatasetCollector
from .training.horizon_slicer import HorizonSlicer
from .training.model_trainer import ModelTrainer
from .walkforward import IndependentAnchorSampler, WalkForward


def build_master(span_days: int) -> pd.DataFrame:
    out = C.DATASETS_DIR / "master_independent.parquet"
    if out.exists():
        print(f"master exists: {out}")
        return pd.read_parquet(out)
    store = CandleStore(C.CANDLES_DIR)
    curve = CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    now = pd.Timestamp.now(tz="UTC")
    sampler = IndependentAnchorSampler(start=now - pd.Timedelta(days=span_days), end=now,
                                       per_day=1)
    collector = DatasetCollector(store, curve, sampler, TargetBuilder(),
                                 C.CHUNKS_DIR / "_wf_master")
    collector.collect(store.symbols(), out)
    return pd.read_parquet(out)


def _summary(stats: pd.DataFrame) -> None:
    if stats.empty:
        print("no trades collected")
        return
    print(f"\n=== OUT-OF-SAMPLE per model (all folds, independent anchors) ===")
    for name, g in stats.groupby("model"):
        base = g["pnl_pct"].mean()
        print(f"  {name:<9} n={len(g):>5}  win={g['won'].mean():.3f}  "
              f"avg_pnl={base:+.4f}%")

    print("\n=== per-model at probability thresholds ===")
    for name, g in stats.groupby("model"):
        cells = []
        for thr in (0.80, 0.85, 0.90, 0.95):
            m = g["prob"] >= thr
            if m.sum() >= 20:
                cells.append(f"{thr:.2f}:win={g.loc[m,'won'].mean():.2f} "
                             f"pnl={g.loc[m,'pnl_pct'].mean():+.2f} n={int(m.sum())}")
        if cells:
            print(f"  {name:<9} " + " | ".join(cells))

    print("\n=== daily consistency (avg pnl per day, ALL models) ===")
    daily = stats.groupby("day")["pnl_pct"].mean()
    green = int((daily > 0).sum())
    print(f"  days={len(daily)}  green={green}  red={len(daily)-green}  "
          f"worst={daily.min():+.3f}%  best={daily.max():+.3f}%")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--span-days", type=int, default=200)
    p.add_argument("--train-days", type=int, default=90)
    p.add_argument("--test-days", type=int, default=14)
    p.add_argument("--folds", type=int, default=4)
    args = p.parse_args()

    master = build_master(args.span_days)
    print(f"master: {len(master)} independent anchors")

    store = CandleStore(C.CANDLES_DIR)
    curve = CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    trainer = ModelTrainer(HorizonSlicer(curve))
    wf = WalkForward(store, trainer, train_days=args.train_days,
                     test_days=args.test_days, n_folds=args.folds)
    stats = wf.run(master)

    out = C.OUTPUTS_DIR / "analysis" / "walkforward_stats.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(out, index=False)
    _summary(stats)
    print(f"\nstats -> {out}")


if __name__ == "__main__":
    main()
