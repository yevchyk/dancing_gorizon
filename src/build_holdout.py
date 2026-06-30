"""Build a fresh scored holdout over the last N days and save it as a test run.

Used to benchmark on a window the OLD models could not have trained on
(e.g. last 3 days). Produces a scored.parquet (new-model prob columns + targets)
that both run_optimize_thresholds and run_legacy_benchmark can consume.

Usage:
  python -m src.build_holdout --days 3 --anchors 80
"""

from __future__ import annotations

import argparse
import datetime as dt

from . import config as C
from .database import CandleStore
from .features import CurveBuilder
from .dataset import AnchorSampler, TargetBuilder, DatasetCollector
from .training import ModelRegistry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=float, default=3)
    p.add_argument("--anchors", type=int, default=80, help="anchors per symbol")
    p.add_argument("--min-date", type=str, default=None,
                   help="drop anchors at/before this UTC time (strict leak-free floor)")
    args = p.parse_args()

    store = CandleStore(C.CANDLES_DIR)
    curve = CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    sampler = AnchorSampler(args.anchors, start_offset_days=args.days, end_offset_days=0)
    chunks = C.CHUNKS_DIR / f"_holdout{args.days}d"
    collector = DatasetCollector(store, curve, sampler, TargetBuilder(), chunks)

    out_path = C.DATASETS_DIR / f"holdout_{args.days}d.parquet"
    collector.collect(store.symbols(), out_path)

    import pandas as pd
    holdout = pd.read_parquet(out_path)
    if args.min_date:
        floor = pd.Timestamp(args.min_date, tz="UTC")
        before = len(holdout)
        at = pd.to_datetime(holdout["anchor_time"], utc=True)
        holdout = holdout[at > floor].reset_index(drop=True)
        print(f"leak-free filter > {floor}: kept {len(holdout)}/{before} anchors")
    at = pd.to_datetime(holdout["anchor_time"], utc=True)
    print(f"anchor window: {at.min()} .. {at.max()}")
    registry = ModelRegistry.load_default()
    scored = registry.score(holdout)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = C.TEST_RESULTS_DIR / f"run_{args.days}d_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(run_dir / "scored.parquet", index=False)
    print(f"{args.days}d holdout: {len(scored)} rows -> {run_dir / 'scored.parquet'}")


if __name__ == "__main__":
    main()
