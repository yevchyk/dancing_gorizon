"""Build the full training dataset (features + 15 targets) from local candles.

Usage:
  python -m src.collect_dataset                 # all symbols, config defaults
  python -m src.collect_dataset --fresh         # clear chunk cache first
  python -m src.collect_dataset --limit 20      # first N symbols (debug)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import config as C
from .database import CandleStore
from .features import CurveBuilder
from .dataset import AnchorSampler, TargetBuilder, DatasetCollector


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", action="store_true", help="clear chunk cache (needed if anchor count changed)")
    p.add_argument("--limit", type=int, default=0, help="0 = all symbols")
    p.add_argument("--out", type=Path, default=C.DATASETS_DIR / "train.parquet")
    args = p.parse_args()

    if args.fresh and Path(C.CHUNKS_DIR).exists():
        shutil.rmtree(C.CHUNKS_DIR)
        print(f"cleared chunk cache: {C.CHUNKS_DIR}")

    store = CandleStore(C.CANDLES_DIR)
    symbols = store.symbols()
    if args.limit:
        symbols = symbols[: args.limit]

    curve = CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    sampler = AnchorSampler(C.ANCHORS_PER_SYMBOL, C.TRAIN_START_OFFSET_DAYS, C.TRAIN_END_OFFSET_DAYS)
    collector = DatasetCollector(store, curve, sampler, TargetBuilder(), C.CHUNKS_DIR)

    print(f"symbols={len(symbols)} anchors/symbol={C.ANCHORS_PER_SYMBOL} curve_cols={C.CURVE_COLUMNS}")
    collector.collect(symbols, args.out)


if __name__ == "__main__":
    main()
