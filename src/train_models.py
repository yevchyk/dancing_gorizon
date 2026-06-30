"""Train all 15 models from the collected dataset.

Usage:
  python -m src.train_models
  python -m src.train_models --dataset data/datasets/train.parquet --iterations 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config as C
from .features import CurveBuilder
from .training import HorizonSlicer, ModelTrainer, MultiModelPipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=C.DATASETS_DIR / "train.parquet")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--depth", type=int, default=5)
    args = p.parse_args()

    curve = CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    slicer = HorizonSlicer(curve)
    trainer = ModelTrainer(slicer, iterations=args.iterations, depth=args.depth,
                           test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
    MultiModelPipeline(trainer).run(args.dataset)


if __name__ == "__main__":
    main()
