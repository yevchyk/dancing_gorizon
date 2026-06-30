"""Train HC UP/DOWN CatBoost models."""

from __future__ import annotations

import argparse
from pathlib import Path

from .hc import config as HC
from .hc.data import load_dataset
from .hc.train import train_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exec", action="store_true", help="use leak-free executable dataset/models")
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--folds", choices=["fold1", "all"], default=None)
    ap.add_argument("--fold-plan", choices=["default", "exec_v2"], default=None)
    ap.add_argument("--primary-days", type=int, default=1)
    ap.add_argument("--spring-days", type=int, default=14)
    ap.add_argument("--task-type", choices=["GPU", "CPU"], default=None)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--l2-leaf-reg", type=float, default=None)
    ap.add_argument("--border-count", type=int, default=None)
    ap.add_argument("--gpu-ram-part", type=float, default=None)
    ap.add_argument("--od-wait", type=int, default=None)
    ap.add_argument("--verbose", type=int, default=100)
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="uniform random row subsample (time-distribution preserved) to fit GPU memory")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir or (
        HC.SMOKE_DATASET_DIR if args.smoke else HC.EXEC_DATASET_DIR if args.exec else HC.DATASET_DIR
    )
    model_dir = args.model_dir or (
        HC.SMOKE_MODELS_DIR if args.smoke else HC.EXEC_MODELS_DIR if args.exec else HC.MODELS_DIR
    )
    task_type = args.task_type or ("CPU" if args.smoke else "GPU")
    iterations = args.iterations if args.iterations is not None else (80 if args.smoke else 4000)
    max_folds = 1 if (args.smoke or args.folds == "fold1") else 3
    fold_plan = args.fold_plan or ("exec_v2" if args.exec else "default")

    print(f"loading HC dataset from {dataset_dir}")
    df = load_dataset(dataset_dir)
    if args.sample_frac < 1.0:
        before = len(df)
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
        print(f"subsampled rows {before} -> {len(df)} (frac={args.sample_frac})")
    print(
        f"dataset rows={len(df)} symbols={df['symbol'].nunique()} "
        f"features={len(HC.FEATURE_COLUMNS)} time={df['base_time'].min()}..{df['base_time'].max()}"
    )
    train_all(
        df,
        model_dir,
        max_folds=max_folds,
        task_type=task_type,
        devices=args.devices,
        iterations=iterations,
        depth=args.depth,
        verbose=args.verbose,
        fold_plan=fold_plan,
        primary_days=args.primary_days,
        spring_days=args.spring_days,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        border_count=args.border_count,
        gpu_ram_part=args.gpu_ram_part,
        od_wait=args.od_wait,
    )
    print(f"models -> {model_dir}")


if __name__ == "__main__":
    main()
