"""Evaluate HC UP/DOWN models and emit clean markdown/csv tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .hc import config as HC
from .hc.data import load_dataset
from .hc.evaluation import build_report, load_folds, score_fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exec", action="store_true", help="use leak-free executable dataset/models")
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--analysis-dir", type=Path, default=None)
    ap.add_argument("--results-md", type=Path, default=None)
    ap.add_argument("--results-csv", type=Path, default=None)
    ap.add_argument("--folds", choices=["fold1", "all"], default=None)
    args = ap.parse_args()

    dataset_dir = args.dataset_dir or (
        HC.SMOKE_DATASET_DIR if args.smoke else HC.EXEC_DATASET_DIR if args.exec else HC.DATASET_DIR
    )
    model_dir = args.model_dir or (
        HC.SMOKE_MODELS_DIR if args.smoke else HC.EXEC_MODELS_DIR if args.exec else HC.MODELS_DIR
    )
    analysis_dir = args.analysis_dir or (
        HC.SMOKE_ANALYSIS_DIR if args.smoke else HC.EXEC_ANALYSIS_DIR if args.exec else HC.ANALYSIS_DIR
    )
    results_md = args.results_md or (
        HC.SMOKE_RESULTS_MD if args.smoke else HC.RESULTS_MD.parent / "HC_EXEC_MODEL_RESULTS.md" if args.exec else HC.RESULTS_MD
    )
    results_csv = args.results_csv or (
        HC.SMOKE_RESULTS_CSV if args.smoke else HC.RESULTS_CSV.parent / "HC_EXEC_MODEL_RESULTS.csv" if args.exec else HC.RESULTS_CSV
    )
    max_folds = 1 if (args.smoke or args.folds == "fold1") else 3

    print(f"loading HC dataset from {dataset_dir}")
    df = load_dataset(dataset_dir)
    folds = load_folds(model_dir, df, max_folds=max_folds)
    scored_frames = []
    for fold in folds:
        print(f"scoring {fold.name}: {fold.test_start} -> {fold.test_end}")
        scored_frames.append(score_fold(df, model_dir, fold))
    scored = pd.concat(scored_frames, ignore_index=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    scored_path = analysis_dir / "hc_scored.parquet"
    scored.to_parquet(scored_path, index=False)
    tables = build_report(scored, folds, out_md=results_md, out_csv=results_csv, smoke=args.smoke)
    print(f"scored -> {scored_path}")
    print(f"results -> {results_md}")
    print(f"csv -> {results_csv}")
    print(tables["D_decision_level"].to_string(index=False))


if __name__ == "__main__":
    main()
