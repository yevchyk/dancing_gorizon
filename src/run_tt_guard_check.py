"""Holdout-integrity verifier for TT — proves the model never trained on the test tail.

Loads the TRAIN dataset the model actually fit on (data/tt_curve) and checks that NO
training row has base_time at/after the cutoff, and reports the embargo gap. Run it
yourself any time you doubt the split:

  .venv/Scripts/python -m src.run_tt_guard_check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .hc.data import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/tt_curve/dataset"))
    ap.add_argument("--summary", type=Path, default=Path("data/tt_curve/dataset_summary.json"))
    a = ap.parse_args()

    summ = json.loads(a.summary.read_text(encoding="utf-8"))
    cut = pd.Timestamp(summ["cutoff"])
    df = load_dataset(a.dataset_dir, columns=["base_time"])
    bt = pd.to_datetime(df["base_time"], utc=True)
    after = int((bt >= cut).sum())

    print("TRAIN dataset (data/tt_curve) = exactly what the model fit on:")
    print(f"  rows                       : {len(df):,}")
    print(f"  cutoff (boundary)          : {cut}")
    print(f"  train base_time MIN        : {bt.min()}")
    print(f"  train base_time MAX        : {bt.max()}")
    print(f"  >>> rows AT/AFTER cutoff   : {after}   (MUST be 0)")
    print(f"  >>> embargo (cutoff - max) : {cut - bt.max()}")
    print(f"  config holdout_days={summ.get('holdout_days')}  embargo_min={summ.get('embargo_min')}  "
          f"data_edge={summ.get('data_edge')}")
    print()
    print("VERDICT:", "CLEAN — no training row reaches the holdout." if after == 0
          else f"LEAK — {after} training rows are inside the holdout!")


if __name__ == "__main__":
    main()
