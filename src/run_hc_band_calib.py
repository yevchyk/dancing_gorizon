"""Calibration report for a band model (the SIMPLE judge, not gradators).

Philosophy: high predicted probability must mean high realized win%. We bucket
the directional probability p_dir and show, per bucket, the realized win-rate
(net>0 after cost) + avg net + raw directional-hit%. Clean monotone rising
buckets with a high top bucket = good; flat/inverted = garbage.

OOS holdout = rows with base_time >= cutoff (the model trained on exit<=cutoff).

  python -m src.run_hc_band_calib --model-dir models/band_B --dataset-dir data/hc_bands_v2/dataset \
      --cutoff-local "2026-06-07 07:10" --horizon-min 30 --horizon-max 90
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .hc.data import load_dataset
from .run_hc_prod_train import parse_cutoff

BUCKETS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.0001)]


def load_folds(model_dir: Path):
    folds = []
    for sub in sorted(model_dir.iterdir()):
        up, dn = sub / "up.cbm", sub / "down.cbm"
        if up.exists() and dn.exists():
            u = CatBoostClassifier(); u.load_model(up)
            d = CatBoostClassifier(); d.load_model(dn)
            folds.append((u, d))
    if not folds:
        raise FileNotFoundError(f"no up/down folds under {model_dir}")
    return folds


def feature_cols(model_dir: Path) -> list[str]:
    fn = model_dir / "feature_names.json"
    if fn.exists():
        return json.loads(fn.read_text(encoding="utf-8"))
    snap = model_dir / "config_snapshot.json"
    if snap.exists():
        s = json.loads(snap.read_text(encoding="utf-8"))
        if s.get("feature_columns"):
            return s["feature_columns"]
    raise FileNotFoundError(f"no feature_names.json / feature_columns in {model_dir}")


def table(df: pd.DataFrame, cost: float, label: str) -> None:
    side_long = df["up_prob"].to_numpy() >= df["down_prob"].to_numpy()
    p_dir = np.where(side_long, df["up_prob"], df["down_prob"])
    ret = df["ret_pct"].to_numpy()
    ret_side = np.where(side_long, ret, -ret)
    net = ret_side - cost
    hit = np.where(side_long, df["up_label"].to_numpy(), df["down_label"].to_numpy())  # raw directional threshold hit
    print(f"\n=== {label} | cost {cost:.2f}% | legs={len(df)} ===")
    print(f"  {'bucket p_dir':14s} {'n':>6s} {'win%(net>0)':>11s} {'avg net%':>9s} {'sum$@$15':>9s} {'rawhit%':>8s}")
    for lo, hi in BUCKETS:
        m = (p_dir >= lo) & (p_dir < hi)
        n = int(m.sum())
        if not n:
            print(f"  {lo:.2f}-{hi:.2f}      {0:>6d}        --        --        --       --")
            continue
        win = float((net[m] > 0).mean())
        avg = float(net[m].mean())
        s15 = float((15.0 * net[m] / 100.0).sum())
        rh = float(hit[m].mean())
        print(f"  {lo:.2f}-{hi:.2f}      {n:>6d}   {win*100:>9.0f}%   {avg:>+8.3f}   {s15:>+8.2f}   {rh*100:>6.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/hc_bands_v2/dataset"))
    ap.add_argument("--cutoff-local", required=True, help="holdout start (Kyiv); OOS = base_time >= cutoff")
    ap.add_argument("--cutoff-max-local", default=None, help="optional holdout end (Kyiv) for a disjoint window")
    ap.add_argument("--horizon-min", type=int, default=None)
    ap.add_argument("--horizon-max", type=int, default=None)
    ap.add_argument("--cost", type=float, default=0.75)
    ap.add_argument("--extra-cost", type=float, default=None, help="also print a 2nd table at this cost (band A: 0.45)")
    args = ap.parse_args()

    cutoff = parse_cutoff(args.cutoff_local)
    feat = feature_cols(args.model_dir)
    need = list(dict.fromkeys(feat + ["symbol", "base_time", "horizon_minutes", "ret_pct", "up_label", "down_label"]))
    df = load_dataset(args.dataset_dir, columns=need)
    df["base_time"] = pd.to_datetime(df["base_time"], utc=True)
    df = df[df["base_time"] >= cutoff]
    if args.cutoff_max_local:
        df = df[df["base_time"] <= parse_cutoff(args.cutoff_max_local)]
    if args.horizon_min is not None:
        df = df[df["horizon_minutes"] >= args.horizon_min]
    if args.horizon_max is not None:
        df = df[df["horizon_minutes"] <= args.horizon_max]
    df = df.reset_index(drop=True)
    if df.empty:
        raise SystemExit("no holdout rows for this band/window")

    folds = load_folds(args.model_dir)
    X = df[feat]
    up = np.mean([u.predict_proba(X)[:, 1] for u, _ in folds], axis=0)
    dn = np.mean([d.predict_proba(X)[:, 1] for _, d in folds], axis=0)
    df["up_prob"] = up; df["down_prob"] = dn

    band = f"h[{args.horizon_min or '*'}..{args.horizon_max or '*'}]"
    print(f"model={args.model_dir.name} {band} holdout base>= {cutoff.isoformat()} "
          f"legs={len(df)} symbols={df['symbol'].nunique()} folds={len(folds)}")
    table(df, args.cost, f"{args.model_dir.name} {band}")
    if args.extra_cost is not None:
        table(df, args.extra_cost, f"{args.model_dir.name} {band} (liquid-cost bound)")


if __name__ == "__main__":
    main()
