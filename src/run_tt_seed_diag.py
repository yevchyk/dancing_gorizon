"""Diagnostic: do the 3 TT seeds give DIFFERENT predictions?

Scores each `curve_seed*/curve.cbm` SEPARATELY on a sample of the dataset and reports,
per horizon: cross-seed DIRECTION agreement, pairwise correlation of the predicted
curve, and the cross-seed DISPERSION vs signal magnitude (= what the SNR filter uses).
Also splits by conviction (top-decile |ensemble|) to show whether high-conviction
legs are where the seeds actually agree.

  .venv/Scripts/python -m src.run_tt_seed_diag
  .venv/Scripts/python -m src.run_tt_seed_diag --sample 30000 --horizons 30,60,120,180,240
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from .hc.data import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("models/tt_curve"))
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/tt_now/dataset"))
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--horizons", default="30,60,120,240")
    a = ap.parse_args()

    md = a.model_dir
    feat = json.loads((md / "feature_names.json").read_text(encoding="utf-8"))
    std = json.loads((md / "standardizer.json").read_text(encoding="utf-8"))
    mu = np.asarray(std["mu"], dtype="float64")
    sd = np.asarray(std["sd"], dtype="float64")
    seed_paths = sorted(md.glob("curve_seed*/curve.cbm"))
    if not seed_paths:
        raise SystemExit(f"no curve_seed*/curve.cbm under {md}")
    seeds = [p.parent.name for p in seed_paths]
    models = [CatBoostRegressor().load_model(str(p)) for p in seed_paths]
    S = len(models)

    df = load_dataset(a.dataset_dir, columns=list(dict.fromkeys(feat)))
    if len(df) > a.sample:
        df = df.sample(a.sample, random_state=0).reset_index(drop=True)
    X = df[feat]
    P = np.stack([m.predict(X) for m in models])      # [S, n, 240] standardized
    Pd = P * sd + mu                                  # -> vol-norm cumret (the actual signal space)
    horizons = [int(h) for h in a.horizons.split(",") if h.strip()]

    print(f"seeds={seeds}  trees={[m.tree_count_ for m in models]}  rows={len(df)}")
    print(f"{'h':>5} {'dir-agree3':>10} {'pairwise corr':>22} {'med disp':>9} "
          f"{'med disp/|ens|':>14} {'per-seed mean|pred|':>26} {'HIconv dir-agree3':>18}")
    for h in horizons:
        ph = Pd[:, :, h - 1]                          # [S, n] at this horizon
        ens = ph.mean(0)
        sgn = np.sign(ph)
        agree = (np.abs(sgn.sum(0)) == S)             # all seeds same sign
        corrs = [round(float(np.corrcoef(ph[i], ph[j])[0, 1]), 3)
                 for i in range(S) for j in range(i + 1, S)]
        disp = ph.std(0)                              # cross-seed std per leg (vol-norm)
        rel = disp / (np.abs(ens) + 1e-9)
        per_seed_mag = [round(float(np.mean(np.abs(ph[s]))), 4) for s in range(S)]
        hi = np.abs(ens) >= np.quantile(np.abs(ens), 0.90)   # top-decile conviction
        hi_agree = agree[hi].mean() if hi.any() else float("nan")
        print(f"{h:>5} {agree.mean()*100:>9.1f}% {str(corrs):>22} {np.median(disp):>9.4f} "
              f"{np.median(rel):>14.2f} {str(per_seed_mag):>26} {hi_agree*100:>17.1f}%")

    # how far is each seed from the ensemble (RMSE over all nodes/rows, vol-norm units)
    ens_full = Pd.mean(0)
    rmse = [round(float(np.sqrt(np.mean((Pd[s] - ens_full) ** 2))), 4) for s in range(S)]
    print(f"\nper-seed RMSE-from-ensemble (vol-norm, all 240 nodes): "
          f"{dict(zip(seeds, rmse))}")
    print("read: dir-agree3 = % legs where ALL 3 seeds share the sign; "
          "disp/|ens| = cross-seed std relative to the signal (SNR = 1/that, roughly); "
          "HIconv = same but only top-10% |ensemble| legs.")


if __name__ == "__main__":
    main()
