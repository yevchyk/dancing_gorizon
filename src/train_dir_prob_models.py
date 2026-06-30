"""Production training of the separate UP / DOWN probability models.

Per horizon, two independent calibrated classifiers:
  p_up_{H}   = P(ret_H >  fee)   (profitable long)
  p_down_{H} = P(ret_H < -fee)   (profitable short)

Trained on everything EXCEPT the last HOLDOUT_DAYS (so the holdout is unseen),
saved to models/dir_prob/, and evaluated on the holdout.

Usage:
  python -m src.train_dir_prob_models
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from . import config as C
from .features import CurveBuilder
from .training.horizon_slicer import HorizonSlicer

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
DIR_PROB_DIR = C.MODELS_DIR / "dir_prob"


def _fit(X, y):
    m = CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6,
                           l2_leaf_reg=5, min_data_in_leaf=20, subsample=0.8,
                           colsample_bylevel=0.5, loss_function="Logloss",
                           eval_metric="AUC", auto_class_weights="Balanced",
                           random_seed=42, verbose=0, allow_writing_files=False)
    cut = int(len(X) * 0.85)
    m.fit(X.iloc[:cut], y.iloc[:cut], eval_set=(X.iloc[cut:], y.iloc[cut:]),
          use_best_model=True, early_stopping_rounds=50)
    return m


def main() -> None:
    ds = pd.read_parquet(C.DATASETS_DIR / "master_reg.parquet")
    t = pd.to_datetime(ds["anchor_time"], utc=True)
    cutoff = t.max() - pd.Timedelta(days=C.HOLDOUT_DAYS)
    train, hold = ds[t < cutoff], ds[t >= cutoff]
    slicer = HorizonSlicer(CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN))
    DIR_PROB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"train={len(train)} holdout={len(hold)} (cutoff {cutoff.date()})")

    holdpred = {}
    print("\n=== TRAIN (val AUC) ===")
    for h in C.HORIZONS:
        cols = slicer.columns_for(h)
        ret = train[f"ret_{h.label}"]
        for kind, y in (("up", (ret > FEE).astype(int)), ("down", (ret < -FEE).astype(int))):
            m = _fit(train[cols], y)
            name = f"{kind}_{h.label}"
            joblib.dump(m, DIR_PROB_DIR / f"{name}.joblib")
            joblib.dump(cols, DIR_PROB_DIR / f"{name}_columns.joblib")
            holdpred[name] = m.predict_proba(hold[cols])[:, 1]
            print(f"  {name:<8} pos_rate={y.mean():.3f}")

    print("\n=== HOLDOUT calibration + zone count ===")
    real = {h.label: hold[f"ret_{h.label}"].to_numpy() for h in C.HORIZONS}
    n = len(hold)
    for h in C.HORIZONS:
        lab = h.label
        pu, pd_ = holdpred[f"up_{lab}"], holdpred[f"down_{lab}"]
        up_hit = (real[lab] > FEE)
        # win-rate when prob high
        for thr in (0.6, 0.7):
            lu = pu >= thr
            if lu.sum() >= 10:
                wr = up_hit[lu].mean()
                pnl = (real[lab][lu] - FEE).mean() * 100
                hi = ((pu >= 0.85) & (pu < 0.95)).sum()
                print(f"  up_{lab} p>={thr}: n={int(lu.sum())} win={wr:.3f} pnl={pnl:+.3f}%  "
                      f"| in[0.85,0.95)={hi}")
                break
    print(f"\n  max p_up across holdout: "
          f"{max(holdpred[f'up_{h.label}'].max() for h in C.HORIZONS):.3f}")
    print(f"models -> {DIR_PROB_DIR}")


if __name__ == "__main__":
    main()
