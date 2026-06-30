"""Separate UP / DOWN probability models (the user's ask): per horizon train
  P_up   = P(ret_H >  fee)   profitable long
  P_down = P(ret_H < -fee)   profitable short
as two independent calibrated classifiers, walk-forward OOS. Reports calibration,
win-rate/PnL by probability, and how many predictions land in the high-confidence
zone (0.7..0.9+) -- i.e. whether a 0.85-0.9 signal even exists.

Usage:
  python -m src.run_dir_prob_walkforward --folds 4 --train-days 90 --test-days 14
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from . import config as C
from .features import CurveBuilder
from .training.horizon_slicer import HorizonSlicer

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
HZ = list(C.HORIZONS)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--test-days", type=int, default=14)
    args = ap.parse_args()

    ds = pd.read_parquet(C.DATASETS_DIR / "master_reg.parquet")
    ds["_t"] = pd.to_datetime(ds["anchor_time"], utc=True)
    slicer = HorizonSlicer(CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN))

    # fold plan
    t_min, t_max = ds["_t"].min(), ds["_t"].max()
    folds, te_end = [], t_max
    for _ in range(args.folds):
        ts = te_end - pd.Timedelta(days=args.test_days)
        tr = ts - pd.Timedelta(days=args.train_days)
        if tr < t_min:
            break
        folds.append((tr, ts, te_end)); te_end = ts
    folds = list(reversed(folds))
    print(f"dir-prob walk-forward: {len(folds)} folds")

    recs = []
    for i, (tr0, ts0, te0) in enumerate(folds):
        tr = ds[(ds._t >= tr0) & (ds._t < ts0)]
        te = ds[(ds._t >= ts0) & (ds._t < te0)]
        if len(tr) < 1500 or te.empty:
            continue
        for h in HZ:
            cols = slicer.columns_for(h)
            ret_tr = tr[f"ret_{h.label}"]
            up = _fit(tr[cols], (ret_tr > FEE).astype(int))
            dn = _fit(tr[cols], (ret_tr < -FEE).astype(int))
            recs.append(pd.DataFrame({
                "horizon": h.label, "day": te["_t"].dt.strftime("%Y-%m-%d").to_numpy(),
                "p_up": up.predict_proba(te[cols])[:, 1],
                "p_down": dn.predict_proba(te[cols])[:, 1],
                "real_ret": te[f"ret_{h.label}"].to_numpy()}))
        print(f"  fold {i}: {ts0.date()}..{te0.date()} train={len(tr)} test={len(te)}", flush=True)

    R = pd.concat(recs, ignore_index=True)
    R.to_parquet(C.OUTPUTS_DIR / "analysis" / "dir_prob_stats.parquet", index=False)

    print("\n=== CALIBRATION P_up (pred prob -> realized P(ret>fee)) ===")
    q = pd.qcut(R["p_up"], 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"q": q, "pred": R["p_up"], "real": (R["real_ret"] > FEE).astype(int)})
    for i, g in cal.groupby("q"):
        print(f"  decile {int(i)}: pred={g.pred.mean():.3f}  realized={g.real.mean():.3f}  n={len(g)}")

    print("\n=== WIN-RATE / PnL by probability (long on p_up, short on p_down) ===")
    for thr in (0.55, 0.60, 0.65, 0.70):
        lu = R["p_up"] >= thr; ld = R["p_down"] >= thr
        up_pnl = R.loc[lu, "real_ret"] - FEE
        dn_pnl = -R.loc[ld, "real_ret"] - FEE
        pnl = pd.concat([up_pnl, dn_pnl])
        if len(pnl):
            print(f"  p>={thr:.2f}: n={len(pnl):>5} win={(pnl>0).mean():.3f} "
                  f"avg_pnl={pnl.mean()*100:+.4f}%  (long={lu.sum()} short={ld.sum()})")

    print("\n=== HIGH-CONFIDENCE ZONE: how many predictions reach 0.7..0.9+ ? ===")
    tot = len(R)
    for lo, hi in [(0.70, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.01)]:
        nu = ((R.p_up >= lo) & (R.p_up < hi)).sum()
        nd = ((R.p_down >= lo) & (R.p_down < hi)).sum()
        print(f"  [{lo:.2f},{hi:.2f}): p_up={nu} ({nu/tot*100:.2f}%)  "
              f"p_down={nd} ({nd/tot*100:.2f}%)")
    print(f"  total predictions per side: {tot}")
    print(f"  max p_up={R.p_up.max():.3f}  max p_down={R.p_down.max():.3f}")


if __name__ == "__main__":
    main()
