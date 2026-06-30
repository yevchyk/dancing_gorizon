"""Build the combined engine stats table (one row per anchor x horizon) with
everything needed to decide thresholds, coin pool and sizing:
  symbol, day, horizon, p_up, p_down, pred_mae, real_ret, real_mae, fold

Walk-forward OOS (train strictly before each test slice). This is the 'score
once' table; run_engine_analysis.py then answers many questions from it.

Usage:
  python -m src.run_engine_walkforward --folds 4 --train-days 90 --test-days 14
"""

from __future__ import annotations

import argparse

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from . import config as C
from .features import CurveBuilder
from .training.horizon_slicer import HorizonSlicer

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0


def _clf(X, y):
    m = CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                           min_data_in_leaf=20, subsample=0.8, colsample_bylevel=0.5,
                           loss_function="Logloss", eval_metric="AUC",
                           auto_class_weights="Balanced", random_seed=42, verbose=0,
                           allow_writing_files=False)
    cut = int(len(X) * 0.85)
    m.fit(X.iloc[:cut], y.iloc[:cut], eval_set=(X.iloc[cut:], y.iloc[cut:]),
          use_best_model=True, early_stopping_rounds=50)
    return m


def _reg(X, y):
    m = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                          min_data_in_leaf=20, subsample=0.8, colsample_bylevel=0.5,
                          loss_function="RMSE", random_seed=42, verbose=0,
                          allow_writing_files=False)
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

    t_min, t_max = ds["_t"].min(), ds["_t"].max()
    folds, te_end = [], t_max
    for _ in range(args.folds):
        ts = te_end - pd.Timedelta(days=args.test_days)
        tr = ts - pd.Timedelta(days=args.train_days)
        if tr < t_min:
            break
        folds.append((tr, ts, te_end)); te_end = ts
    folds = list(reversed(folds))
    print(f"engine walk-forward: {len(folds)} folds")

    recs = []
    for i, (tr0, ts0, te0) in enumerate(folds):
        tr = ds[(ds._t >= tr0) & (ds._t < ts0)]
        te = ds[(ds._t >= ts0) & (ds._t < te0)]
        if len(tr) < 1500 or te.empty:
            continue
        for h in C.HORIZONS:
            cols = slicer.columns_for(h)
            ret = tr[f"ret_{h.label}"]
            p_up = _clf(tr[cols], (ret > FEE).astype(int)).predict_proba(te[cols])[:, 1]
            p_dn = _clf(tr[cols], (ret < -FEE).astype(int)).predict_proba(te[cols])[:, 1]
            pred_ret = _reg(tr[cols], tr[f"ret_{h.label}"]).predict(te[cols])
            pred_mfe = _reg(tr[cols], tr[f"mfe_{h.label}"]).predict(te[cols])
            pred_mae = _reg(tr[cols], tr[f"mae_{h.label}"]).predict(te[cols])
            recs.append(pd.DataFrame({
                "fold": i, "symbol": te["symbol"].to_numpy(),
                "day": te["_t"].dt.strftime("%Y-%m-%d").to_numpy(),
                "horizon": h.label, "p_up": p_up, "p_down": p_dn,
                "pred_ret": pred_ret, "pred_mfe": pred_mfe, "pred_mae": pred_mae,
                "real_ret": te[f"ret_{h.label}"].to_numpy(),
                "real_mfe": te[f"mfe_{h.label}"].to_numpy(),
                "real_mae": te[f"mae_{h.label}"].to_numpy()}))
        print(f"  fold {i}: {ts0.date()}..{te0.date()} train={len(tr)} test={len(te)}", flush=True)

    out = C.OUTPUTS_DIR / "analysis" / "engine_stats.parquet"
    pd.concat(recs, ignore_index=True).to_parquet(out, index=False)
    print(f"engine stats -> {out}")


if __name__ == "__main__":
    main()
