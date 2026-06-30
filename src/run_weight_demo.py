"""One-off demo: does profit-weighting the training samples capture more PnL?

Trains three up_5m classifiers on the existing fast_v2 dataset, identical in
every way to production EXCEPT the per-sample weight:
  baseline : equal weights (current production)
  w_abs    : sample_weight = |ret|        (big movers matter, both sides)
  w_plus   : sample_weight = max(ret, 0)  (value the profit side only)

Then ranks the untouched holdout by predicted P(up), takes top-K/day, and
reports realized PnL + AUC so the win and its cost are both visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from .fast import config as FC
from .fast.train_eval import columns_for

H = "5m"
ITERS = 450
DEPTH = 6


def fit(Xtr, y, w):
    cut = int(len(Xtr) * 0.85)
    m = CatBoostClassifier(
        iterations=ITERS, learning_rate=0.03, depth=DEPTH, l2_leaf_reg=5,
        min_data_in_leaf=20, subsample=0.8, colsample_bylevel=0.5,
        loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
        random_seed=42, verbose=0, allow_writing_files=False,
    )
    kw = {}
    if w is not None:
        kw["sample_weight"] = w[:cut]
    m.fit(Xtr.iloc[:cut], y[:cut], eval_set=(Xtr.iloc[cut:], y[cut:]),
          use_best_model=True, early_stopping_rounds=50, **kw)
    return m


def topk(p, real, day, k):
    df = pd.DataFrame({"p": p, "real": real, "day": day})
    df["rk"] = df.groupby("day")["p"].rank(ascending=False, method="first")
    d = df[df.rk <= k]
    pnl = d["real"].to_numpy() - FC.EVAL_COST
    return {
        "win": float((pnl > 0).mean()),
        "avg_pnl%": float(pnl.mean() * 100),
        "mean_move%": float(d["real"].mean() * 100),
        "total%": float(pnl.sum() * 100),
        "n": int(len(d)),
    }


def main() -> None:
    ds = pd.read_parquet(FC.FAST_DATASETS_DIR / "master.parquet")
    ds = ds.sort_values("anchor_time").reset_index(drop=True)
    tr = ds[ds.split == "train"]
    ho = ds[ds.split == "holdout"]
    cols = columns_for([h for h in FC.HORIZONS if h.label == H][0])

    Xtr, Xho = tr[cols], ho[cols]
    ret_tr = tr[f"ret_{H}"].to_numpy()
    y = (ret_tr > FC.TARGET_EDGE).astype(int)
    real = ho[f"ret_{H}"].to_numpy()
    day = pd.to_datetime(ho.anchor_time, utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    y_ho = (real > FC.TARGET_EDGE).astype(int)

    mean_abs = np.abs(ret_tr).mean()
    w_abs = np.abs(ret_tr) / mean_abs
    pos = np.maximum(ret_tr, 0.0)
    w_plus = np.where(pos > 0, pos / pos[pos > 0].mean(), 0.15)

    variants = {
        "baseline": None,
        "w=|ret|": w_abs,
        "w=max(ret,0)": w_plus,
    }

    print(f"up_{H}: train={len(tr)} holdout={len(ho)} cols={len(cols)} "
          f"pos_rate={y.mean():.3f} mean|ret|={mean_abs*100:.3f}%\n")

    preds = {}
    for name, w in variants.items():
        m = fit(Xtr, y, w)
        p = m.predict_proba(Xho)[:, 1]
        preds[name] = p
        auc = roc_auc_score(y_ho, p)
        print(f"[{name:<13}] trained  best_iter={m.get_best_iteration()}  holdout_AUC={auc:.4f}")

    for k in (20, 50, 100):
        print(f"\n=== top {k}/day by P(up)   (4 holdout days, fee={FC.EVAL_COST*100:.2f}%) ===")
        print(f"{'variant':<14}{'n':>6}{'win':>8}{'avg_pnl%':>10}{'mean_move%':>12}{'total%':>10}")
        for name in variants:
            r = topk(preds[name], real, day, k)
            print(f"{name:<14}{r['n']:>6}{r['win']:>8.3f}{r['avg_pnl%']:>+10.4f}"
                  f"{r['mean_move%']:>+12.4f}{r['total%']:>+10.2f}")


if __name__ == "__main__":
    main()
