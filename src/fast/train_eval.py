"""Train/evaluate fast_v1 base models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from . import config as FC
from .curve import FastCurve


def columns_for(horizon) -> list[str]:
    curve = FastCurve(
        FC.CURVE_POINTS,
        FC.CURVE_MIN_STEP_MIN,
        FC.CURVE_MAX_DEPTH_MIN,
        FC.CURVE_SEGMENTS,
    )
    cols = curve.columns_for_lookback(horizon.lookback_min)
    if FC.BTC_CONTEXT:
        # BTC market context is relevant at every horizon, so every model sees it.
        cols = cols + FC.btc_columns()
    return cols


def _split_fit(X: pd.DataFrame, y: pd.Series, kind: str, iterations: int, depth: int):
    cut = int(len(X) * 0.85)
    X_tr, X_val = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_val = y.iloc[:cut], y.iloc[cut:]
    if kind == "clf":
        model = CatBoostClassifier(
            iterations=iterations, learning_rate=0.03, depth=depth,
            l2_leaf_reg=5, min_data_in_leaf=20, subsample=0.8,
            colsample_bylevel=0.5, loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", random_seed=42, verbose=0,
            allow_writing_files=False,
        )
    else:
        model = CatBoostRegressor(
            iterations=iterations, learning_rate=0.03, depth=depth,
            l2_leaf_reg=5, min_data_in_leaf=20, subsample=0.8,
            colsample_bylevel=0.5, loss_function="RMSE", random_seed=42,
            verbose=0, allow_writing_files=False,
        )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True,
              early_stopping_rounds=50)
    pred = model.predict_proba(X_val)[:, 1] if kind == "clf" else model.predict(X_val)
    metric = {}
    if kind == "clf":
        metric["pos_rate"] = float(y.mean())
    else:
        metric["rmse"] = float(np.sqrt(np.mean((pred - y_val.to_numpy()) ** 2)))
        metric["corr"] = float(np.corrcoef(pred, y_val.to_numpy())[0, 1]) if y_val.nunique() > 1 else 0.0
    metric["n_train"] = int(len(X_tr))
    metric["n_val"] = int(len(X_val))
    metric["best_iter"] = int(model.get_best_iteration() or 0)
    return model, metric


def train_and_score(iterations: int = 450, depth: int = 6) -> Path:
    FC.ensure_dirs()
    ds = pd.read_parquet(FC.FAST_DATASETS_DIR / "master.parquet")
    ds = ds.sort_values("anchor_time").reset_index(drop=True)
    train = ds[ds["split"] == "train"].copy()
    hold = ds[ds["split"] == "holdout"].copy()
    print(f"fast train={len(train)} holdout={len(hold)} iterations={iterations} depth={depth}")

    scores_base = hold[["symbol", "anchor_time"]].copy()
    metrics: list[dict] = []
    for h in FC.HORIZONS:
        cols = columns_for(h)
        Xtr = train[cols]
        Xh = hold[cols]
        ret = train[f"ret_{h.label}"]
        real_ret = hold[f"ret_{h.label}"].to_numpy()
        scores_base[f"real_ret_{h.label}"] = real_ret
        scores_base[f"real_mfe_{h.label}"] = hold[f"mfe_{h.label}"].to_numpy()
        scores_base[f"real_mae_{h.label}"] = hold[f"mae_{h.label}"].to_numpy()

        for name, y in (
            (f"up_{h.label}", (ret > FC.TARGET_EDGE).astype(int)),
            (f"down_{h.label}", (ret < -FC.TARGET_EDGE).astype(int)),
        ):
            model, metric = _split_fit(Xtr, y, "clf", iterations, depth)
            joblib.dump(model, FC.FAST_MODELS_DIR / f"{name}.joblib")
            joblib.dump(cols, FC.FAST_MODELS_DIR / f"{name}_columns.joblib")
            scores_base[f"p_{name}"] = model.predict_proba(Xh)[:, 1]
            metric.update({"name": name, "type": "clf", "horizon": h.label, "n_features": len(cols)})
            metrics.append(metric)
            print(f"  {name:<8} pos={metric['pos_rate']:.3f} best={metric['best_iter']}")

        for kind in ("ret", "mfe", "mae"):
            name = f"{kind}_{h.label}"
            model, metric = _split_fit(Xtr, train[name].astype(float), "reg", iterations, depth)
            joblib.dump(model, FC.FAST_MODELS_DIR / f"{name}.joblib")
            joblib.dump(cols, FC.FAST_MODELS_DIR / f"{name}_columns.joblib")
            scores_base[f"pred_{name}"] = model.predict(Xh)
            metric.update({"name": name, "type": "reg", "horizon": h.label, "n_features": len(cols)})
            metrics.append(metric)
            print(f"  {name:<8} corr={metric['corr']:+.3f} rmse={metric['rmse']:.5f} best={metric['best_iter']}")

    (FC.FAST_MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows = []
    for h in FC.HORIZONS:
        rows.append(pd.DataFrame({
            "symbol": scores_base["symbol"].to_numpy(),
            "anchor_time": scores_base["anchor_time"].to_numpy(),
            "day": pd.to_datetime(scores_base["anchor_time"], utc=True).dt.strftime("%Y-%m-%d").to_numpy(),
            "horizon": h.label,
            "p_up": scores_base[f"p_up_{h.label}"].to_numpy(),
            "p_down": scores_base[f"p_down_{h.label}"].to_numpy(),
            "pred_ret": scores_base[f"pred_ret_{h.label}"].to_numpy(),
            "pred_mfe": scores_base[f"pred_mfe_{h.label}"].to_numpy(),
            "pred_mae": scores_base[f"pred_mae_{h.label}"].to_numpy(),
            "real_ret": scores_base[f"real_ret_{h.label}"].to_numpy(),
            "real_mfe": scores_base[f"real_mfe_{h.label}"].to_numpy(),
            "real_mae": scores_base[f"real_mae_{h.label}"].to_numpy(),
        }))
    scored = pd.concat(rows, ignore_index=True)
    out = FC.FAST_ANALYSIS_DIR / "holdout_scores.parquet"
    scored.to_parquet(out, index=False)
    print(f"holdout scores -> {out}")
    return out


def _eval(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0}
    pnl = df["pnl"].to_numpy()
    daily = df.groupby("day")["pnl"].mean() * 100
    return {
        "n": int(len(df)),
        "win": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean() * 100),
        "green": int((daily > 0).sum()),
        "days": int(len(daily)),
        "total": float(pnl.sum() * 100),
    }


def compare() -> pd.DataFrame:
    s = pd.read_parquet(FC.FAST_ANALYSIS_DIR / "holdout_scores.parquet")
    s["conf"] = np.maximum(s["p_up"], s["p_down"])
    s["side"] = np.where(s["p_up"] >= s["p_down"], 1, -1)
    s["opp"] = np.minimum(s["p_up"], s["p_down"])
    s["spread"] = s["conf"] - s["opp"]
    fav = np.where(s["side"] == 1, s["pred_mfe"], -s["pred_mae"])
    adv = np.where(s["side"] == 1, np.abs(s["pred_mae"]), s["pred_mfe"])
    s["rr"] = np.clip(fav / (np.abs(adv) + 1e-4), 0, 5)
    s["pnl"] = s["side"] * s["real_ret"] - FC.EVAL_COST

    results = []
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        d = s[s.conf >= thr].copy()
        r = _eval(d)
        r.update({"strategy": f"conf>={thr:.2f}"})
        results.append(r)

    for k in (20, 50, 100, 200):
        d = s.copy()
        d["_rk"] = d.groupby("day")["conf"].rank(ascending=False, method="first")
        r = _eval(d[d._rk <= k].copy())
        r.update({"strategy": f"top{k}/day conf"})
        results.append(r)

        d = s.copy()
        d["_score"] = d["conf"] * d["rr"]
        d["_rk"] = d.groupby("day")["_score"].rank(ascending=False, method="first")
        r = _eval(d[d._rk <= k].copy())
        r.update({"strategy": f"top{k}/day riskadj"})
        results.append(r)

    for floor in (0.60, 0.70, 0.80):
        fire = s[(s.conf >= floor) & (s.opp <= 0.30)].copy()
        grp = fire.groupby(["symbol", "anchor_time", "side"]).size().rename("agree").reset_index()
        agree = grp[grp.agree >= 2][["symbol", "anchor_time", "side"]]
        d = fire.merge(agree, on=["symbol", "anchor_time", "side"], how="inner")
        d["_rk"] = d.groupby("day")["spread"].rank(ascending=False, method="first")
        for k in (20, 50, 100):
            r = _eval(d[d._rk <= k].copy())
            r.update({"strategy": f"clean{floor:.2f} agree2 top{k}/day"})
            results.append(r)

    out = pd.DataFrame(results)
    out = out[["strategy", "n", "win", "avg_pnl", "green", "days", "total"]]
    path = FC.FAST_ANALYSIS_DIR / "strategy_compare.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False, formatters={
        "win": "{:.3f}".format,
        "avg_pnl": "{:+.4f}".format,
        "total": "{:+.2f}".format,
    }))
    print(f"strategy compare -> {path}")
    return out
