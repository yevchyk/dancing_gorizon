"""fast_v3 — new recipe, trained on the existing fast_v2 crypto dataset.

Recipe (see docs/FAST_V3_MARKETS.md):
  * horizons 1/2/4/8/12/20m
  * up/down classifiers ONLY (no ret/mfe/mae regressors)
  * plus-only profit weighting per side:
        up_h   sample_weight = max(ret, 0)   (floor for the rest)
        down_h sample_weight = max(-ret, 0)
  * no time-of-day feature
  * holdout = last 24h, train = everything before (with 20m embargo)
  * reuses fast_v2's curve features + anchors; only the targets are recomputed.

Stages:  build (reuse features + new targets) -> train (12 clf) -> eval (holdout)

Run:  python -m src.run_fast_v3 --stage all
fast_v2 is never touched; everything lands under data/fast_v3 + models/fast_v3.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config as C
from .database import OKXClient
from .fast import config as FC
from .fast.candles import ensure_1m, load_1m
from .fast.curve import FastCurve

NS_PER_MIN = 60_000_000_000
WEIGHT_FLOOR = 0.15  # non-profit-side samples keep a small flat weight (validated in demo)

# (minutes, label, lookback_min) — lookbacks follow fast_v2's scheme (2m->24h, 8m->30d match).
HORIZONS_V3 = (
    (1, "1m", 12 * 60),
    (2, "2m", 24 * 60),
    (4, "4m", 3 * 24 * 60),
    (8, "8m", 30 * 24 * 60),
    (12, "12m", 45 * 24 * 60),
    (20, "20m", 60 * 24 * 60),
)
MAX_HORIZON = max(m for m, _, _ in HORIZONS_V3)

V3_DIR = C.DATA_DIR / "fast_v3"
V3_DATASET = V3_DIR / "datasets" / "master.parquet"
V3_MODELS = C.MODELS_DIR / "fast_v3" / "base"
V3_ANALYSIS = C.OUTPUTS_DIR / "analysis" / "fast_v3"


def _ensure_dirs() -> None:
    for p in (V3_DATASET.parent, V3_MODELS, V3_ANALYSIS):
        p.mkdir(parents=True, exist_ok=True)


def _targets(ts_ns, high, low, close, anchors_ns) -> dict[str, np.ndarray]:
    n = len(anchors_ns)
    out: dict[str, np.ndarray] = {}
    for _, lab, _ in HORIZONS_V3:
        for k in ("ret", "mfe", "mae"):
            out[f"{k}_{lab}"] = np.full(n, np.nan, dtype="float32")
    entry_idx = np.searchsorted(ts_ns, anchors_ns, side="right") - 1
    for i in range(n):
        ei = int(entry_idx[i])
        if ei < 0:
            continue
        entry = close[ei]
        if not np.isfinite(entry) or entry <= 0:
            continue
        a = anchors_ns[i]
        for m, lab, _ in HORIZONS_V3:
            fj = int(np.searchsorted(ts_ns, a + m * NS_PER_MIN, side="right"))
            if fj <= ei + 1:
                continue
            hh = high[ei + 1:fj]
            ll = low[ei + 1:fj]
            out[f"ret_{lab}"][i] = close[fj - 1] / entry - 1.0
            out[f"mfe_{lab}"][i] = hh.max() / entry - 1.0
            out[f"mae_{lab}"][i] = ll.min() / entry - 1.0
    return out


def stage_refetch(workers: int) -> None:
    """The fast_v1 1m target store rolls forward; re-pull 1m over the anchor window
    so every symbol/horizon target can be recomputed cleanly (features are untouched)."""
    src = pd.read_parquet(FC.FAST_DATASETS_DIR / "master.parquet")
    syms = sorted(src["symbol"].unique())
    at = pd.to_datetime(src["anchor_time"], utc=True)
    start = at.min().floor("1D") - pd.Timedelta(hours=1)
    end = at.max() + pd.Timedelta(minutes=MAX_HORIZON + 5)
    print(f"refetch 1m: {len(syms)} symbols  {start} -> {end}  workers={workers}")

    def work(sym):
        try:
            r = ensure_1m(sym, start, end, client=OKXClient(timeout=25.0))
            c = load_1m(sym)
            cov = 0.0 if c is None or c.empty else (c.index.max() - c.index.min()).total_seconds() / 86400
            return sym, str(r.get("status")), cov
        except Exception as exc:
            return sym, f"FAIL {type(exc).__name__}", 0.0

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, s) for s in syms]
        for i, f in enumerate(as_completed(futs), 1):
            sym, status, cov = f.result()
            ok += int(status == "ok")
            fail += int(status != "ok")
            if i % 20 == 0 or i == len(syms):
                print(f"  {i}/{len(syms)} ok={ok} fail={fail} last={sym} {status} cov={cov:.1f}d", flush=True)
    print(f"refetch done: ok={ok} fail={fail}")


def stage_build() -> None:
    _ensure_dirs()
    src = pd.read_parquet(FC.FAST_DATASETS_DIR / "master.parquet")
    curve_cols = [c for c in src.columns if c.startswith("p_")]
    print(f"source fast_v2 master: {len(src)} rows, {len(curve_cols)} curve cols, "
          f"{src.symbol.nunique()} symbols")

    frames = []
    syms = list(src.groupby("symbol", sort=False))
    for i, (sym, g) in enumerate(syms, 1):
        candles = load_1m(sym)
        if candles is None or candles.empty:
            continue
        candles = candles.sort_index()
        ts_ns = candles.index.as_unit("ns").asi8
        high = candles["high"].to_numpy("float64")
        low = candles["low"].to_numpy("float64")
        close = candles["close"].to_numpy("float64")
        anchors = pd.DatetimeIndex(pd.to_datetime(g["anchor_time"], utc=True))
        anchors_ns = anchors.as_unit("ns").asi8
        tg = _targets(ts_ns, high, low, close, anchors_ns)
        out = g[["symbol", "anchor_time"] + curve_cols].copy()
        for k, v in tg.items():
            out[k] = v
        frames.append(out)
        if i % 25 == 0 or i == len(syms):
            print(f"  targets {i}/{len(syms)} last={sym}", flush=True)

    ds = pd.concat(frames, ignore_index=True)
    at = pd.to_datetime(ds["anchor_time"], utc=True)
    tmax = at.max()
    hold_start = tmax - pd.Timedelta(hours=24)
    emb = pd.Timedelta(minutes=MAX_HORIZON)
    ds["split"] = np.where(at >= hold_start, "holdout",
                           np.where(at < hold_start - emb, "train", "embargo"))
    ds = ds[ds.split != "embargo"].copy()

    tcols = [f"{k}_{lab}" for _, lab, _ in HORIZONS_V3 for k in ("ret", "mfe", "mae")]
    before = len(ds)
    ds = ds.dropna(subset=tcols)
    ds.to_parquet(V3_DATASET, index=False)
    print(f"fast_v3 dataset: {len(ds)} rows (dropped {before-len(ds)} NaN), "
          f"train={int((ds.split=='train').sum())} holdout={int((ds.split=='holdout').sum())}")
    print(f"  holdout window: {hold_start} -> {tmax}  (embargo {MAX_HORIZON}m)")
    print(f"  -> {V3_DATASET}")


def _fit(X, y, w, iterations, depth):
    cut = int(len(X) * 0.85)
    m = CatBoostClassifier(
        iterations=iterations, learning_rate=0.03, depth=depth, l2_leaf_reg=5,
        min_data_in_leaf=20, subsample=0.8, colsample_bylevel=0.5,
        loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
        random_seed=42, verbose=0, allow_writing_files=False,
    )
    m.fit(X.iloc[:cut], y[:cut], sample_weight=w[:cut],
          eval_set=(X.iloc[cut:], y[cut:]), use_best_model=True, early_stopping_rounds=50)
    return m


def stage_train(iterations: int, depth: int) -> None:
    _ensure_dirs()
    ds = pd.read_parquet(V3_DATASET).sort_values("anchor_time").reset_index(drop=True)
    tr = ds[ds.split == "train"]
    ho = ds[ds.split == "holdout"]
    curve = FastCurve(FC.CURVE_POINTS, FC.CURVE_MIN_STEP_MIN, FC.CURVE_MAX_DEPTH_MIN, FC.CURVE_SEGMENTS)
    print(f"train={len(tr)} holdout={len(ho)} iterations={iterations} fee={FC.TARGET_EDGE:.4f}")

    scores = ho[["symbol", "anchor_time"]].copy()
    scores["day"] = pd.to_datetime(ho["anchor_time"], utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    metrics = []
    for m, lab, lb in HORIZONS_V3:
        cols = curve.columns_for_lookback(lb)
        Xtr, Xho = tr[cols], ho[cols]
        ret = tr[f"ret_{lab}"].to_numpy()
        scores[f"real_ret_{lab}"] = ho[f"ret_{lab}"].to_numpy()
        scores[f"real_mfe_{lab}"] = ho[f"mfe_{lab}"].to_numpy()
        scores[f"real_mae_{lab}"] = ho[f"mae_{lab}"].to_numpy()
        y_ho_up = (ho[f"ret_{lab}"].to_numpy() > FC.TARGET_EDGE).astype(int)
        y_ho_dn = (ho[f"ret_{lab}"].to_numpy() < -FC.TARGET_EDGE).astype(int)
        for side in ("up", "down"):
            if side == "up":
                y = (ret > FC.TARGET_EDGE).astype(int)
                prof = np.maximum(ret, 0.0)
                y_ho = y_ho_up
            else:
                y = (ret < -FC.TARGET_EDGE).astype(int)
                prof = np.maximum(-ret, 0.0)
                y_ho = y_ho_dn
            pm = prof[prof > 0].mean() if (prof > 0).any() else 1.0
            w = np.where(prof > 0, prof / pm, WEIGHT_FLOOR).astype("float64")
            model = _fit(Xtr, y, w, iterations, depth)
            joblib.dump(model, V3_MODELS / f"{side}_{lab}.joblib")
            joblib.dump(cols, V3_MODELS / f"{side}_{lab}_columns.joblib")
            p = model.predict_proba(Xho)[:, 1]
            scores[f"p_{side}_{lab}"] = p
            auc = roc_auc_score(y_ho, p) if len(np.unique(y_ho)) > 1 else float("nan")
            metrics.append({"name": f"{side}_{lab}", "auc": float(auc),
                            "pos_rate": float(y.mean()), "best_iter": int(model.get_best_iteration() or 0),
                            "n_features": len(cols)})
            print(f"  {side}_{lab:<4} AUC={auc:.4f} pos={y.mean():.3f} best={model.get_best_iteration()}")
    (V3_MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scores.to_parquet(V3_ANALYSIS / "holdout_scores.parquet", index=False)
    print(f"holdout scores -> {V3_ANALYSIS/'holdout_scores.parquet'}")


def stage_eval(k: int) -> None:
    s = pd.read_parquet(V3_ANALYSIS / "holdout_scores.parquet")
    cost = FC.EVAL_COST
    print(f"\nfast_v3 holdout: top {k}/day, fee={cost*100:.2f}%  "
          f"({s['day'].nunique() if 'day' in s else '?'} day(s))\n")
    print(f"{'model':<9}{'AUC':>7}{'n':>6}{'win':>7}{'avg%':>9}{'TP(avg+)':>10}{'SL(avg-)':>10}{'total%':>9}")
    rows = []
    metrics = {m["name"]: m for m in json.loads((V3_MODELS / "metrics.json").read_text())}
    for m, lab, lb in HORIZONS_V3:
        for side in ("up", "down"):
            name = f"{side}_{lab}"
            p = s[f"p_{side}_{lab}"].to_numpy()
            real = s[f"real_ret_{lab}"].to_numpy()
            sign = 1.0 if side == "up" else -1.0
            df = pd.DataFrame({"p": p, "real": real, "day": s["day"],
                               "mfe": s[f"real_mfe_{lab}"].to_numpy(),
                               "mae": s[f"real_mae_{lab}"].to_numpy()})
            df["rk"] = df.groupby("day")["p"].rank(ascending=False, method="first")
            d = df[df.rk <= k]
            if len(d) == 0:
                continue
            pnl = sign * d["real"].to_numpy() - cost
            # exit calibration: favorable/adverse excursion in the model's own direction
            fav = d["mfe"].mean() if side == "up" else -d["mae"].mean()
            adv = d["mae"].mean() if side == "up" else -d["mfe"].mean()
            r = {"model": name, "auc": metrics[name]["auc"], "n": len(d),
                 "win": float((pnl > 0).mean()), "avg%": float(pnl.mean() * 100),
                 "tp%": float(fav * 100), "sl%": float(adv * 100), "total%": float(pnl.sum() * 100)}
            rows.append(r)
            print(f"{name:<9}{r['auc']:>7.3f}{r['n']:>6}{r['win']:>7.3f}{r['avg%']:>+9.4f}"
                  f"{r['tp%']:>+10.4f}{r['sl%']:>+10.4f}{r['total%']:>+9.2f}")
    pd.DataFrame(rows).to_csv(V3_ANALYSIS / "holdout_eval.csv", index=False)
    print(f"\neval -> {V3_ANALYSIS/'holdout_eval.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["refetch", "build", "train", "eval", "all"], default="all")
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.stage in ("refetch", "all"):
        stage_refetch(args.workers)
    if args.stage in ("build", "all"):
        stage_build()
    if args.stage in ("train", "all"):
        stage_train(args.iterations, args.depth)
    if args.stage in ("eval", "all"):
        stage_eval(args.topk)


if __name__ == "__main__":
    main()
