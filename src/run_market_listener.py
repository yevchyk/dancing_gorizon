"""Model C — the "market listener" (general market listener).

Features are MARKET-WIDE, not per-symbol: BTC's own price curve at many lookbacks
(where the market driver is) + BTC realized volatility (how turbulent). Predicts
the SAME up/down target as the A/B models, but from market state only -> its
probability is essentially "is the market favorable for this direction right now",
orthogonal to the per-symbol-curve models. Used to GATE the A/B ensemble.

Reuses the bluechip master dataset (targets + split + anchors).

  python -m src.run_market_listener --tag listener
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .trading.timeutil import index_to_ns

NS_MIN = 60_000_000_000
BTC = "BTC_USDT_SWAP"
BTC_OFFSETS = [2, 5, 10, 20, 40, 80, 160, 320, 720, 1440, 2880, 5760, 11520, 23040, 43200]  # 2m..30d
VOL_WINDOWS = [30, 120]  # minutes
MASTER = C.DATA_DIR / "fast_bluechip" / "datasets" / "master_bluechip.parquet"
HORIZONS = ["18m", "24m", "32m", "48m", "68m", "100m"]
OUT_MODELS = C.MODELS_DIR / "fast_bluechip"
OUT_ANALYSIS = C.OUTPUTS_DIR / "analysis" / "fast_bluechip"


def market_features(anchors_ns: np.ndarray, ts: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Per-anchor BTC curve + realized vol. Shape (n, len(BTC_OFFSETS)+len(VOL_WINDOWS))."""
    n = len(anchors_ns)
    entry_idx = np.searchsorted(ts, anchors_ns, side="right") - 1
    entry_idx = np.clip(entry_idx, 0, len(close) - 1)
    entry = close[entry_idx]
    cols = []
    for off in BTC_OFFSETS:
        sidx = np.searchsorted(ts, anchors_ns - off * NS_MIN, side="right") - 1
        sidx = np.clip(sidx, 0, len(close) - 1)
        cols.append(close[sidx] / entry)
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    for w in VOL_WINDOWS:
        vcol = np.empty(n)
        for i in range(n):
            ei = int(entry_idx[i])
            lo = max(0, ei - w)
            vcol[i] = logret[lo:ei + 1].std() if ei > lo else 0.0
        cols.append(vcol)
    return np.column_stack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="listener")
    ap.add_argument("--iterations", type=int, default=300)
    args = ap.parse_args()

    # only need targets + anchors (NOT the 320 curve cols) -> avoids OOM
    needed = ["symbol", "anchor_time", "split"] + [f"ret_{h}" for h in HORIZONS]
    ds = pd.read_parquet(MASTER, columns=needed).sort_values("anchor_time").reset_index(drop=True)
    btc = CandleStore(C.DATA_DIR / "bluechip" / "candles_1m").load(BTC).sort_index()
    ts = index_to_ns(btc.index); close = btc["close"].to_numpy("float64")

    # compute BTC features per UNIQUE anchor, then map to rows
    uniq = ds["anchor_time"].drop_duplicates().reset_index(drop=True)
    uniq_ns = pd.DatetimeIndex(pd.to_datetime(uniq, utc=True)).as_unit("ns").asi8
    print(f"market features for {len(uniq)} unique anchors ({len(ds)} rows)...", flush=True)
    feats = market_features(uniq_ns, ts, close)
    fcols = [f"btc_{o}" for o in BTC_OFFSETS] + [f"vol_{w}" for w in VOL_WINDOWS]
    fmap = pd.DataFrame(feats, columns=fcols)
    fmap["anchor_time"] = uniq  # tz-aware Series, aligns by 0..k-1 index
    ds = ds.merge(fmap, on="anchor_time", how="left")

    tr = ds[ds.split == "train"]; ho = ds[ds.split == "holdout"]
    print(f"train={len(tr):,} holdout={len(ho):,} features={len(fcols)}", flush=True)
    Xtr, Xho = tr[fcols], ho[fcols]
    models_dir = OUT_MODELS / args.tag; analysis = OUT_ANALYSIS / args.tag
    models_dir.mkdir(parents=True, exist_ok=True); analysis.mkdir(parents=True, exist_ok=True)
    scores = ho[["symbol", "anchor_time"]].copy()
    scores["day"] = pd.to_datetime(ho["anchor_time"], utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    for h in HORIZONS:
        ret = tr[f"ret_{h}"].to_numpy()
        scores[f"real_ret_{h}"] = ho[f"ret_{h}"].to_numpy()
        for side in ("up", "down"):
            y = (ret > FC.TARGET_EDGE).astype(int) if side == "up" else (ret < -FC.TARGET_EDGE).astype(int)
            yh = ((ho[f"ret_{h}"].to_numpy() > FC.TARGET_EDGE) if side == "up"
                  else (ho[f"ret_{h}"].to_numpy() < -FC.TARGET_EDGE)).astype(int)
            cut = int(len(Xtr) * 0.85)
            m = CatBoostClassifier(iterations=args.iterations, learning_rate=0.03, depth=5,
                                   l2_leaf_reg=5, loss_function="Logloss", eval_metric="AUC",
                                   auto_class_weights="Balanced", random_seed=42, verbose=0,
                                   allow_writing_files=False)
            m.fit(Xtr.iloc[:cut], y[:cut], eval_set=(Xtr.iloc[cut:], y[cut:]),
                  use_best_model=True, early_stopping_rounds=50)
            joblib.dump(m, models_dir / f"{side}_{h}.joblib")
            joblib.dump(fcols, models_dir / f"{side}_{h}_columns.joblib")
            p = m.predict_proba(Xho)[:, 1]
            scores[f"p_{side}_{h}"] = p
            auc = roc_auc_score(yh, p) if len(np.unique(yh)) > 1 else float("nan")
            print(f"  {side}_{h:<4} AUC={auc:.4f} best={m.get_best_iteration()}", flush=True)
    scores.to_parquet(analysis / "holdout_scores.parquet", index=False)
    print(f"listener scores -> {analysis/'holdout_scores.parquet'}")


if __name__ == "__main__":
    main()
