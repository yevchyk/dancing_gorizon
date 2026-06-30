"""Strong market-regime detector ("посос" detector). Unlike model C (which predicted
per-symbol up/down -> AUC 0.51), this predicts the MARKET AGGREGATE: will the median
forward return of the whole universe be in the worst quintile (danger). Clean target.

Features = BTC multi-scale curve + BTC realized vol (market state). One sample per
anchor timestamp. Train on master train-split, test on holdout.

  python -m src.run_crash_detector --horizon 32m
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from . import config as C
from .database import CandleStore
from .trading.timeutil import index_to_ns
from .run_market_listener import market_features, BTC_OFFSETS, VOL_WINDOWS

sys.stdout.reconfigure(encoding="utf-8")
NS_MIN = 60_000_000_000
MASTER = C.DATA_DIR / "fast_bluechip" / "datasets" / "master_bluechip.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="32m")
    ap.add_argument("--danger-q", type=float, default=0.20, help="bottom-q of market move = danger")
    args = ap.parse_args()
    h = args.horizon

    ds = pd.read_parquet(MASTER, columns=["symbol", "anchor_time", "split", f"ret_{h}"])
    # market aggregate per anchor = median forward return across the universe
    agg = ds.groupby("anchor_time").agg(mkt=(f"ret_{h}", "median"),
                                        split=("split", "first")).reset_index()
    thr = agg["mkt"].quantile(args.danger_q)
    agg["danger"] = (agg["mkt"] <= thr).astype(int)
    print(f"anchors={len(agg)}  danger threshold (mkt median ret <= {thr*100:+.3f}%)  "
          f"danger rate={agg['danger'].mean():.2f}")

    # BTC features per anchor
    btc = CandleStore(C.DATA_DIR / "bluechip" / "candles_1m").load("BTC_USDT_SWAP").sort_index()
    ts = index_to_ns(btc.index); close = btc["close"].to_numpy("float64")
    a_ns = pd.DatetimeIndex(pd.to_datetime(agg["anchor_time"], utc=True)).as_unit("ns").asi8
    feats = market_features(a_ns, ts, close)
    fcols = [f"btc_{o}" for o in BTC_OFFSETS] + [f"vol_{w}" for w in VOL_WINDOWS]
    F = pd.DataFrame(feats, columns=fcols)
    F["split"] = agg["split"].values; F["danger"] = agg["danger"].values
    F["day"] = pd.to_datetime(agg["anchor_time"], utc=True).dt.strftime("%Y-%m-%d").values
    F["mkt"] = agg["mkt"].values

    tr = F[F.split == "train"]; ho = F[F.split == "holdout"]
    m = CatBoostClassifier(iterations=400, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                           loss_function="Logloss", eval_metric="AUC",
                           auto_class_weights="Balanced", random_seed=42, verbose=0,
                           allow_writing_files=False)
    cut = int(len(tr) * 0.85)
    m.fit(tr[fcols].iloc[:cut], tr["danger"].to_numpy()[:cut],
          eval_set=(tr[fcols].iloc[cut:], tr["danger"].to_numpy()[cut:]),
          use_best_model=True, early_stopping_rounds=50)
    p = m.predict_proba(ho[fcols])[:, 1]
    auc = roc_auc_score(ho["danger"].to_numpy(), p)
    print(f"\nDANGER detector AUC (holdout) = {auc:.4f}   (old per-symbol listener was 0.51)")

    # does p_danger spike on the real crash days?
    print(f"\n{'day':<12}{'p_danger(avg)':>14}{'mkt_ret%(avg)':>14}{'danger_rate':>12}")
    hh = ho.copy(); hh["p"] = p
    for d, g in hh.groupby("day"):
        print(f"{d:<12}{g['p'].mean():>14.3f}{g['mkt'].mean()*100:>+14.4f}{g['danger'].mean():>12.3f}")

    # save per-anchor p_danger for engine wiring
    out = C.OUTPUTS_DIR / "analysis" / "fast_bluechip" / "danger_holdout.parquet"
    pd.DataFrame({"anchor_time": agg[agg.split == "holdout"]["anchor_time"].values,
                  "p_danger": p}).to_parquet(out, index=False)
    print(f"\np_danger -> {out}")


if __name__ == "__main__":
    main()
