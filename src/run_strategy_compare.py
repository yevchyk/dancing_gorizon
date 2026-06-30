"""Compare ALL squeeze ideas on the SAME out-of-fold table, on the most recent
fold (fold 3 = ~last 14d) and the strict last-10-days subset:

  0 baseline   : engine argmax(p_up,p_down), top by conf
  1 stacking   : meta-model (trained on folds 0..n-1) predicts trade pnl, top by it
  2 risk-adj   : argmax side, top by  p_dir * MFE/|MAE|  (uses the strong excursion models)
  3 ranking    : cross-sectional top-K per day by conf (regime-robust, small pool)
  4 calib+size : isotonic-calibrate conf on earlier folds, size by edge (sizing, not selection)
  5 horiz-agree: take a side only if >=2 horizons agree (conf>=0.65)

Score-based strategies are compared at a MATCHED trade count (= baseline's), so
win/pnl are apples-to-apples. Cost = fee + slippage.

Usage:
  python -m src.run_strategy_compare --slip 0.05 --base-thr 0.75
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
HMIN = {h.label: h.minutes for h in C.HORIZONS}


def evaluate(df: pd.DataFrame, cost: float) -> dict:
    pnl = df["side"].to_numpy() * df["real_ret"].to_numpy() - cost
    w = df["w"].to_numpy() if "w" in df else np.ones(len(pnl))
    daily = pd.DataFrame({"day": df["day"].to_numpy(), "pnl": pnl, "w": w})
    dg = daily.groupby("day").apply(lambda g: np.average(g.pnl, weights=g.w)) * 100
    avg = np.average(pnl, weights=w) * 100
    return {"n": len(pnl), "win": float((pnl > 0).mean()),
            "avg_pnl": avg, "green": int((dg > 0).sum()), "days": len(dg),
            "total": float(pnl.sum()) * 100}


def topn(score: np.ndarray, n: int) -> np.ndarray:
    idx = np.argsort(-score)[:n]
    m = np.zeros(len(score), dtype=bool); m[idx] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--base-thr", type=float, default=0.75)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0

    s = pd.read_parquet(C.OUTPUTS_DIR / "analysis" / "engine_stats.parquet")
    s["conf"] = np.maximum(s.p_up, s.p_down)
    s["side"] = np.where(s.p_up >= s.p_down, 1, -1)
    last_fold = s.fold.max()
    tr = s[s.fold < last_fold].copy()
    te = s[s.fold == last_fold].copy()
    cut = (pd.to_datetime(te.day).max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    print(f"train folds={sorted(tr.fold.unique())} ({len(tr)}), "
          f"test fold={last_fold} ({len(te)}), last10 > {cut}, cost={cost*100:.3f}%")

    base_mask = te.conf >= args.base_thr
    N = int(base_mask.sum())
    print(f"baseline budget N={N} trades (conf>={args.base_thr})\n")

    feats = ["p_up", "p_down", "pred_ret", "pred_mfe", "pred_mae", "conf"]
    results = {}

    def pack(df):
        d = df.copy()
        if "w" not in d:
            d["w"] = 1.0
        return d

    # 0 baseline
    results["0 baseline"] = pack(te[base_mask])

    # 1 stacking: meta predicts the engine-trade pnl
    meta = CatBoostRegressor(iterations=400, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                             subsample=0.8, random_seed=42, verbose=0, allow_writing_files=False)
    trX = tr[feats].assign(hmin=tr.horizon.map(HMIN))
    trY = tr.side * tr.real_ret - cost
    meta.fit(trX, trY)
    mscore = meta.predict(te[feats].assign(hmin=te.horizon.map(HMIN)))
    results["1 stacking"] = pack(te[topn(mscore, N)])

    # 2 risk-adjusted: p_dir * favorable/adverse excursion (the strong MFE/MAE models)
    fav = np.where(te.side == 1, te.pred_mfe, -te.pred_mae)
    adv = np.where(te.side == 1, np.abs(te.pred_mae), te.pred_mfe)
    rr = np.clip(fav / (np.abs(adv) + 1e-4), 0, 5)
    results["2 risk-adj"] = pack(te[topn(te.conf.to_numpy() * rr, N)])

    # 3 cross-sectional ranking: top-K per day by conf
    K = max(1, N // te.day.nunique())
    keep = te.groupby("day", group_keys=False).apply(
        lambda g: g.assign(_k=g.conf.rank(ascending=False, method="first")))
    results[f"3 rank top{K}/day"] = pack(te[(keep["_k"] <= K).values])

    # 4 calibration + sizing (baseline selection, weight by calibrated edge)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(tr.conf, (tr.side * tr.real_ret > cost).astype(int))
    d4 = te[base_mask].copy()
    d4["w"] = np.clip(2 * iso.predict(d4.conf) - 1, 0, 1)
    results["4 calib+size"] = d4

    # 5 horizon agreement: side taken if >=2 horizons (same symbol,day) conf>=0.65
    fire = te[te.conf >= 0.65]
    grp = fire.groupby(["symbol", "day", "side"]).size().rename("cnt").reset_index()
    agree = grp[grp.cnt >= 2][["symbol", "day", "side"]]
    m5 = fire.merge(agree, on=["symbol", "day", "side"], how="inner")
    results["5 horiz-agree"] = pack(m5)

    def show(title, subset_fn):
        print(f"=== {title} ===")
        print(f"  {'strategy':<16} {'n':>5} {'win':>5} {'avg_pnl':>9} {'green':>7} {'total':>8}")
        for name, df in results.items():
            d = subset_fn(df)
            if len(d) < 3:
                print(f"  {name:<16} n={len(d):>3} (too few)"); continue
            r = evaluate(d, cost)
            print(f"  {name:<16} {r['n']:>5} {r['win']:>5.3f} {r['avg_pnl']:>+8.4f}% "
                  f"{r['green']:>3}/{r['days']:<3} {r['total']:>+7.2f}%")
        print()

    show("FOLD 3 (last ~14 days)", lambda df: df)
    show(f"STRICT last-10-days (> {cut})", lambda df: df[df.day > cut])


if __name__ == "__main__":
    main()
