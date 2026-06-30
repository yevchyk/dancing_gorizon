"""Re-inference of the last ~3h: new (fast_v2) vs old (standard) models.

Scores both model families at every 2-minute anchor over the recent window from
the production candle store, computes the realized forward return at each horizon
from the same candles, and reports:
  1. last-3h verdict (top signals per fast_v2 model, top-10 by prob, + pooled),
  2. old standard models (5m, 15m only) the same way -- do they do better?,
  3. probability / disagreement: where new vs old call OPPOSITE sides, what wins.

Run: python -m src.run_last3h_models [hours]
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .fast.curve import FastCurve
from .trading.fast_combo_engine import FastComboEngine
from .trading.timeutil import index_to_ns

EVAL = FC.EVAL_COST
FAST_H = {"2m": 2, "5m": 5, "8m": 8, "10m": 10}
STD_H = {"5m": 5, "15m": 15}
NS = 60_000_000_000


def fwd_ret(ts_ns, close, anchor_ns, h_min):
    ei = int(np.searchsorted(ts_ns, anchor_ns, "right")) - 1
    if ei < 0:
        return np.nan
    xj = int(np.searchsorted(ts_ns, anchor_ns + h_min * NS, "right")) - 1
    if xj <= ei:
        return np.nan
    e = close[ei]
    return close[xj] / e - 1.0 if e > 0 else np.nan


def main() -> None:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    store = CandleStore(C.CANDLES_DIR)
    feng = FastComboEngine("pulse00")               # fast models + 320-col curve
    scurve = FastCurve(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    scols = scurve.columns()
    smodels = {}
    mdir = C.MODELS_DIR / "dir_prob"
    for lab in STD_H:
        for d in ("up", "down"):
            smodels[f"{d}_{lab}"] = (joblib.load(mdir / f"{d}_{lab}.joblib"),
                                     joblib.load(mdir / f"{d}_{lab}_columns.joblib"))

    now = pd.Timestamp.now(tz="UTC").floor("1min")
    end = now - pd.Timedelta(minutes=16)            # last anchor that has its 15m outcome
    start = now - pd.Timedelta(hours=hours)
    anchors = pd.date_range(start.ceil("2min"), end.floor("2min"), freq="2min")
    anchors_ns = anchors.as_unit("ns").asi8
    syms = sorted({p.stem for p in FC.FAST_CHUNKS_DIR.glob("*.parquet")}
                  - set(C.BLACKLIST_SYMBOLS))
    print(f"window {start:%H:%M}->{end:%H:%M} UTC  anchors={len(anchors)}  symbols={len(syms)}")

    rows = []
    for sym in syms:
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        ts_ns = index_to_ns(c.index)
        close = c["close"].to_numpy("float64")
        # fast (320-col) features
        ff, fv = feng.curve.build_matrix(ts_ns, close, anchors_ns)
        # std (300-col) features
        sf, sv = scurve.build_matrix(ts_ns, close, anchors_ns)
        valid = fv & sv
        if valid.sum() == 0:
            continue
        idx = np.where(valid)[0]
        Xf = pd.DataFrame(ff[idx], columns=feng.columns)
        Xs = pd.DataFrame(sf[idx], columns=scols)
        rec = {"symbol": sym, "anchor": anchors[idx]}
        for name, (m, cols) in feng._models.items():        # up_2m..down_10m
            rec[f"f_{name}"] = m.predict_proba(Xf[cols])[:, 1]
        for name, (m, cols) in smodels.items():              # up_5m/15m, down_5m/15m
            rec[f"s_{name}"] = m.predict_proba(Xs[cols])[:, 1]
        sub = pd.DataFrame(rec)
        for h, mn in {**FAST_H, **STD_H}.items():
            sub[f"ret_{h}"] = [fwd_ret(ts_ns, close, a, mn)
                               for a in anchors[idx].as_unit("ns").asi8]
        rows.append(sub)
    d = pd.concat(rows, ignore_index=True)

    def topN_stats(prob, side, ret, n=10):
        m = np.isfinite(prob) & np.isfinite(ret)
        x = pd.DataFrame({"p": prob[m], "ret": ret[m]}).sort_values("p", ascending=False).head(n)
        pnl = side * x["ret"] - EVAL
        return len(x), float((side*x["ret"] > 0).mean()), float(pnl.mean()*100), float(pnl.sum()*100), float(x["p"].mean())

    # ---- 1. fast_v2 per-model, top-10 by prob ----
    print("\n=== 1. NEW fast_v2 — top-10 highest-prob signals per model (last 3h) ===")
    print(f"{'model':10s} {'n':>2s} {'avgP':>5s} {'win':>5s} {'avg%':>7s} {'total%':>7s}")
    pooled = []
    for h in FAST_H:
        for d_ in ("up", "down"):
            side = 1 if d_ == "up" else -1
            n, win, avg, tot, ap = topN_stats(d[f"f_{d_}_{h}"].to_numpy(),
                                              side, d[f"ret_{h}"].to_numpy())
            print(f"{d_+'_'+h:10s} {n:2d} {ap:5.2f} {win:5.2f} {avg:+7.3f} {tot:+7.2f}")
            pooled.append(tot)
    print(f"{'POOLED':10s} (sum of the 8 top-10 baskets) total% = {sum(pooled):+.2f}")

    # ---- 2. OLD standard 5m/15m, top-10 by prob ----
    print("\n=== 2. OLD standard — top-10 highest-prob per model (5m, 15m) ===")
    print(f"{'model':12s} {'n':>2s} {'avgP':>5s} {'win':>5s} {'avg%':>7s} {'total%':>7s}")
    for h in STD_H:
        for d_ in ("up", "down"):
            side = 1 if d_ == "up" else -1
            n, win, avg, tot, ap = topN_stats(d[f"s_{d_}_{h}"].to_numpy(),
                                              side, d[f"ret_{h}"].to_numpy())
            print(f"{'std_'+d_+'_'+h:12s} {n:2d} {ap:5.2f} {win:5.2f} {avg:+7.3f} {tot:+7.2f}")

    # ---- 3. new vs old at 5m: agreement / disagreement ----
    print("\n=== 3. NEW vs OLD direction at 5m (where they agree vs differ) ===")
    fp_up, fp_dn = d["f_up_5m"].to_numpy(), d["f_down_5m"].to_numpy()
    sp_up, sp_dn = d["s_up_5m"].to_numpy(), d["s_down_5m"].to_numpy()
    ret5 = d["ret_5m"].to_numpy()
    fside = np.where(fp_up >= fp_dn, 1, -1)      # new model's call
    sside = np.where(sp_up >= sp_dn, 1, -1)      # old model's call
    fconf = np.maximum(fp_up, fp_dn)
    sconf = np.maximum(sp_up, sp_dn)
    conf = (fconf >= 0.55) & (sconf >= 0.55) & np.isfinite(ret5)   # both have an opinion
    agree = conf & (fside == sside)
    differ = conf & (fside != sside)
    def blk(mask, side_arr, label):
        n = int(mask.sum())
        if n == 0:
            print(f"  {label}: n=0"); return
        dc = ((side_arr[mask] * ret5[mask]) > 0).mean()
        avg = (side_arr[mask] * ret5[mask] - EVAL).mean() * 100
        print(f"  {label}: n={n:4d} dir_correct={dc:.3f} avg%={avg:+.4f}")
    blk(agree, fside, "AGREE (new & old same side)")
    # when they differ, who is right? score each family's own side
    print("  DIFFER (opposite sides) -> who wins:")
    blk(differ, fside, "    follow NEW")
    blk(differ, sside, "    follow OLD")

    print("\nnote: top-10 baskets are tiny (n<=10) -> read direction, not exact %.")


if __name__ == "__main__":
    main()
