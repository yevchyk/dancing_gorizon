"""Crisis (down-regime) engine research on the fresh selloff.

Re-inference over the recent window (default last 6h) with both fast_v2 and the
old standard 5m/15m models, the realized forward return at each horizon, and a
per-anchor regime gauge (cross-sectional % of coins falling). Then it scores a
set of SHORT-biased "crisis" engines -- alone, combined, and with old-model
confirmation -- over the whole window and over the crisis-only anchors, plus a
long contrast to confirm longs bleed in the crash.

Run: python -m src.run_crisis_engine [hours]
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
NS = 60_000_000_000
HMIN = {"2m": 2, "5m": 5, "8m": 8, "10m": 10, "15m": 15}


def ret_at(ts_ns, close, a_ns, h):
    ei = int(np.searchsorted(ts_ns, a_ns, "right")) - 1
    xj = int(np.searchsorted(ts_ns, a_ns + h * NS, "right")) - 1
    if ei < 0 or xj <= ei or close[ei] <= 0:
        return np.nan
    return close[xj] / close[ei] - 1.0


def past_ret(ts_ns, close, a_ns, h):
    xj = int(np.searchsorted(ts_ns, a_ns, "right")) - 1
    ei = int(np.searchsorted(ts_ns, a_ns - h * NS, "right")) - 1
    if ei < 0 or xj <= ei or close[ei] <= 0:
        return np.nan
    return close[xj] / close[ei] - 1.0


def main() -> None:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    store = CandleStore(C.CANDLES_DIR)
    feng = FastComboEngine("pulse00")
    scurve = FastCurve(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    scols = scurve.columns()
    smodels = {}
    mdir = C.MODELS_DIR / "dir_prob"
    for key in ("up_15m", "down_5m", "down_15m"):
        smodels[key] = (joblib.load(mdir / f"{key}.joblib"),
                        joblib.load(mdir / f"{key}_columns.joblib"))

    now = pd.Timestamp.now(tz="UTC").floor("1min")
    end = now - pd.Timedelta(minutes=16)
    start = now - pd.Timedelta(hours=hours)
    anchors = pd.date_range(start.ceil("2min"), end.floor("2min"), freq="2min")
    ans = anchors.as_unit("ns").asi8
    syms = sorted({p.stem for p in FC.FAST_CHUNKS_DIR.glob("*.parquet")} - set(C.BLACKLIST_SYMBOLS))

    rows = []
    for sym in syms:
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        ts = index_to_ns(c.index); cl = c["close"].to_numpy("float64")
        ff, fv = feng.curve.build_matrix(ts, cl, ans)
        sf, sv = scurve.build_matrix(ts, cl, ans)
        v = fv & sv
        if v.sum() == 0:
            continue
        idx = np.where(v)[0]
        Xf = pd.DataFrame(ff[idx], columns=feng.columns)
        Xs = pd.DataFrame(sf[idx], columns=scols)
        r = {"symbol": sym, "anchor": anchors[idx]}
        for nm, (m, co) in feng._models.items():
            r[f"f_{nm}"] = m.predict_proba(Xf[co])[:, 1]
        for nm, (m, co) in smodels.items():
            r[f"s_{nm}"] = m.predict_proba(Xs[co])[:, 1]
        a_idx = anchors[idx].as_unit("ns").asi8
        for h, mn in HMIN.items():
            r[f"ret_{h}"] = [ret_at(ts, cl, a, mn) for a in a_idx]
        r["past6"] = [past_ret(ts, cl, a, 6) for a in a_idx]
        rows.append(pd.DataFrame(r))
    d = pd.concat(rows, ignore_index=True)

    # ---- regime: per-anchor cross-sectional % of coins down over last 6 min ----
    reg = d.dropna(subset=["past6"]).groupby("anchor")["past6"].agg(
        med="median", frac_down=lambda x: float((x < 0).mean()))
    d = d.merge(reg, on="anchor", how="left")
    mkt_drift = d.drop_duplicates("anchor")["med"].median()
    crisis_thr = 0.60
    crisis_anchors = reg.index[reg["frac_down"] >= crisis_thr]
    d["crisis"] = d["anchor"].isin(crisis_anchors)
    print(f"window {start:%H:%M}->{end:%H:%M} UTC  anchors={d['anchor'].nunique()} "
          f"symbols={d['symbol'].nunique()}")
    print(f"market: median 6m drift={mkt_drift*100:+.3f}%/6m  "
          f"crisis anchors(>= {crisis_thr:.0%} coins red)={len(crisis_anchors)}/{len(reg)} "
          f"({len(crisis_anchors)/max(1,len(reg))*100:.0f}%)")

    def stat(mask, side, hcol, name, sub):
        m = mask & np.isfinite(d[hcol]) & sub
        n = int(m.sum())
        if n == 0:
            return f"  {name:34s} n=   0"
        ret = side * d.loc[m, hcol].to_numpy()
        pnl = ret - EVAL
        return (f"  {name:34s} n={n:4d} win={ (ret>0).mean():.3f} "
                f"avg%={pnl.mean()*100:+.4f} total%={pnl.sum()*100:+.1f}")

    fU2, fU5, fU8, fU10 = (d.f_up_2m, d.f_up_5m, d.f_up_8m, d.f_up_10m)
    fD2, fD5, fD8, fD10 = (d.f_down_2m, d.f_down_5m, d.f_down_8m, d.f_down_10m)
    sU15, sD5, sD15 = d.s_up_15m, d.s_down_5m, d.s_down_15m

    engines = [
        # how down_8m & down_10m work together
        ("down_8m&10m >=0.80 [exit10m]", (fD8 >= 0.80) & (fD10 >= 0.80), -1, "ret_10m"),
        ("down_8m&10m >=0.85 [exit10m]", (fD8 >= 0.85) & (fD10 >= 0.85), -1, "ret_10m"),
        ("down_8m alone >=0.85 [8m]", fD8 >= 0.85, -1, "ret_8m"),
        ("down_10m alone >=0.85 [10m]", fD10 >= 0.85, -1, "ret_10m"),
        # >=2 of the 3 short fast models
        (">=2 of down_5/8/10 >=0.80 [10m]",
         ((fD5 >= 0.80).astype(int)+(fD8 >= 0.80).astype(int)+(fD10 >= 0.80).astype(int)) >= 2,
         -1, "ret_10m"),
        # old confirmation
        ("down_8m&10m + std_down_15m>=0.85 [15m]",
         (fD8 >= 0.80) & (fD10 >= 0.80) & (sD15 >= 0.85), -1, "ret_15m"),
        ("std_down_15m alone >=0.88 [15m]", sD15 >= 0.88, -1, "ret_15m"),
        ("std_down_5m alone >=0.88 [5m]", sD5 >= 0.88, -1, "ret_5m"),
        # contrast: a LONG engine (Unicorn-like up agreement) in this regime
        ("CONTRAST up_8m&10m>=0.80 LONG [10m]", (fU8 >= 0.80) & (fU10 >= 0.80), 1, "ret_10m"),
    ]

    for label, sub in (("ALL anchors", pd.Series(True, index=d.index)),
                       ("CRISIS anchors only", d["crisis"])):
        print(f"\n=== {label} ===")
        for name, mask, side, hcol in engines:
            print(stat(mask, side, hcol, name, sub))


if __name__ == "__main__":
    main()
