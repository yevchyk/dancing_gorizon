"""What works over the last few hours? A broad signal-level scan.

Re-inference of fast_v2 + standard models over the recent window, then a battery
of strategies -- each model as-is AND faded (opposite side), the agreement
engines and their fades, the Drill short and anti-Drill long -- ranked by total
PnL. The point is to see whether ANYTHING has edge in the current regime, and in
particular whether fading the models (contrarian) beats following them.

Run: python -m src.run_last3h_scan [hours]
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


def ret_at(ts, cl, a, h):
    ei = int(np.searchsorted(ts, a, "right")) - 1
    xj = int(np.searchsorted(ts, a + h * NS, "right")) - 1
    return (cl[xj] / cl[ei] - 1.0) if (ei >= 0 and xj > ei and cl[ei] > 0) else np.nan


def main() -> None:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    store = CandleStore(C.CANDLES_DIR)
    feng = FastComboEngine("pulse00")
    scurve = FastCurve(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    scols = scurve.columns()
    smodels = {}
    mdir = C.MODELS_DIR / "dir_prob"
    for k in ("up_5m", "down_5m", "up_15m", "down_15m"):
        smodels[k] = (joblib.load(mdir / f"{k}.joblib"), joblib.load(mdir / f"{k}_columns.joblib"))

    now = pd.Timestamp.now(tz="UTC").floor("1min")
    end = now - pd.Timedelta(minutes=16)
    start = now - pd.Timedelta(hours=hours)
    anch = pd.date_range(start.ceil("2min"), end.floor("2min"), freq="2min")
    ans = anch.as_unit("ns").asi8
    syms = sorted({p.stem for p in FC.FAST_CHUNKS_DIR.glob("*.parquet")} - set(C.BLACKLIST_SYMBOLS))
    rows = []
    for sym in syms:
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index(); ts = index_to_ns(c.index); cl = c["close"].to_numpy("float64")
        ff, fv = feng.curve.build_matrix(ts, cl, ans)
        sf, sv = scurve.build_matrix(ts, cl, ans)
        v = fv & sv
        if v.sum() == 0:
            continue
        idx = np.where(v)[0]
        Xf = pd.DataFrame(ff[idx], columns=feng.columns)
        Xs = pd.DataFrame(sf[idx], columns=scols)
        r = {"anchor": anch[idx]}
        for nm, (m, co) in feng._models.items():
            r[f"f_{nm}"] = m.predict_proba(Xf[co])[:, 1]
        for nm, (m, co) in smodels.items():
            r[f"s_{nm}"] = m.predict_proba(Xs[co])[:, 1]
        aidx = anch[idx].as_unit("ns").asi8
        for h, mn in HMIN.items():
            r[f"ret_{h}"] = [ret_at(ts, cl, a, mn) for a in aidx]
        rows.append(pd.DataFrame(r))
    d = pd.concat(rows, ignore_index=True)
    print(f"window last {hours:.0f}h  anchors={d['anchor'].nunique()}  rows={len(d)}")

    results = []

    def add(name, mask, side, hcol):
        m = mask & np.isfinite(d[hcol])
        n = int(m.sum())
        if n < 3:
            return
        ret = side * d.loc[m, hcol].to_numpy()
        pnl = ret - EVAL
        results.append((name, n, float((ret > 0).mean()), float(pnl.mean()*100), float(pnl.sum()*100)))

    def topN(prob, side, hcol, name, n=15):
        m = np.isfinite(prob) & np.isfinite(d[hcol])
        x = pd.DataFrame({"p": prob[m], "r": d.loc[m, hcol].to_numpy()}).sort_values("p", ascending=False).head(n)
        if len(x) < 3:
            return
        ret = side * x["r"].to_numpy(); pnl = ret - EVAL
        results.append((name, len(x), float((ret > 0).mean()), float(pnl.mean()*100), float(pnl.sum()*100)))

    # per fast model: follow vs fade (top-15 by conviction)
    for h in ("2m", "5m", "8m", "10m"):
        for dr, s in (("up", 1), ("down", -1)):
            p = d[f"f_{dr}_{h}"].to_numpy()
            topN(p, s, f"ret_{h}", f"FOLLOW f_{dr}_{h}")
            topN(p, -s, f"ret_{h}", f"FADE   f_{dr}_{h}")
    # standard 5m/15m follow vs fade
    for h in ("5m", "15m"):
        for dr, s in (("up", 1), ("down", -1)):
            p = d[f"s_{dr}_{h}"].to_numpy()
            topN(p, s, f"ret_{h}", f"FOLLOW std_{dr}_{h}")
            topN(p, -s, f"ret_{h}", f"FADE   std_{dr}_{h}")
    # drill (down8&10>=0.80) short vs anti-drill long
    dmask = (d.f_down_8m >= 0.80) & (d.f_down_10m >= 0.80)
    add("DRILL down8&10 SHORT", dmask, -1, "ret_10m")
    add("ANTI-DRILL down8&10 LONG", dmask, 1, "ret_10m")
    # up-combo (Unicorn-ish) long vs fade
    umask = (d.f_up_8m >= 0.80) & (d.f_up_10m >= 0.80)
    add("UPCOMBO up8&10 LONG", umask, 1, "ret_10m")
    add("FADE UPCOMBO -> SHORT", umask, -1, "ret_10m")

    res = pd.DataFrame(results, columns=["strategy", "n", "win", "avg%", "total%"]).sort_values("total%", ascending=False)
    pd.set_option("display.width", 200)
    print("\n=== RANKED: what worked last {}h (signal-level, top-15 / threshold) ===".format(int(hours)))
    print(res.to_string(index=False, formatters={"win": "{:.3f}".format, "avg%": "{:+.4f}".format, "total%": "{:+.2f}".format}))
    pos = res[res["total%"] > 0]
    print(f"\npositive strategies: {len(pos)}/{len(res)}.  best: {res.iloc[0]['strategy']} ({res.iloc[0]['total%']:+.1f}%)")
    print("note: tiny samples (n<=15) over a few hours = mostly noise; read the PATTERN (follow vs fade), not single rows.")


if __name__ == "__main__":
    main()
