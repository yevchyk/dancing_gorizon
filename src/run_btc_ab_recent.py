"""Does BTC context help on the RECENT down-market?

Re-inference of two model families over the last 3h and last 3 days, on
production candles, with realized returns recomputed from the same candles:
  v2  = fast_v2 (no BTC context)
  btc = fast_btc (BTC context)
Reports per-model AUC(event) and top-signal edge for each family.

Run: python -m src.run_btc_ab_recent
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .fast.curve import FastCurve
from .trading.timeutil import index_to_ns

EVAL, EDGE = FC.EVAL_COST, FC.TARGET_EDGE
HZ = {"2m": 2, "5m": 5, "8m": 8, "10m": 10}
NS = 60_000_000_000
FAMILIES = {"v2": "fast_v2", "btc": "fast_btc"}


def load_models(base):
    d = {}
    for h in HZ:
        for k in ("up", "down"):
            n = f"{k}_{h}"
            d[n] = (joblib.load(base / f"{n}.joblib"), joblib.load(base / f"{n}_columns.joblib"))
    return d


def fwd_ret(ts, close, a_ns, h):
    ei = int(np.searchsorted(ts, a_ns, "right")) - 1
    if ei < 0:
        return np.nan
    xj = int(np.searchsorted(ts, a_ns + h * NS, "right")) - 1
    return close[xj] / close[ei] - 1.0 if (xj > ei and close[ei] > 0) else np.nan


def topedge(prob, side, ret, n=200):
    x = pd.DataFrame({"p": prob, "r": ret}).dropna().sort_values("p", ascending=False).head(n)
    return (side * x["r"] - EVAL).mean() * 100 if len(x) else np.nan


def score_window(label, start, end, fcurve, bcurve, fcols_all, bcols_all,
                 fams, store, syms, btc_series):
    anchors = pd.date_range(start.ceil("2min"), end.floor("2min"), freq="2min")
    a_ns = anchors.as_unit("ns").asi8
    bt_ts, bt_close = btc_series
    rows = []
    for sym in syms:
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        ts = index_to_ns(c.index); cl = c["close"].to_numpy("float64")
        ff, fv = fcurve.build_matrix(ts, cl, a_ns)
        bf, bv = bcurve.build_matrix(bt_ts, bt_close, a_ns)
        ok = fv & bv
        if ok.sum() == 0:
            continue
        idx = np.where(ok)[0]
        X = pd.DataFrame(np.hstack([ff[idx], bf[idx]]), columns=fcols_all + bcols_all)
        r = {"symbol": sym}
        for key, models in fams.items():
            for n, (m, cols) in models.items():
                r[f"{key}_{n}"] = m.predict_proba(X[cols])[:, 1]
        sub = pd.DataFrame(r)
        for h, mn in HZ.items():
            sub[f"ret_{h}"] = [fwd_ret(ts, cl, x, mn) for x in anchors[idx].as_unit("ns").asi8]
        rows.append(sub)
    d = pd.concat(rows, ignore_index=True)

    keys = list(fams)
    print(f"\n===== {label}  ({start:%m-%d %H:%M}->{end:%m-%d %H:%M} UTC, rows={len(d)}) =====")
    hdr = "model".ljust(9) + "".join(f"AUC_{k}".rjust(8) for k in keys) + "  |" + \
          "".join(f"edge_{k}".rjust(9) for k in keys)
    print(hdr)
    agg = {k: {"auc": [], "edge": [], "auc_dn": [], "edge_dn": []} for k in keys}
    for h in HZ:
        ret = d[f"ret_{h}"].to_numpy()
        mask = np.isfinite(ret)
        for kk, side in (("up", 1), ("down", -1)):
            lab = (side * ret[mask] > EDGE).astype(int)
            line = f"{kk}_{h}".ljust(9)
            edges = ""
            for k in keys:
                p = d[f"{k}_{kk}_{h}"].to_numpy()
                au = roc_auc_score(lab, p[mask]) if lab.min() != lab.max() else np.nan
                ed = topedge(p, side, ret)
                line += f"{au:8.3f}"; edges += f"{ed:+9.3f}"
                agg[k]["auc"].append(au); agg[k]["edge"].append(ed)
                if kk == "down":
                    agg[k]["auc_dn"].append(au); agg[k]["edge_dn"].append(ed)
            print(line + "  |" + edges)
    print("  MEAN AUC : " + "  ".join(f"{k}={np.nanmean(agg[k]['auc']):.4f}" for k in keys))
    print("  DOWN AUC : " + "  ".join(f"{k}={np.nanmean(agg[k]['auc_dn']):.4f}" for k in keys))
    print("  DOWN edge: " + "  ".join(f"{k}={np.nanmean(agg[k]['edge_dn']):+.3f}" for k in keys))


def main() -> None:
    store = CandleStore(C.CANDLES_DIR)
    fcurve = FastCurve(FC.CURVE_POINTS, FC.CURVE_MIN_STEP_MIN, FC.CURVE_MAX_DEPTH_MIN, FC.CURVE_SEGMENTS)
    bcurve = FastCurve(0, FC.CURVE_MIN_STEP_MIN, FC.CURVE_MAX_DEPTH_MIN, offsets_min=FC.BTC_OFFSETS_MIN)
    fcols_all = fcurve.columns()
    bcols_all = FC.btc_columns()
    fams = {k: load_models(C.MODELS_DIR / exp / "base") for k, exp in FAMILIES.items()}
    syms = sorted({p.stem for p in FC.FAST_CHUNKS_DIR.glob("*.parquet")} - set(C.BLACKLIST_SYMBOLS))

    b = store.load(FC.BTC_SYMBOL).sort_index()
    btc_series = (index_to_ns(b.index), b["close"].to_numpy("float64"))
    now = pd.Timestamp.now(tz="UTC").floor("1min")

    def btc_move(hrs):
        p0 = b["close"][b.index <= now - pd.Timedelta(hours=hrs)].iloc[-1]
        p1 = b["close"][b.index <= now].iloc[-1]
        return (p1 / p0 - 1) * 100
    print(f"families: {FAMILIES}")
    print(f"BTC move: last 3h = {btc_move(3):+.2f}%   last 3d = {btc_move(72):+.2f}%")

    score_window("LAST 3 HOURS", now - pd.Timedelta(hours=3), now - pd.Timedelta(minutes=12),
                 fcurve, bcurve, fcols_all, bcols_all, fams, store, syms, btc_series)
    score_window("LAST 3 DAYS", now - pd.Timedelta(days=3), now - pd.Timedelta(minutes=12),
                 fcurve, bcurve, fcols_all, bcols_all, fams, store, syms, btc_series)


if __name__ == "__main__":
    main()
