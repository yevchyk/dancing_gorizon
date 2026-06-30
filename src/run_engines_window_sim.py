"""Replay verkh_v2 (FINAL) + unicorn_v2 over the last N hours using live models +
candle store. Honest-ish (re-scored at each anchor, no lookahead) but fills at
candle close with fixed cost -- no real slippage/caps. Last gut-check before live.

  python -m src.run_engines_window_sim --hours 12
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .run_engines_v2_sim import VERKH_THRESH, UNI_THR, UNI_N, UNI_EXIT, VERKH_SIZE, VERKH_LEV, UNI_SIZE, UNI_LEV
from .trading.fast_v3_engine import FastV3Engine, V3_DATASET, V3_LABELS
from .trading.timeutil import index_to_ns

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HMIN = {"1m": 1, "2m": 2, "4m": 4, "8m": 8, "12m": 12, "20m": 20}
NS_MIN = 60_000_000_000
COST = FC.EVAL_COST


def realized(cache, sym, a_ns, h, last_ns):
    ts, close = cache[sym]
    ei = int(np.searchsorted(ts, a_ns, side="right")) - 1
    xj = int(np.searchsorted(ts, a_ns + h * NS_MIN, side="right")) - 1
    if ei < 0 or xj <= ei or ts[xj] > last_ns:
        return None
    return close[xj] / close[ei] - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--cadence", type=int, default=2)
    ap.add_argument("--market", choices=["crypto", "tradfi"], default="crypto")
    args = ap.parse_args()

    eng = FastV3Engine("verkh_v2")  # crypto-trained models, applied as-is (transfer)
    if args.market == "tradfi":
        import json
        store = CandleStore(C.DATA_DIR / "nasdaq" / "okx_candles_1m")
        watch = json.loads((C.ROOT / "configs" / "nasdaq_symbols.json").read_text())["symbols"]
    else:
        store = CandleStore(C.CANDLES_DIR)
        watch = list(pd.read_parquet(V3_DATASET, columns=["symbol"])["symbol"].unique())
    print(f"market={args.market}  candidate symbols={len(watch)}", flush=True)
    cache = {}
    for sym in watch:
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        cache[sym] = (index_to_ns(c.index), c["close"].to_numpy("float64"))
    last_ns = max(ts[-1] for ts, _ in cache.values())
    end = pd.Timestamp(last_ns, tz="UTC")
    anchors = pd.date_range(end - pd.Timedelta(hours=args.hours), end, freq=f"{args.cadence}min")
    print(f"window {anchors[0]} -> {anchors[-1]}  ({len(anchors)} anchors, {args.hours}h)  symbols={len(cache)}", flush=True)

    verkh_pnl, verkh_lab, verkh_hr = [], [], []
    uni_pnl, uni_side, uni_hr = [], [], []
    for a in anchors:
        a_ns = int(a.value)
        syms, rows = [], []
        for sym, (ts, close) in cache.items():
            f, valid = eng.curve.build_matrix(ts, close, np.array([a_ns], dtype="int64"))
            if bool(valid[0]):
                syms.append(sym); rows.append(f[0])
        if not rows:
            continue
        X = pd.DataFrame(rows, index=syms, columns=eng.columns)
        P = {}
        for lab in V3_LABELS:
            for side in ("up", "down"):
                model, cols = eng._models[f"{side}_{lab}"]
                P[f"{side}_{lab}"] = model.predict_proba(X[cols])[:, 1]
        # verkh: flat up pool
        for lab, thr in VERKH_THRESH.items():
            p = P[f"up_{lab}"]; h = HMIN[lab]
            for i, sym in enumerate(syms):
                if p[i] < thr:
                    continue
                r = realized(cache, sym, a_ns, h, last_ns)
                if r is None:
                    continue
                verkh_pnl.append(r - COST); verkh_lab.append(lab); verkh_hr.append(a.floor("h"))
        # unicorn: agreement both sides
        up_hits = np.zeros(len(syms), int); dn_hits = np.zeros(len(syms), int)
        for lab in V3_LABELS:
            up_hits += (P[f"up_{lab}"] >= UNI_THR).astype(int)
            dn_hits += (P[f"down_{lab}"] >= UNI_THR).astype(int)
        for i, sym in enumerate(syms):
            side = None
            if up_hits[i] >= UNI_N and dn_hits[i] == 0:
                side, sgn = "long", 1.0
            elif dn_hits[i] >= UNI_N and up_hits[i] == 0:
                side, sgn = "short", -1.0
            if side is None:
                continue
            r = realized(cache, sym, a_ns, HMIN[UNI_EXIT], last_ns)
            if r is None:
                continue
            uni_pnl.append(sgn * r - COST); uni_side.append(side); uni_hr.append(a.floor("h"))

    def summ(name, pnls, notional, extra=None):
        p = np.array(pnls)
        usd = p * notional
        print(f"\n--- {name}  (${notional:.0f} notional) ---")
        print(f"    signals={len(p)}  win={ (p>0).mean():.3f}  avg%={p.mean()*100:+.4f}  "
              f"total$={usd.sum():+.2f}  $/hr={usd.sum()/args.hours:+.2f}")
        if extra is not None:
            for key in sorted(set(extra)):
                m = np.array(extra) == key
                pk = p[m]
                print(f"      {str(key):<8} n={m.sum():<4} win={(pk>0).mean():.3f} "
                      f"avg%={pk.mean()*100:+.4f} $={ (pk*notional).sum():+.2f}")

    summ("verkh_v2 FINAL  (flat long pool)", verkh_pnl, VERKH_SIZE*VERKH_LEV, verkh_lab)
    summ("unicorn_v2  (agreement both sides)", uni_pnl, UNI_SIZE*UNI_LEV, uni_side)

    # per-hour table (per engine + combined + cumulative)
    vdf = pd.DataFrame({"hr": verkh_hr, "x": np.array(verkh_pnl) * VERKH_SIZE * VERKH_LEV})
    udf = pd.DataFrame({"hr": uni_hr, "x": np.array(uni_pnl) * UNI_SIZE * UNI_LEV})
    vg = vdf.groupby("hr")["x"].agg(s="sum", n="size")
    ug = udf.groupby("hr")["x"].agg(s="sum", n="size")
    allh = sorted(set(vg.index) | set(ug.index))
    print(f"\n{'hour UTC':<12}{'verkh n':>8}{'verkh$':>9}{'uni n':>7}{'uni$':>9}{'comb$':>9}{'cum$':>9}")
    cum = 0.0
    for h in allh:
        vs = float(vg["s"].get(h, 0.0)); vn = int(vg["n"].get(h, 0))
        us = float(ug["s"].get(h, 0.0)); un = int(ug["n"].get(h, 0))
        c = vs + us; cum += c
        print(f"{str(h)[5:16]:<12}{vn:>8}{vs:>+9.2f}{un:>7}{us:>+9.2f}{c:>+9.2f}{cum:>+9.2f}")
    vt = float(np.array(verkh_pnl).sum() * VERKH_SIZE * VERKH_LEV)
    ut = float(np.array(uni_pnl).sum() * UNI_SIZE * UNI_LEV)
    print(f"\nTOTAL 12h:  verkh={vt:+.2f}  unicorn={ut:+.2f}  COMBINED={vt+ut:+.2f}")


if __name__ == "__main__":
    main()
