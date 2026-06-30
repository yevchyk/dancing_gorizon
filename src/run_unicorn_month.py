"""Unicorn (PulseClean3 >=3 agree, exit 10m) day-by-day over the last ~month.

Re-inference using the fast_v2 worthy models. Features from the production store,
forward 10m return + entry from the 1m fast cache. One row per (symbol, anchor);
Unicorn fires long if >=3 up-worthy agree (no down) / short if >=3 down (no up).
Signal-level PnL = side*ret_10m - cost. Prints per-day win / avg / total so we can
pick the bad days to train on.

Run: python -m src.run_unicorn_month [days] [step_min]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .fast.candles import load_1m
from .trading.fast_combo_engine import FastComboEngine, WORTHY
from .trading.timeutil import index_to_ns

EVAL = FC.EVAL_COST
NS = 60_000_000_000


def fwd_ret(ts, cl, ans, h):
    ei = np.searchsorted(ts, ans, "right") - 1
    xj = np.searchsorted(ts, ans + h * NS, "right") - 1
    ok = (ei >= 0) & (xj > ei)
    out = np.full(len(ans), np.nan)
    e = np.where(ei >= 0, cl[np.clip(ei, 0, len(cl) - 1)], np.nan)
    x = np.where(xj >= 0, cl[np.clip(xj, 0, len(cl) - 1)], np.nan)
    out[ok] = x[ok] / e[ok] - 1.0
    return out


def main() -> None:
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 31.0
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    eng = FastComboEngine("pulse00")
    store = CandleStore(C.CANDLES_DIR)
    now = pd.Timestamp.now(tz="UTC").floor("1min")
    end = now - pd.Timedelta(minutes=15)
    start = now - pd.Timedelta(days=days)
    anch = pd.date_range(start.ceil(f"{step}min"), end.floor(f"{step}min"), freq=f"{step}min")
    ans = anch.as_unit("ns").asi8
    syms = sorted({p.stem for p in FC.FAST_CHUNKS_DIR.glob("*.parquet")} - set(C.BLACKLIST_SYMBOLS))
    print(f"window {start:%m-%d} -> {end:%m-%d %H:%M} UTC  anchors/sym={len(anch)}  symbols={len(syms)}")

    rows = []
    done = 0
    for sym in syms:
        feat = store.load(sym)
        tgt = load_1m(sym)
        if feat is None or feat.empty or tgt is None or tgt.empty:
            continue
        feat = feat.sort_index(); tgt = tgt.sort_index()
        ff, fv = eng.curve.build_matrix(index_to_ns(feat.index), feat["close"].to_numpy("float64"), ans)
        if fv.sum() == 0:
            continue
        idx = np.where(fv)[0]
        X = pd.DataFrame(ff[idx], columns=eng.columns)
        up = np.zeros(len(idx)); dn = np.zeros(len(idx))
        for _name, (mname, _sn, side, base) in WORTHY.items():
            m, cols = eng._models[mname]
            p = m.predict_proba(X[cols])[:, 1]
            if side == 1:
                up += (p >= base)
            else:
                dn += (p >= base)
        long_ok = (up >= 3) & (dn == 0)
        short_ok = (dn >= 3) & (up == 0)
        fire = long_ok | short_ok
        if fire.sum() == 0:
            done += 1
            continue
        a_fire = ans[idx][fire]
        ret = fwd_ret(index_to_ns(tgt.index), tgt["close"].to_numpy("float64"), a_fire, 10)
        side = np.where(long_ok[fire], 1, -1)
        rows.append(pd.DataFrame({
            "anchor": anch[idx][fire], "side": side, "ret": ret,
        }))
        done += 1
        if done % 25 == 0:
            print(f"  scored {done}/{len(syms)}", flush=True)
    d = pd.concat(rows, ignore_index=True)
    d = d[np.isfinite(d["ret"])].copy()
    d["pnl"] = d["side"] * d["ret"] - EVAL
    d["day"] = pd.to_datetime(d["anchor"], utc=True).dt.strftime("%m-%d")

    print(f"\n=== UNICORN per day (last {days:.0f}d, step {step}m) — signal level ===")
    print(f"{'day':6s} {'n':>4s} {'long%':>5s} {'win':>5s} {'avg%':>7s} {'total%':>8s}")
    g = d.groupby("day")
    rep = []
    for day, x in g:
        rep.append((day, len(x), float((x.side == 1).mean()), float((x.pnl > 0).mean()),
                    float(x.pnl.mean()*100), float(x.pnl.sum()*100)))
    for day, n, lp, win, avg, tot in rep:
        flag = "  <-- BAD" if tot < 0 else ""
        print(f"{day:6s} {n:4d} {lp:5.2f} {win:5.3f} {avg:+7.3f} {tot:+8.1f}{flag}")
    bad = sorted([r for r in rep if r[5] < 0], key=lambda r: r[5])
    print(f"\nWORST days (total% ascending): " + ", ".join(f"{r[0]}({r[5]:+.0f})" for r in bad[:8]))
    print(f"overall: n={len(d)} win={ (d.pnl>0).mean():.3f} total={d.pnl.sum()*100:+.1f}%")


if __name__ == "__main__":
    main()
