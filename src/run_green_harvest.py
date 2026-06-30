"""Test the 'green harvest' idea: open on a clean signal, then every CHECK_MIN
minutes close the position the moment it's in profit (net of cost); if it never
goes green, close at the horizon. The bet: even on no-edge days price oscillates,
so we grab the green wiggles instead of needing a directional edge.

Compares green-harvest vs the plain fixed-horizon close, per trade and per day,
and reports how often a position touches green within 2/5 minutes.

Usage:
  python -m src.run_green_harvest --floor 0.75 --check-min 2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .trading.timeutil import index_to_ns, anchors_to_ns, NS_PER_MIN

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
OPP, AGREE, EXCL = 0.30, 2, {"down_1h", "down_2h"}
HMIN = {h.label: h.minutes for h in C.HORIZONS}


def entries(g: pd.DataFrame, floor: float, excl=EXCL, agree=AGREE) -> pd.DataFrame:
    """v4 clean+agree signals -> one row per (symbol,time,side) best-spread horizon."""
    raw = {}
    for h in C.HORIZONS:
        lab = h.label
        for kind, sgn in (("up", 1), ("down", -1)):
            if f"{kind}_{lab}" in excl:
                continue
            p = g[f"p_{kind}_{lab}"].to_numpy()
            opp = (g[f"p_down_{lab}"] if kind == "up" else g[f"p_up_{lab}"]).to_numpy()
            ok = np.isfinite(g[f"exit_{lab}"].to_numpy())
            m = ok & (p >= floor) & (opp <= OPP)
            idx = np.where(m)[0]
            for i in idx:
                key = (g["symbol"].iat[i], g["time"].iat[i], sgn)
                raw.setdefault(key, []).append((h.minutes, p[i] - opp[i], lab))
    rows = []
    for (sym, t, sgn), lst in raw.items():
        if len(lst) < agree:
            continue
        hm, _, lab = max(lst, key=lambda x: x[1])
        model = f"{'up' if sgn > 0 else 'down'}_{lab}"
        rows.append({"symbol": sym, "time": t, "side": sgn, "horizon_min": hm, "model": model})
    return pd.DataFrame(rows)


def _px(ts, close, t_ns):
    i = int(np.searchsorted(ts, t_ns, side="right")) - 1
    return close[i], i


def harvest(ts, close, entry_ns, side, horizon_min, check_min, cost, lag_min=0.0, exit_slip=0.0):
    ei = int(np.searchsorted(ts, entry_ns, side="right")) - 1
    if ei < 0:
        return None
    entry = close[ei]
    end_ns = entry_ns + horizon_min * NS_PER_MIN
    touched_2 = touched_5 = False
    chk = entry_ns + check_min * NS_PER_MIN
    while chk <= end_ns:
        px, ci = _px(ts, close, chk)
        if ci > ei:
            mins = (chk - entry_ns) / NS_PER_MIN
            green_now = side * (px / entry - 1.0) - cost > 0
            if side * (px / entry - 1.0) > 0:
                if mins <= 2: touched_2 = True
                if mins <= 5: touched_5 = True
            if green_now:
                # realism: fill not at the observed bar but lag_min later, minus exit slip
                fpx, fi = _px(ts, close, chk + int(lag_min * NS_PER_MIN))
                pnl = side * (fpx / entry - 1.0) - cost - exit_slip / 100.0
                return pnl, "green", touched_2, touched_5
        chk += check_min * NS_PER_MIN
    fi = int(np.searchsorted(ts, end_ns, side="right")) - 1
    pnl = side * (close[fi] / entry - 1.0) - cost if fi > ei else -cost
    return pnl, "horizon", touched_2, touched_5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.75)
    ap.add_argument("--check-min", type=float, default=2)
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--day", default="", help="only this day, e.g. 05-31")
    ap.add_argument("--lag-min", type=float, default=0.0, help="execution lag (min) before fill")
    ap.add_argument("--exit-slip", type=float, default=0.0, help="extra exit slippage %%")
    ap.add_argument("--exclude", default="down_1h,down_2h", help="comma models to drop")
    ap.add_argument("--agree", type=int, default=2)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0

    g = pd.read_parquet(C.DATASETS_DIR / "sim_grid.parquet")
    g["time"] = pd.to_datetime(g["time"], utc=True)
    E = entries(g, args.floor, set(x for x in args.exclude.split(",") if x), args.agree)
    E["day"] = E["time"].dt.strftime("%m-%d")
    if args.day:
        E = E[E["day"] == args.day]
    store = CandleStore(C.CANDLES_DIR)
    print(f"floor={args.floor} check={args.check_min}min -> {len(E)} entries")

    recs = []
    for sym, grp in E.groupby("symbol"):
        c = store.load(sym)
        if c is None:
            continue
        ts = index_to_ns(c.index); close = c["close"].to_numpy(float)
        ans = anchors_to_ns(grp["time"])
        for a, side, hm, day, mdl in zip(ans, grp["side"], grp["horizon_min"], grp["day"], grp["model"]):
            r = harvest(ts, close, int(a), int(side), int(hm), args.check_min, cost,
                        args.lag_min, args.exit_slip)
            if r is None:
                continue
            recs.append({"day": day, "model": mdl, "pnl": r[0], "reason": r[1],
                         "t2": r[2], "t5": r[3], "won": int(r[0] > 0)})
    H = pd.DataFrame(recs)
    print(f"\n=== GREEN HARVEST (close at first green / else horizon) ===")
    print(f"  trades={len(H)}  win={H.won.mean():.3f}  avg_pnl={H.pnl.mean()*100:+.4f}%  "
          f"total%={H.pnl.sum()*100:+.1f}")
    print(f"  closed green: {(H.reason=='green').mean()*100:.0f}%  at horizon: {(H.reason=='horizon').mean()*100:.0f}%")
    print(f"  touched green within 2min: {H.t2.mean()*100:.0f}%   within 5min: {H.t5.mean()*100:.0f}%")
    print(f"\n  by reason avg pnl: green={H[H.reason=='green'].pnl.mean()*100:+.4f}%  "
          f"horizon={H[H.reason=='horizon'].pnl.mean()*100:+.4f}%")
    print("\n  === PER-MODEL (green harvest) ===")
    print(f"   {'model':<9} {'n':>5} {'win':>6} {'avg_pnl':>9} {'green%':>7}")
    for m, gg in sorted(H.groupby("model"), key=lambda kv: -kv[1].pnl.mean()):
        print(f"   {m:<9} {len(gg):>5} {gg.won.mean():>6.3f} {gg.pnl.mean()*100:>+8.4f}% "
              f"{(gg.reason=='green').mean()*100:>6.0f}%")
    print("\n  === PER-DAY (green harvest) ===")
    for day, gg in H.groupby("day"):
        print(f"   {day}  n={len(gg):>4} win={gg.won.mean():.3f} pnl%={gg.pnl.sum()*100:+.2f}")


if __name__ == "__main__":
    main()
