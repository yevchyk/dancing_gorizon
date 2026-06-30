"""Falling-knife vs floor: inside the herd gate, can we separate the mid-dump
LOSERS (05-15, 05-27, 06-03) from the bottom-bounce WINNERS (05-17, 06-04)
using REACTIVE turn/capitulation signals (no crash prediction model)?

Tested signals (per anchor, from the 120-coin universe + BTC, no leakage):
  breadth_slope = breadth_now - breadth_30m_ago   (turning up?)
  btc_accel     = ret(last15m) - ret(prev15m)      (decline decelerating?)
  btc_r15       = BTC return last 15m              (still knifing?)
  btc_dd        = BTC drawdown from 24h high       (how deep?)
  btc_vol_slope = vol(30m) - vol(prev30m)          (vol rolling over?)
  wick_frac     = universe lower-wick dominance    (capitulation buying?)

  python -m src.run_knife_test --horizon 8m --days 25 --k 50
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .trading.timeutil import index_to_ns

sys.stdout.reconfigure(encoding="utf-8")
COST = 0.0012
STORE = C.DATA_DIR / "bluechip" / "candles_1m"
SYMS = C.ROOT / "configs" / "bluechip_symbols.json"


def knife_metrics(anchor_ns: np.ndarray) -> pd.DataFrame:
    syms = json.loads(SYMS.read_text(encoding="utf-8"))
    syms = syms.get("symbols", syms) if isinstance(syms, dict) else syms
    store = CandleStore(STORE)
    n = len(anchor_ns)
    r60n = np.full((n, len(syms)), np.nan)
    r60p = np.full((n, len(syms)), np.nan)
    wick = np.full((n, len(syms)), np.nan)
    btc_accel = np.full(n, np.nan); btc_r15 = np.full(n, np.nan)
    btc_dd = np.full(n, np.nan); btc_vs = np.full(n, np.nan)
    for j, sym in enumerate(syms):
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index(); ts = index_to_ns(c.index)
        close = c["close"].to_numpy("float64"); high = c["high"].to_numpy("float64")
        low = c["low"].to_numpy("float64"); op = c["open"].to_numpy("float64")
        ei = np.searchsorted(ts, anchor_ns, side="right") - 1
        ok = ei >= 90
        idx = ei[ok]
        r60n[ok, j] = close[idx] / close[idx - 60] - 1
        r60p[ok, j] = close[idx - 30] / close[idx - 90] - 1
        rng = high[idx] - low[idx]
        lw = np.minimum(op[idx], close[idx]) - low[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            wick[ok, j] = np.where(rng > 0, lw / rng, np.nan)
        if sym.startswith("BTC"):
            lg = np.diff(np.log(close), prepend=np.log(close[0]))
            for k in np.where(ei >= 1440)[0]:
                e = ei[k]
                r_recent = close[e] / close[e - 15] - 1
                r_prior = close[e - 15] / close[e - 30] - 1
                btc_accel[k] = (r_recent - r_prior) * 100
                btc_r15[k] = r_recent * 100
                btc_dd[k] = (close[e] / close[e - 1440:e + 1].max() - 1) * 100
                btc_vs[k] = lg[e - 30:e + 1].std() - lg[e - 60:e - 30].std()
    nv = (~np.isnan(r60n)).sum(1).clip(min=1)
    breadth_now = np.nansum((r60n > 0), 1) / nv
    nvp = (~np.isnan(r60p)).sum(1).clip(min=1)
    breadth_30ago = np.nansum((r60p > 0), 1) / nvp
    return pd.DataFrame({
        "anchor_ns": anchor_ns,
        "breadth_now": breadth_now,
        "togetherness": np.maximum(breadth_now, 1 - breadth_now),
        "breadth_slope": breadth_now - breadth_30ago,
        "btc_accel": btc_accel, "btc_r15": btc_r15,
        "btc_dd": btc_dd, "btc_vol_slope": btc_vs,
        "wick_frac": np.nanmean(wick, 1),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bluechip_short")
    ap.add_argument("--experiment", default="fast_bluechip")
    ap.add_argument("--horizon", default="8m")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--days", type=int, default=25)
    ap.add_argument("--notional", type=float, default=30.0)
    args = ap.parse_args()
    h = args.horizon; N = args.notional

    path = f"outputs/analysis/{args.experiment}/{args.tag}/holdout_scores.parquet"
    s = pd.read_parquet(path, columns=["symbol", "anchor_time", "day", f"p_up_{h}", f"real_ret_{h}"])
    days = sorted(s["day"].unique())[-args.days:]
    s = s[s["day"].isin(days)].copy().rename(columns={f"p_up_{h}": "p", f"real_ret_{h}": "ret"})
    s["rk"] = s.groupby("day")["p"].rank(ascending=False, method="first")
    sel = s[s.rk <= args.k].copy()
    sel["pnl"] = sel["ret"] - COST
    sel["winb"] = (sel["pnl"] > 0).astype(int)

    uniq = sel["anchor_time"].drop_duplicates().reset_index(drop=True)
    uns = pd.DatetimeIndex(pd.to_datetime(uniq, utc=True)).as_unit("ns").asi8
    km = knife_metrics(uns)
    km["anchor_time"] = pd.to_datetime(uniq.values, utc=True)
    feats = ["breadth_slope", "btc_accel", "btc_r15", "btc_dd", "btc_vol_slope", "wick_frac"]
    sel = sel.merge(km[["anchor_time", "togetherness", "breadth_now"] + feats], on="anchor_time", how="left")

    tog_hi = sel["togetherness"].quantile(0.80)
    g = sel[sel["togetherness"] >= tog_hi].copy()
    print(f"horizon={h} window={days[0]}..{days[-1]} ({len(days)}d)")

    def stat(name, x):
        if len(x) == 0:
            print(f"  {name:<30} (none)"); return
        dpd = x.groupby("day")["pnl"].sum() * N
        print(f"  {name:<30} n={len(x):<4} days={x['day'].nunique():>2} win={x['winb'].mean():.3f} "
              f"avg%={x['pnl'].mean()*100:+.4f} $/day={dpd.sum()/len(days):+.2f} posos={dpd.min():+.2f}")

    print("\n=== inside the HERD GATE (togeth>=p80): does a turn/capitulation filter separate win/lose? ===")
    stat("GATE all (baseline)", g)
    stat("GATE & breadth_slope>0", g[g.breadth_slope > 0])
    stat("GATE & breadth_slope<=0", g[g.breadth_slope <= 0])
    stat("GATE & btc_accel>0", g[g.btc_accel > 0])
    stat("GATE & btc_accel<=0", g[g.btc_accel <= 0])
    stat("GATE & btc_r15>-0.3% (not knifing)", g[g.btc_r15 > -0.3])
    stat("GATE & btc_r15<=-0.3% (knifing)", g[g.btc_r15 <= -0.3])
    wmed = g.wick_frac.median()
    stat("GATE & wick_frac>median (bounced intrabar)", g[g.wick_frac > wmed])
    stat("GATE & wick_frac<=median (clean bar)", g[g.wick_frac <= wmed])
    stat("GATE & low-wick & r15<=-0.3 (clean knife)", g[(g.wick_frac <= wmed) & (g.btc_r15 <= -0.3)])
    stat("GATE & vol_slope<=0 (vol rolling over)", g[g.btc_vol_slope <= 0])
    print("  --- combined turn confirm ---")
    stat("GATE & (slope>0 OR accel>0)", g[(g.breadth_slope > 0) | (g.btc_accel > 0)])
    stat("GATE & slope>0 & accel>0", g[(g.breadth_slope > 0) & (g.btc_accel > 0)])

    print("\n=== gated trades by DAY: knife-vs-floor signals (eyeball 05-15/27 lose vs 05-17/06-04 win) ===")
    print(f"{'day':<12}{'n':>4}{'win':>7}{'$day':>8}{'br_now':>8}{'br_slp':>8}{'accel':>8}{'r15':>8}{'dd24h':>8}{'wick':>7}")
    for d, x in g.groupby("day"):
        print(f"{d:<12}{len(x):>4}{x['winb'].mean():>7.3f}{x['pnl'].sum()*N:>+8.2f}"
              f"{x['breadth_now'].mean() if 'breadth_now' in x else 0:>8.3f}"
              f"{x['breadth_slope'].mean():>+8.3f}{x['btc_accel'].mean():>+8.3f}"
              f"{x['btc_r15'].mean():>+8.3f}{x['btc_dd'].mean():>+8.2f}{x['wick_frac'].mean():>7.3f}")


if __name__ == "__main__":
    main()
