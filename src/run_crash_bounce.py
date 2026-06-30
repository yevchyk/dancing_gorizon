"""Is the 'crash-day top-longs win' a real, tradeable, frequent edge or 2-day noise?
Tests B's high-conviction longs bucketed by REAL-TIME BTC volatility (not hindsight
crash labels), + per-day breakdown + signal counts.

  python -m src.run_crash_bounce --horizon 32m
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .trading.timeutil import index_to_ns

sys.stdout.reconfigure(encoding="utf-8")
COST = 0.0012
NS_MIN = 60_000_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="32m")
    ap.add_argument("--volwin", type=int, default=60, help="BTC realized-vol window (min)")
    args = ap.parse_args()
    h = args.horizon

    s = pd.read_parquet(f"outputs/analysis/fast_bluechip/bluechip/holdout_scores.parquet",
                        columns=["symbol", "anchor_time", "day", f"p_up_{h}", f"real_ret_{h}"])
    s = s.rename(columns={f"p_up_{h}": "p", f"real_ret_{h}": "ret"})
    days = s["day"].nunique()

    # BTC realized vol per unique anchor
    btc = CandleStore(C.DATA_DIR / "bluechip" / "candles_1m").load("BTC_USDT_SWAP").sort_index()
    ts = index_to_ns(btc.index); close = btc["close"].to_numpy("float64")
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    uniq = s["anchor_time"].drop_duplicates().reset_index(drop=True)
    uns = pd.DatetimeIndex(pd.to_datetime(uniq, utc=True)).as_unit("ns").asi8
    ei = np.searchsorted(ts, uns, side="right") - 1
    vol = np.array([logret[max(0, e - args.volwin):e + 1].std() if e > 0 else 0.0 for e in ei])
    vmap = pd.DataFrame({"anchor_time": uniq, "btcvol": vol * 100})  # %
    s = s.merge(vmap, on="anchor_time", how="left")

    # --- per-day: top-50 longs, with the day's avg BTC vol ---
    print(f"=== per-day: B top-50 longs (horizon {h}) ===")
    print(f"{'day':<12}{'n':>5}{'win':>7}{'avg%':>9}{'btcvol%':>9}{'mkt_ret%':>9}")
    s["rk"] = s.groupby("day")["p"].rank(ascending=False, method="first")
    for d, g in s.groupby("day"):
        top = g[g.rk <= 50]; pnl = top["ret"].to_numpy() - COST
        print(f"{d:<12}{len(top):>5}{(pnl>0).mean():>7.3f}{pnl.mean()*100:>+9.4f}"
              f"{g['btcvol'].mean():>9.4f}{g['ret'].mean()*100:>+9.4f}")

    # --- REAL-TIME vol buckets: high-conviction longs by BTC-vol quintile (all days) ---
    print(f"\n=== high-conviction longs (top 5% by p) bucketed by REAL-TIME BTC vol ===")
    hp = s[s["p"] >= s["p"].quantile(0.95)].copy()
    hp["vbucket"] = pd.qcut(hp["btcvol"], 5, labels=["v1 low", "v2", "v3", "v4", "v5 high"], duplicates="drop")
    print(f"{'vol bucket':<10}{'n':>6}{'n/day':>7}{'win':>7}{'avg%':>9}{'$/d@30':>9}")
    for vb, g in hp.groupby("vbucket", observed=True):
        pnl = g["ret"].to_numpy() - COST
        print(f"{str(vb):<10}{len(g):>6}{len(g)//days:>7}{(pnl>0).mean():>7.3f}{pnl.mean()*100:>+9.4f}{pnl.sum()/days*30:>+9.2f}")

    # --- tradeable test: vol-trigger (top 20% vol) + top longs, across ALL days ---
    print(f"\n=== TRADEABLE: when BTC vol in top 20% -> take top-conviction longs ===")
    volthr = s["btcvol"].quantile(0.80)
    hv = s[s["btcvol"] >= volthr]
    for q in (0.90, 0.95):
        f = hv[hv["p"] >= hv["p"].quantile(q)]
        pnl = f["ret"].to_numpy() - COST
        print(f"  hi-vol & p-top{int((1-q)*100)}%: n={len(f)} ({len(f)//days}/day) "
              f"win={(pnl>0).mean():.3f} avg%={pnl.mean()*100:+.4f}")


if __name__ == "__main__":
    main()
