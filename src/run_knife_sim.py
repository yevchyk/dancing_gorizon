"""Faithful CAUSAL paper-sim of the Knife (the validated long-only recipe).

Fixes the non-causal top-K/day selection used in analysis: here a trade fires when
  (causal herd gate) AND (clean wick) AND (p_up >= fixed high threshold)
all known at decision time. Weather thresholds are trailing-only (module add_causal).
Exit policy: single horizon, or STAGE-ADAPTIVE (deepest stress -> longer 32m exit,
per run_regime_direction: capitulation/dump long edge strengthens with horizon).

  python -m src.run_knife_sim --thr 0.72 --warmup 7
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .trading.weather import WeatherStation, LEAD

sys.stdout.reconfigure(encoding="utf-8")
COST = 0.0012
N = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bluechip_short")
    ap.add_argument("--thr", type=float, default=0.72, help="p_up high-conviction threshold")
    ap.add_argument("--warmup", type=int, default=7)
    ap.add_argument("--cap", type=int, default=8, help="max concurrent longs per anchor")
    args = ap.parse_args()

    horizons = ["8m", "12m", "32m"]
    cols = ["symbol", "anchor_time", "day"] + [f"p_up_{h}" for h in horizons] + [f"real_ret_{h}" for h in horizons]
    s = pd.read_parquet(f"outputs/analysis/fast_bluechip/{args.tag}/holdout_scores.parquet", columns=cols)

    # weather (causal) at all unique anchors
    store = CandleStore(C.DATA_DIR / "bluechip" / "candles_1m")
    syms = json.loads((C.ROOT / "configs" / "bluechip_symbols.json").read_text(encoding="utf-8"))
    syms = syms.get("symbols", syms) if isinstance(syms, dict) else syms
    ws = WeatherStation(store, list(syms), LEAD["crypto"])
    uniq = s["anchor_time"].drop_duplicates().reset_index(drop=True)
    uns = pd.DatetimeIndex(pd.to_datetime(uniq, utc=True)).as_unit("ns").asi8
    st = ws.compute(uns)
    st["anchor_time"] = pd.to_datetime(uniq.values, utc=True)
    st["day"] = st["anchor_time"].dt.strftime("%Y-%m-%d")
    st = WeatherStation.add_causal(st)
    s = s.merge(st[["anchor_time", "gate", "clean", "knife", "stage"]], on="anchor_time", how="left")

    days = sorted(s["day"].unique())
    val = days[args.warmup:]
    sv = s[s["day"].isin(val)].copy()
    nd = len(val)
    print(f"validation {val[0]}..{val[-1]} ({nd}d)  p_up>=much {args.thr}  cap={args.cap}/anchor  ${N:.0f}/pos")

    def sim(name, exit_of_row, gatecol="knife"):
        # gate on DOWN-herd 'knife' (clean & still-dropping) -> excludes euphoria,
        # which symmetric togetherness wrongly admitted. entry conviction = 8m prob.
        base = sv[sv[gatecol] == True].copy()
        base = base[base["p_up_8m"] >= args.thr]
        if base.empty:
            print(f"  {name:<22} (no trades)"); return None
        exh = base.apply(exit_of_row, axis=1)
        ret = np.array([base.iloc[i][f"real_ret_{exh.iloc[i]}"] for i in range(len(base))])
        base = base.assign(exh=exh.values, ret=ret, pnl=ret - COST)
        # cap concurrent per anchor: keep top-`cap` by p_up_8m
        base["rk"] = base.groupby("anchor_time")["p_up_8m"].rank(ascending=False, method="first")
        base = base[base["rk"] <= args.cap]
        dpd = base.groupby("day")["pnl"].sum() * N
        eq = dpd.reindex(val).fillna(0).cumsum()
        mdd = (eq.cummax() - eq).max()
        win = (base["pnl"] > 0).mean()
        print(f"  {name:<22} n={len(base):<4} n/d={len(base)/nd:>4.1f} win={win:.3f} "
              f"avg%={base['pnl'].mean()*100:+.4f} $/day={dpd.sum()/nd:+.2f} "
              f"posos={dpd.min():+.2f} maxDD={mdd:.1f} equity={eq.iloc[-1]:+.1f} green={(dpd>0).mean():.2f}")
        return base

    print("\n=== single-horizon exits ===")
    sim("knife exit 8m", lambda r: "8m")
    sim("knife exit 12m", lambda r: "12m")
    sim("knife exit 32m", lambda r: "32m")

    print("\n=== stage-adaptive exit (deep stress -> longer) ===")
    def adaptive(r):
        if r["stage"] in ("capitulation",):
            return "32m"
        if r["stage"] in ("dump",):
            return "12m"
        return "8m"
    sim("knife adaptive", adaptive)

    # stage mix of the fired trades
    fired = sv[(sv["knife"] == True) & (sv["p_up_8m"] >= args.thr)]
    print(f"\nfired-trade stage mix: " +
          "  ".join(f"{k}={v}" for k, v in fired["stage"].value_counts().items()))


if __name__ == "__main__":
    main()
