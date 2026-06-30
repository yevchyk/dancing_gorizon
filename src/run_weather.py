"""Demo + sanity for the Weather Station.

(A) REPRODUCE the validated causal ladder (L1->L2->L3) from the module alone, to
    prove src/trading/weather.py is a correct single source of truth.
(B) HOURLY weather telemetry for crypto AND tradfi (the user's "погодинно" view):
    stage distribution per day + a sample readout.

  python -m src.run_weather --horizon 8m --warmup 7
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


def syms_for(path):
    d = json.loads((C.ROOT / "configs" / path).read_text(encoding="utf-8"))
    return d.get("symbols", d) if isinstance(d, dict) else d


def station(market):
    if market == "crypto":
        store = CandleStore(C.DATA_DIR / "bluechip" / "candles_1m")
        syms = syms_for("bluechip_symbols.json")
    else:
        store = CandleStore(C.DATA_DIR / "nasdaq" / "okx_candles_1m")
        syms = [p.stem for p in (C.DATA_DIR / "nasdaq" / "okx_candles_1m").glob("*.parquet")]
    lead = LEAD[market]
    if lead not in syms:
        syms = list(syms) + [lead]
    return WeatherStation(store, list(syms), lead)


def reproduce_ladder(horizon, warmup, N=30.0):
    print("=" * 70)
    print(f"(A) REPRODUCE validated ladder from the module  (horizon {horizon})")
    ws = station("crypto")
    s = pd.read_parquet("outputs/analysis/fast_bluechip/bluechip_short/holdout_scores.parquet",
                        columns=["symbol", "anchor_time", "day", f"p_up_{horizon}", f"real_ret_{horizon}"])
    s = s.rename(columns={f"p_up_{horizon}": "p", f"real_ret_{horizon}": "ret"})
    s["rk"] = s.groupby("day")["p"].rank(ascending=False, method="first")
    sel = s[s.rk <= 50].copy()
    sel["pnl"] = sel["ret"] - COST; sel["winb"] = (sel["pnl"] > 0).astype(int)

    allanch = s["anchor_time"].drop_duplicates().reset_index(drop=True)
    uns = pd.DatetimeIndex(pd.to_datetime(allanch, utc=True)).as_unit("ns").asi8
    st = ws.compute(uns)
    st["anchor_time"] = pd.to_datetime(allanch.values, utc=True)
    st["day"] = st["anchor_time"].dt.strftime("%Y-%m-%d")
    st = WeatherStation.add_causal(st)
    sel = sel.merge(st[["anchor_time", "gate", "clean", "knife", "stage"]],
                    on="anchor_time", how="left")
    days = sorted(s["day"].unique())
    val = days[warmup:]
    v = sel[sel["day"].isin(val)]
    nd = len(val)

    def rep(name, x):
        if len(x) == 0:
            print(f"  {name:<28} (none)"); return
        dpd = x.groupby("day")["pnl"].sum() * N
        print(f"  {name:<28} n={len(x):<5} win={x['winb'].mean():.3f} "
              f"$/day={dpd.sum()/nd:+.2f} posos={dpd.min():+.2f}")
    print(f"  validation {val[0]}..{val[-1]} ({nd}d) — compare to run_ladder_validate")
    rep("L1 baseline top-50", v)
    rep("L2 module gate", v[v.gate == True])
    rep("L3 module gate+clean", v[v.clean == True])
    rep("L3b module clean-knife", v[v.knife == True])


def telemetry(market, days_back, step_min):
    print("=" * 70)
    print(f"(B) HOURLY weather telemetry — {market}")
    ws = station(market)
    lead = ws.store.load(ws.lead_symbol)
    if lead is None or lead.empty:
        print(f"  no lead data for {market}"); return
    hi = lead.index.max().floor("h")
    lo = hi - pd.Timedelta(days=days_back)
    grid = pd.date_range(lo, hi, freq=f"{step_min}min", tz="UTC")
    uns = grid.as_unit("ns").asi8
    st = ws.compute(uns)
    st["anchor_time"] = grid
    st["day"] = st["anchor_time"].dt.strftime("%Y-%m-%d")
    st = WeatherStation.add_causal(st)
    st = st.dropna(subset=["breadth"])
    print(f"  {ws.lead_symbol} window {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}  ({len(st)} {step_min}m frames)")
    print(f"\n  stage share by day:")
    stages = ["chop_up", "chop_down", "euphoria", "dump", "recovery", "capitulation", "calm"]
    have = [s for s in stages if (st["stage"] == s).any()]
    print(f"  {'day':<12}" + "".join(f"{s[:9]:>11}" for s in have) + f"{'breadth':>9}{'tog':>7}{'vol_pct':>8}")
    for d, g in st.groupby("day"):
        shares = "".join(f"{(g['stage']==s).mean()*100:>10.0f}%" for s in have)
        volpct = (g["lead_vol"] >= g["vol_hi"]).mean()
        print(f"  {d:<12}{shares}{g['breadth'].mean():>9.3f}{g['togetherness'].mean():>7.3f}{volpct:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="8m")
    ap.add_argument("--warmup", type=int, default=7)
    ap.add_argument("--days-back", type=int, default=12)
    ap.add_argument("--step-min", type=int, default=60)
    args = ap.parse_args()
    reproduce_ladder(args.horizon, args.warmup)
    telemetry("crypto", args.days_back, args.step_min)
    telemetry("tradfi", args.days_back, args.step_min)


if __name__ == "__main__":
    main()
