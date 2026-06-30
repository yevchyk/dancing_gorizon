"""Regime x Direction: does the model's edge tilt / REVERSE by weather regime?

For each Weather-Station stage, measure (a) the market's forward DRIFT, (b) the
high-conviction LONG edge (p_up>=thr), (c) the high-conviction SHORT edge
(p_down>=thr). This answers: where to go long, where to short, where the signal
inverts (longs lose & shorts win = reverse).

  python -m src.run_regime_direction --horizon 8m --thr 0.75
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
STAGES = ["capitulation", "recovery", "dump", "euphoria", "chop_up", "chop_down", "calm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="8m")
    ap.add_argument("--thr", type=float, default=0.75, help="high-conviction prob threshold")
    ap.add_argument("--tag", default="bluechip_short")
    args = ap.parse_args()
    h = args.horizon; thr = args.thr

    s = pd.read_parquet(f"outputs/analysis/fast_bluechip/{args.tag}/holdout_scores.parquet",
                        columns=["symbol", "anchor_time", "day",
                                 f"p_up_{h}", f"p_down_{h}", f"real_ret_{h}"])
    s = s.rename(columns={f"p_up_{h}": "pu", f"p_down_{h}": "pd", f"real_ret_{h}": "ret"})

    # weather stage per unique anchor (causal)
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
    s = s.merge(st[["anchor_time", "stage", "breadth", "togetherness"]], on="anchor_time", how="left")

    print(f"horizon={h}  high-conviction thr={thr}  rows={len(s):,}  cost={COST*100:.2f}%")
    print(f"\n{'stage':<13}{'anchors':>8}{'mkt_fwd%':>9} | "
          f"{'L n':>6}{'L win':>7}{'L avg%':>8} | {'S n':>6}{'S win':>7}{'S avg%':>8}  verdict")
    for stg in STAGES:
        g = s[s["stage"] == stg]
        if len(g) == 0:
            continue
        mkt = g["ret"].median() * 100
        L = g[g["pu"] >= thr]; S = g[g["pd"] >= thr]
        lw = (L["ret"] > COST).mean() if len(L) else float("nan")
        la = (L["ret"] - COST).mean() * 100 if len(L) else float("nan")
        sw = (-S["ret"] > COST).mean() if len(S) else float("nan")
        sa = (-S["ret"] - COST).mean() * 100 if len(S) else float("nan")
        # verdict
        v = []
        if len(L) >= 30 and lw > 0.55:
            v.append("LONG")
        if len(S) >= 30 and sw > 0.55:
            v.append("SHORT")
        if len(L) >= 30 and lw < 0.45:
            v.append("long-FADE")
        if not v:
            v = ["-"]
        print(f"{stg:<13}{len(g):>8}{mkt:>+9.3f} | "
              f"{len(L):>6}{lw:>7.3f}{la:>+8.4f} | {len(S):>6}{sw:>7.3f}{sa:>+8.4f}  {'/'.join(v)}")

    # reversal scan: per stage, is short-edge > long-edge? (candidate reverse zones)
    print(f"\n=== reversal scan (short_win - long_win by stage; >0 => shorts beat longs) ===")
    for stg in STAGES:
        g = s[s["stage"] == stg]
        L = g[g["pu"] >= thr]; S = g[g["pd"] >= thr]
        if len(L) < 30 or len(S) < 30:
            continue
        lw = (L["ret"] > COST).mean(); sw = (-S["ret"] > COST).mean()
        flag = "  <== SHORT-SKEW" if sw - lw > 0.05 else ("  <== long-skew" if lw - sw > 0.05 else "")
        print(f"  {stg:<13} long_win={lw:.3f}  short_win={sw:.3f}  diff={sw-lw:+.3f}{flag}")


if __name__ == "__main__":
    main()
