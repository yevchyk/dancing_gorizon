"""Does the regime DEPENDENCY transfer to tradfi, or is it inverted?

Crypto edge = MEAN-REVERSION (long wins in DOWN-herd / low breadth = buy the knife).
Equities may be MOMENTUM (long wins in UP-breadth = buy strength). So we do NOT
assume; we bucket the long edge across MANY regime dependencies for tradfi (Krykun)
and crypto (bluechip_short) side by side and read which way each market tilts.

Caveat: Krykun holdout is only 7 days, long-only (no p_down). In-sample buckets.

  python -m src.run_tradfi_regime --horizon 32m --thr 0.75
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


def weather_for(market, tradfi_tag="krykun_long"):
    if market == "crypto":
        store = CandleStore(C.DATA_DIR / "bluechip" / "candles_1m")
        syms = json.loads((C.ROOT / "configs" / "bluechip_symbols.json").read_text(encoding="utf-8"))
        syms = syms.get("symbols", syms) if isinstance(syms, dict) else syms
        scores = "outputs/analysis/fast_bluechip/bluechip_short/holdout_scores.parquet"
    else:
        d = C.DATA_DIR / "nasdaq" / "okx_candles_1m"
        store = CandleStore(d)
        syms = [p.stem for p in d.glob("*.parquet")]
        scores = f"outputs/analysis/fast_nasdaq/{tradfi_tag}/holdout_scores.parquet"
    lead = LEAD[market]
    if lead not in syms:
        syms = list(syms) + [lead]
    return WeatherStation(store, list(syms), lead), scores


def qrow(g, col, label, thr, q=5):
    g = g.copy()
    try:
        g["b"] = pd.qcut(g[col], q, labels=False, duplicates="drop")
    except Exception:
        return
    cells = []
    for i in range(q):
        gi = g[g.b == i]
        L = gi[gi["pu"] >= thr]
        cells.append(f"{(L['ret']>COST).mean():.3f}({len(L)})" if len(L) >= 20 else f"  -  ")
    print(f"  by {label:<14} Q1lo->Q5hi: " + "  ".join(f"{c:>11}" for c in cells))


def analyze(market, horizon, thr, tradfi_tag="krykun_long"):
    ws, scores = weather_for(market, tradfi_tag)
    s = pd.read_parquet(scores, columns=["symbol", "anchor_time", "day", f"p_up_{horizon}", f"real_ret_{horizon}"])
    s = s.rename(columns={f"p_up_{horizon}": "pu", f"real_ret_{horizon}": "ret"})
    uniq = s["anchor_time"].drop_duplicates().reset_index(drop=True)
    uns = pd.DatetimeIndex(pd.to_datetime(uniq, utc=True)).as_unit("ns").asi8
    st = ws.compute(uns)
    st["anchor_time"] = pd.to_datetime(uniq.values, utc=True)
    st["day"] = st["anchor_time"].dt.strftime("%Y-%m-%d")
    st = WeatherStation.add_causal(st)
    s = s.merge(st[["anchor_time", "breadth", "togetherness", "lead_vol", "breadth_slope",
                    "mkt_ret", "stage"]], on="anchor_time", how="left")

    base = s[s["pu"] >= thr]
    bw = (base["ret"] > COST).mean()
    print(f"\n===== {market.upper()} ({'krykun' if market=='tradfi' else 'bluechip_short'}) "
          f"horizon {horizon}  days={s['day'].nunique()}  =====")
    print(f"  baseline long p>={thr}: n={len(base)} win={bw:.3f} avg%={(base['ret']-COST).mean()*100:+.4f}")
    print(f"  [read: long-win rising with breadth = MOMENTUM ; falling = REVERSION]")
    qrow(s, "breadth", "BREADTH", thr)
    qrow(s, "togetherness", "TOGETHERNESS", thr)
    qrow(s, "lead_vol", "VOL", thr)
    qrow(s, "breadth_slope", "BREADTH_SLOPE", thr)
    # by stage
    print("  by STAGE:")
    for stg, g in s.groupby("stage"):
        L = g[g["pu"] >= thr]
        if len(L) >= 20:
            print(f"     {stg:<13} n={len(L):<5} win={(L['ret']>COST).mean():.3f} "
                  f"avg%={(L['ret']-COST).mean()*100:+.4f} mkt_fwd%={g['ret'].median()*100:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="32m")
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--tradfi-tag", default="krykun_long")
    ap.add_argument("--skip-crypto", action="store_true")
    args = ap.parse_args()
    analyze("tradfi", args.horizon, args.thr, args.tradfi_tag)
    if not args.skip_crypto:
        try:
            analyze("crypto", args.horizon, args.thr)
        except Exception as e:
            print(f"\n[crypto {args.horizon} skipped: {e}]")


if __name__ == "__main__":
    main()
