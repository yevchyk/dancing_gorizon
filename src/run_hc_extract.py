"""Max-money extraction with all controllers + correct risk-unit sizing.

A RISK UNIT = one (symbol, scan): all its qualifying legs (multi-leg over
horizons) share ONE stake, so the unit return = mean(leg net%). This avoids
counting correlated legs as independent bets. Reports the quantity/quality/$
tradeoff across the p_dir knob so we can pick a balance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config as C
from .hc import config as HC
from .run_hc_offgrid_sim import build_feature_rows, score_ensemble
from .run_hc_dense_eval import candidates, add_outcomes
from .run_hc_prod_train import parse_cutoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--cutoff-local", required=True)
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_plus_equities.json"))
    ap.add_argument("--dense", default="10,20,30,40,50,60,70,80,90,100,110,120,140,160,180")
    ap.add_argument("--horizons", default="", help="restrict to these horizons (csv); empty=all")
    ap.add_argument("--opp-cap", type=float, default=1.0)
    ap.add_argument("--floor", type=float, default=0.65)
    ap.add_argument("--notional", type=float, default=15.0, help="$ per risk unit")
    args = ap.parse_args()

    cutoff = parse_cutoff(args.cutoff_local)
    dense = tuple(int(x) for x in args.dense.split(","))
    syms = json.loads(args.universe.read_text()); syms = syms.get("symbols", syms)
    edge = pd.read_parquet(C.CANDLES_DIR / f"{HC.BTC_SYMBOL}.parquet")["timestamp"].max()
    edge = pd.Timestamp(edge, tz="UTC") if pd.Timestamp(edge).tzinfo is None else pd.Timestamp(edge)
    days = (edge - cutoff).total_seconds() / 86400.0
    entries = pd.date_range(cutoff.ceil("5min"), edge, freq="5min", tz="UTC")

    feats = build_feature_rows(symbols=syms, entries=entries, horizons=dense, entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN)
    scored = score_ensemble(feats, args.model_dir)
    cand = add_outcomes(candidates(scored, args.floor), edge)
    if args.horizons:
        keep = {int(x) for x in args.horizons.split(",")}
        cand = cand[cand.horizon_minutes.isin(keep)]
    if args.opp_cap < 1.0:
        cand = cand[cand.p_opp <= args.opp_cap]

    print(f"\n### EXTRACT {args.model_dir.name}  days={days:.2f}  notional=${args.notional:.0f}/unit "
          f"horizons={args.horizons or 'all'} opp<={args.opp_cap}")
    print(f"{'p_dir>=':>7} {'legs':>5} {'units':>5} {'leg_win':>7} {'unit_win':>8} "
          f"{'avg_unit%':>9} {'$/day':>8} {'maxconc':>7}")
    for th in [0.70, 0.75, 0.80, 0.85, 0.90]:
        sel = cand[cand.p_dir >= th]
        if len(sel) == 0:
            continue
        u = sel.groupby(["symbol", "base_time"]).agg(unit_net=("net", "mean"),
                                                     k=("net", "size"),
                                                     entry=("entry_time", "first"),
                                                     dl=("deadline", "max")).reset_index()
        usd = args.notional * u["unit_net"] / 100.0
        # max concurrent risk units (overlap of [entry, dl])
        ev = pd.concat([pd.Series(1, index=u["entry"]), pd.Series(-1, index=u["dl"])]).sort_index()
        maxconc = int(ev.cumsum().max()) if len(ev) else 0
        print(f"{th:7.2f} {len(sel):5d} {len(u):5d} {sel.won.mean():6.0%} "
              f"{(u.unit_net>0).mean():7.0%} {u.unit_net.mean():+8.3f} "
              f"{usd.sum()/days:+7.1f} {maxconc:7d}")


if __name__ == "__main__":
    main()
