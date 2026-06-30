"""Funnel / correlator analysis: where do we trade QUANTITY for QUALITY?

For one model on the locked holdout, walks every filter stage and reports
legs / winrate / net so we can SEE where volume is lost and quality gained,
and where there is recoverable edge. Stages:
  A. p_dir threshold curve (the main quantity<->quality knob)
  B. per-horizon edge (which horizons to actually query)
  C. multi-leg vs one-per-scan dedup (volume lost to dedup)
  D. spread threshold curve
  E. opposite-prob cap (does p_opp filtering help)
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


def dedup(df):
    return df.sort_values("p_dir", ascending=False).drop_duplicates(["symbol", "base_time"], keep="first")


def row(label, s):
    if len(s) == 0:
        return f"  {label:18s} n=   0"
    return f"  {label:18s} n={len(s):5d} win={s.won.mean():.0%} net={s.net.mean():+.3f} sum={s.net.sum():+.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--cutoff-local", required=True)
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_plus_equities.json"))
    ap.add_argument("--dense", default="10,20,30,40,50,60,70,80,90,100,110,120,140,160,180")
    ap.add_argument("--floor", type=float, default=0.60)
    args = ap.parse_args()

    cutoff = parse_cutoff(args.cutoff_local)
    dense = tuple(int(x) for x in args.dense.split(","))
    syms = json.loads(args.universe.read_text()); syms = syms.get("symbols", syms)
    edge = pd.read_parquet(C.CANDLES_DIR / f"{HC.BTC_SYMBOL}.parquet")["timestamp"].max()
    edge = pd.Timestamp(edge, tz="UTC") if pd.Timestamp(edge).tzinfo is None else pd.Timestamp(edge)
    entries = pd.date_range(cutoff.ceil("5min"), edge, freq="5min", tz="UTC")
    print(f"\n############ FUNNEL {args.model_dir.name}  ({entries[0]}..{entries[-1]}) ############")
    feats = build_feature_rows(symbols=syms, entries=entries, horizons=dense, entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN)
    scored = score_ensemble(feats, args.model_dir)
    cand = add_outcomes(candidates(scored, args.floor), edge)
    print(f"raw candidates with outcomes (p_dir>={args.floor})={len(cand)}  scans={cand.base_time.nunique()}")

    print("\nA. p_dir threshold curve (dedup one bet/scan) — the quantity<->quality knob")
    for th in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92]:
        print(row(f"p_dir>={th:.2f}", dedup(cand[cand.p_dir >= th])))

    print("\nB. per-horizon edge at p_dir>=0.80 (leg-level) — which horizons carry edge")
    sel = cand[cand.p_dir >= 0.80]
    for h in sorted(cand.horizon_minutes.unique()):
        print(row(f"h={h}m", sel[sel.horizon_minutes == h]))

    print("\nC. multi-leg vs dedup at p_dir>=0.80 — volume lost to one-per-scan")
    print(row("all legs", sel))
    print(row("dedup 1/scan", dedup(sel)))

    print("\nD. spread threshold curve (dedup)")
    for th in [0.50, 0.60, 0.70, 0.80, 0.90]:
        print(row(f"spread>={th:.2f}", dedup(cand[cand.spread >= th])))

    print("\nE. opposite-prob cap at p_dir>=0.80 (dedup) — does p_opp filtering help")
    for cap in [0.05, 0.10, 0.20, 0.50, 1.0]:
        print(row(f"opp<={cap:.2f}", dedup(cand[(cand.p_dir >= 0.80) & (cand.p_opp <= cap)])))


if __name__ == "__main__":
    main()
