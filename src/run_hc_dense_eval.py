"""Dense-horizon gradator eval: does querying MORE horizons surface MORE
high-conviction signals at the SAME winrate? The model is horizon-conditioned
(continuous), so we can ask any horizon — we just weren't.

Builds features once on a dense grid, scores one model dir, computes real
entry+5m->exit outcomes for candidates, and prints the gradator frontier for a
SPARSE subset vs the DENSE grid over a locked holdout window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .hc import config as HC
from .run_hc_offgrid_sim import build_feature_rows, score_ensemble
from .run_hc_classic_sim import ProductionPriceBook
from .run_hc_prod_train import parse_cutoff

COST = 0.75
SPARSE = {30, 60, 90}


def candidates(scored: pd.DataFrame, floor: float) -> pd.DataFrame:
    up = scored["up_prob"].to_numpy()
    dn = scored["down_prob"].to_numpy()
    long = up >= dn
    d = scored[["symbol", "base_time", "entry_time", "horizon_minutes"]].copy()
    d["side"] = np.where(long, 1, -1)
    d["p_dir"] = np.where(long, up, dn)
    d["p_opp"] = np.where(long, dn, up)
    d["spread"] = d["p_dir"] - d["p_opp"]
    return d[d["p_dir"] >= floor].reset_index(drop=True)


def add_outcomes(cand: pd.DataFrame, edge: pd.Timestamp, cost_fn=None) -> pd.DataFrame:
    """Realized net% per candidate. `cost_fn(symbol)->%` gives per-instrument cost
    (Fix 2); default None keeps the legacy flat COST so existing evals are unchanged."""
    cand = cand.copy()
    cand["entry_time"] = pd.to_datetime(cand["entry_time"], utc=True)
    cand["deadline"] = cand["entry_time"] + pd.to_timedelta(cand["horizon_minutes"].astype("int64"), unit="min")
    cand = cand[cand["deadline"] <= edge].reset_index(drop=True)
    book = ProductionPriceBook()
    nets = np.full(len(cand), np.nan)
    for i, r in enumerate(cand.itertuples(index=False)):
        ep = book.at(r.symbol, r.entry_time)
        xp = book.at(r.symbol, r.deadline)
        if ep and xp and ep > 0:
            ret = (xp / ep - 1.0) * 100.0
            nets[i] = r.side * ret - (cost_fn(r.symbol) if cost_fn is not None else COST)
    cand["net"] = nets
    cand = cand.dropna(subset=["net"]).reset_index(drop=True)
    cand["won"] = (cand["net"] > 0).astype(int)
    return cand


def frontier(cand: pd.DataFrame, label: str) -> None:
    def ded(m):
        s = cand[m]
        return s.sort_values("p_dir", ascending=False).drop_duplicates(["symbol", "base_time"], keep="first")
    print(f"\n=== {label}: legs={len(cand)} scans={cand['base_time'].nunique()} ===")
    for nm, m in [("RAW90", cand.p_dir >= 0.90), ("RAW85", cand.p_dir >= 0.85),
                  ("RAW80", cand.p_dir >= 0.80), ("SPREAD80", cand.spread >= 0.80),
                  ("SPREAD70", cand.spread >= 0.70), ("SPREAD60", cand.spread >= 0.60),
                  ("bdw .80&opp.05", (cand.p_dir >= 0.80) & (cand.p_opp <= 0.05))]:
        d = ded(m)
        if len(d):
            print(f"  {nm:16s} n={len(d):4d} win={d.won.mean():.0%} net={d.net.mean():+.3f} sum={d.net.sum():+.1f}")
        else:
            print(f"  {nm:16s} n=0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--cutoff-local", required=True, help="holdout start (Kyiv); test = cutoff..edge")
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_plus_equities.json"))
    ap.add_argument("--dense", default="20,30,40,50,60,70,80,90,100,110,120")
    ap.add_argument("--scan-stride-min", type=int, default=5)
    ap.add_argument("--floor", type=float, default=0.55)
    args = ap.parse_args()

    cutoff = parse_cutoff(args.cutoff_local)
    dense = tuple(int(x) for x in args.dense.split(","))
    syms = json.loads(args.universe.read_text()); syms = syms.get("symbols", syms)
    # edge = freshest BTC candle
    book0 = ProductionPriceBook()
    edge = pd.read_parquet(C.CANDLES_DIR / f"{HC.BTC_SYMBOL}.parquet")["timestamp"].max()
    edge = pd.Timestamp(edge, tz="UTC") if pd.Timestamp(edge).tzinfo is None else pd.Timestamp(edge)
    entries = pd.date_range(cutoff.ceil(f"{args.scan_stride_min}min"), edge,
                            freq=f"{args.scan_stride_min}min", tz="UTC")
    print(f"model={args.model_dir.name} symbols={len(syms)} entries={len(entries)} "
          f"dense_horizons={dense} window {entries[0]}..{entries[-1]}", flush=True)

    feats = build_feature_rows(symbols=syms, entries=entries, horizons=dense,
                               entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN)
    scored = score_ensemble(feats, args.model_dir)
    cand = candidates(scored, args.floor)
    cand = add_outcomes(cand, edge)
    print(f"candidates with outcomes={len(cand)}")

    sparse = cand[cand["horizon_minutes"].isin(SPARSE)]
    frontier(sparse, f"SPARSE {sorted(SPARSE)}")
    frontier(cand, f"DENSE {list(dense)}")


if __name__ == "__main__":
    main()
