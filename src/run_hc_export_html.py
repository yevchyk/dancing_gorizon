"""Export per-model scored legs (raw probs + realized outcomes) to a JS file the
local HTML explorer loads. Lets you stack filters by hand instead of me guessing.

Output: reports/sim_explorer/data.js  (window.SYMS, window.SIM_META, window.SIMS)
Each leg = [symIdx, tmin, horizon, side(+1/-1), p_dir, p_opp, net%, eq(0/1)].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .hc import config as HC
from .hc.costs import cost_fn_from_store
from .markets import is_equity
from .hc_historical_features import iter_feature_row_chunks_for_schema
from .hc_model_registry import EnsembleScorer, model_schema
from .run_hc_offgrid_sim import build_feature_rows, score_ensemble
from .run_hc_dense_eval import candidates, add_outcomes
from .run_hc_prod_train import parse_cutoff

# Fix 2: one memoized per-instrument cost function shared by all models in an export
_COST_FN = cost_fn_from_store()

OUT = C.ROOT / "reports" / "sim_explorer" / "data.js"
MODELS = [("d7 (hc_final)", Path("models/hc_final")),
          ("d8 (hc_final_d8)", Path("models/hc_final_d8")),
          ("OLD", Path("models/hc_exec_stride120_nonoverlap")),
          ("NEW", Path("models/hc_exec_to20260604_prod"))]
# v4 (1-min-horizon) models scored DENSELY on FRESH candles (same path as
# run_hc_build) over the liquid universe, so the explorer matches the dense build
# sim. No dataset = nothing to go stale. floor 0.5 keeps p_dir>=0.55 builds usable.
V4_LIQUID_UNI = Path("configs/hc_universe_liquid.json")
V4_DENSE = tuple(range(10, 121, 5))   # 10..120 by 5 min — covers v4 build horizon ranges
V4_FLOOR = 0.5
V4_MODELS = [("min1 2-120", Path("models/min1_2to120")),
             ("min1 flat d9", Path("models/min1_flat_d9")),
             ("min1 flat d10", Path("models/min1_flat_d10")),
             ("min1 flat d12", Path("models/min1_flat_d12")),
             ("3mo flat d12", Path("models/min1_3mo_d12"))]


def _v4_legs(mdir: Path, sym_index: dict, floor: float, symbols: list,
             entries: pd.DatetimeIndex, horizons: tuple, edge: pd.Timestamp):
    """Score a v4 model DENSELY on fresh candles (5-min scans x dense horizons),
    exactly like run_hc_build, so the explorer reproduces the build sim instead of
    a sparse, frozen dataset. Returns (legs, set(horizons))."""
    scorer = EnsembleScorer(mdir)
    parts = []
    for feats in iter_feature_row_chunks_for_schema(
            "v4", symbols=symbols, entries=entries, horizons=horizons,
            entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN, batch_size=12):
        scored = scorer.score(feats)
        c = add_outcomes(candidates(scored, floor), edge, cost_fn=_COST_FN)
        if not c.empty:
            parts.append(c)
    cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    legs, hset = [], set()
    for r in cand.itertuples(index=False):
        s = str(r.symbol)
        si = sym_index.setdefault(s, len(sym_index)); hset.add(int(r.horizon_minutes))
        tmin = int(pd.Timestamp(r.base_time).value // 60_000_000_000)
        legs.append([si, tmin, int(r.horizon_minutes), int(r.side),
                     round(float(r.p_dir), 4), round(float(r.p_opp), 4),
                     round(float(r.net), 3), 1 if is_equity(s) else 0])
    return legs, hset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_full.json"))
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--dense", default="15,20,30,40,45,60,75,90,120,150,180")
    ap.add_argument("--floor", type=float, default=0.65)
    args = ap.parse_args()

    syms_list = json.loads(args.universe.read_text()); syms_list = syms_list.get("symbols", syms_list)
    dense = tuple(int(x) for x in args.dense.split(","))
    edge = pd.read_parquet(C.CANDLES_DIR / f"{HC.BTC_SYMBOL}.parquet")["timestamp"].max()
    edge = pd.Timestamp(edge, tz="UTC") if pd.Timestamp(edge).tzinfo is None else pd.Timestamp(edge)
    start = edge - pd.Timedelta(hours=args.hours)
    entries = pd.date_range(start.ceil("5min"), edge, freq="5min", tz="UTC")
    print(f"window {start}..{edge} entries={len(entries)} symbols={len(syms_list)} dense={dense}", flush=True)

    feats = build_feature_rows(symbols=syms_list, entries=entries, horizons=dense, entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN)

    sym_index: dict[str, int] = {}
    sims = {}
    for name, mdir in MODELS:
        scored = score_ensemble(feats, mdir)
        cand = add_outcomes(candidates(scored, args.floor), edge, cost_fn=_COST_FN)
        print(f"{name}: legs={len(cand)}", flush=True)
        legs = []
        for r in cand.itertuples(index=False):
            si = sym_index.setdefault(r.symbol, len(sym_index))
            tmin = int(pd.Timestamp(r.base_time).value // 60_000_000_000)  # epoch minutes
            legs.append([si, tmin, int(r.horizon_minutes), int(r.side),
                         round(float(r.p_dir), 4), round(float(r.p_opp), 4),
                         round(float(r.net), 3), 1 if is_equity(r.symbol) else 0])
        sims[name] = {"legs": legs}

    # append v4 (1-min-horizon) models scored DENSELY on fresh candles.
    # FULL universe (same as run_hc_build) so the numbers match; the explorer's
    # "liquid only" checkbox (on by default) then shows the honest tradeable subset.
    v4_floor = min(args.floor, V4_FLOOR)
    v4_syms = syms_list
    v4_horizons: set[int] = set()
    for name, mdir in V4_MODELS:
        if not mdir.exists() or model_schema(mdir) != "v4":
            print(f"{name} (v4) skipped: missing or non-v4 schema", flush=True)
            continue
        try:
            legs_v4, hz = _v4_legs(mdir, sym_index, v4_floor, v4_syms, entries, V4_DENSE, edge)
            sims[name] = {"legs": legs_v4}
            v4_horizons |= hz
            print(f"{name} (v4 dense): legs={len(legs_v4)} syms={len(v4_syms)} floor={v4_floor}", flush=True)
        except Exception as e:
            print(f"{name} (v4) skipped: {e}", flush=True)

    syms_arr = [s for s, _ in sorted(sym_index.items(), key=lambda kv: kv[1])]
    all_horizons = sorted(set(int(x) for x in dense) | v4_horizons)
    meta = {"cost_pct": 0.75, "cost_model": "per-instrument (hc.costs)",
            "window_start": str(start), "window_end": str(edge),
            "hours": args.hours, "horizons": all_horizons, "floor": args.floor,
            "tz": "Europe/Kiev"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("window.SIM_META=" + json.dumps(meta) + ";\n")
        f.write("window.SYMS=" + json.dumps(syms_arr) + ";\n")
        f.write("window.SIMS=" + json.dumps(sims, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT}  ({len(syms_arr)} syms, sims: {list(sims)})  size={OUT.stat().st_size//1024}KB")
    try:
        from .run_hc_export_meta import main as _inject_meta
        _inject_meta()  # (re)add window.LIQUID + window.COST so the explorer keeps them
    except Exception as e:
        print(f"meta inject skipped: {e}")


if __name__ == "__main__":
    main()
