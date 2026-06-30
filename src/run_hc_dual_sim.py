"""Dual parallel engines as ONE portfolio: Zhnyvar (d7) + Snaiper (d8).

Each engine has its own profiled controllers. They run in parallel but share one
risk book: cross-dedup (never two positions on the same symbol/scan; keep higher
p_dir) + shared max-concurrent + per-symbol cooldown. Risk-unit sizing.
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

ENGINES = [
    {"name": "Zhnyvar", "model": Path("models/hc_final"),
     "horizons": {30, 40, 50, 60}, "p_dir": 0.85, "opp": 0.05},
    {"name": "Snaiper", "model": Path("models/hc_final_d8"),
     "horizons": {20, 30, 40, 50, 60, 70, 80, 90, 120, 160}, "p_dir": 0.85, "opp": 1.0},
]


def engine_units(scored, eng, edge):
    cand = candidates(scored, eng["p_dir"])
    cand = cand[cand.horizon_minutes.isin(eng["horizons"]) & (cand.p_opp <= eng["opp"])]
    if cand.empty:
        return pd.DataFrame()
    cand = add_outcomes(cand, edge)
    if cand.empty:
        return pd.DataFrame()
    u = (cand.groupby(["symbol", "base_time"])
         .agg(unit_net=("net", "mean"), k=("net", "size"), entry=("entry_time", "first"),
              dl=("deadline", "max"), p_dir=("p_dir", "max")).reset_index())
    u["engine"] = eng["name"]
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_full.json"))
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--notional", type=float, default=15.0)
    ap.add_argument("--max-concurrent", type=int, default=15)
    ap.add_argument("--cooldown-min", type=int, default=30)
    args = ap.parse_args()

    syms = json.loads(args.universe.read_text()); syms = syms.get("symbols", syms)
    edge = pd.read_parquet(C.CANDLES_DIR / f"{HC.BTC_SYMBOL}.parquet")["timestamp"].max()
    edge = pd.Timestamp(edge, tz="UTC") if pd.Timestamp(edge).tzinfo is None else pd.Timestamp(edge)
    start = edge - pd.Timedelta(hours=args.hours)
    entries = pd.date_range(start.ceil("5min"), edge, freq="5min", tz="UTC")
    union = tuple(sorted(set().union(*[e["horizons"] for e in ENGINES])))
    print(f"### DUAL SIM  window {start}..{edge}  union_horizons={union}  "
          f"${args.notional:.0f}/unit maxconc={args.max_concurrent} cd={args.cooldown_min}m")

    feats = build_feature_rows(symbols=syms, entries=entries, horizons=union, entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN)
    parts = []
    for eng in ENGINES:
        scored = score_ensemble(feats, eng["model"])
        u = engine_units(scored, eng, edge)
        print(f"  {eng['name']:8s} raw units={len(u)}")
        parts.append(u)
    allu = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    # cross-dedup: one position per (symbol, scan), keep higher p_dir
    allu = allu.sort_values("p_dir", ascending=False).drop_duplicates(["symbol", "base_time"], keep="first")
    allu = allu.sort_values("entry").reset_index(drop=True)

    # shared schedule: max-concurrent + per-symbol cooldown
    open_dl, last_sym, opened, blocked = [], {}, [], 0
    for r in allu.itertuples(index=False):
        open_dl = [d for d in open_dl if d > r.entry]
        last = last_sym.get(r.symbol)
        if len(open_dl) >= args.max_concurrent or (last is not None and r.entry < last + pd.Timedelta(minutes=args.cooldown_min)):
            blocked += 1; continue
        open_dl.append(r.dl); last_sym[r.symbol] = r.entry; opened.append(r)

    op = pd.DataFrame(opened)
    op["usd"] = args.notional * op["unit_net"] / 100.0
    op["won"] = (op["unit_net"] > 0).astype(int)
    op["hour"] = pd.to_datetime(op["entry"], utc=True).dt.tz_convert("Europe/Kiev").dt.floor("1h")

    print(f"\nPORTFOLIO: units={len(op)} (blocked={blocked})  win={op.won.mean():.0%}  "
          f"avg_net%={op.unit_net.mean():+.3f}  total=${op.usd.sum():+.2f}  "
          f"$/day={op.usd.sum()/args.hours*24:+.2f}")
    print("by engine (in final portfolio):")
    print(op.groupby("engine").agg(units=("symbol", "size"), win=("won", "mean"),
                                   avg_net=("unit_net", "mean"), usd=("usd", "sum")).to_string(
        formatters={"win": "{:.0%}".format, "avg_net": "{:+.2f}".format, "usd": "{:+.2f}".format}))
    print("\nHOURLY (Kyiv):")
    h = op.groupby("hour").agg(units=("symbol", "size"), win=("won", "mean"),
                               usd=("usd", "sum")).reset_index()
    h["cum$"] = h["usd"].cumsum()
    print(h.to_string(index=False, formatters={
        "hour": lambda t: t.strftime("%m-%d %H:%M"), "win": "{:.0%}".format,
        "usd": "{:+.2f}".format, "cum$": "{:+.2f}".format}))


if __name__ == "__main__":
    main()
