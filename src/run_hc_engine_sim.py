"""Best-config engine sim + hourly report (risk-unit sizing, caps, cooldown).

Operating point chosen from the funnel analysis:
  model d7 (hc_final) | horizons {30,40,50,60} | opp<=0.05 | p_dir>=0.85
A RISK UNIT = one (symbol, scan): all its qualifying legs share ONE stake
(unit return = mean leg net%). Scheduler enforces max-concurrent + per-symbol
cooldown. Reports per Kyiv hour.
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
from .run_hc_dense_eval import candidates, add_outcomes
from .run_hc_prod_train import parse_cutoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("models/hc_final"))
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_full.json"))
    ap.add_argument("--horizons", default="30,40,50,60")
    ap.add_argument("--p-dir", type=float, default=0.85)
    ap.add_argument("--opp-cap", type=float, default=0.05)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--notional", type=float, default=15.0)
    ap.add_argument("--max-concurrent", type=int, default=15)
    ap.add_argument("--cooldown-min", type=int, default=30)
    ap.add_argument("--end-utc", default="", help="window end (UTC); default = freshest candle")
    args = ap.parse_args()

    horizons = tuple(int(x) for x in args.horizons.split(","))
    syms = json.loads(args.universe.read_text()); syms = syms.get("symbols", syms)
    edge = pd.read_parquet(C.CANDLES_DIR / f"{HC.BTC_SYMBOL}.parquet")["timestamp"].max()
    edge = pd.Timestamp(edge, tz="UTC") if pd.Timestamp(edge).tzinfo is None else pd.Timestamp(edge)
    if args.end_utc:
        edge = pd.Timestamp(args.end_utc, tz="UTC")
    start = edge - pd.Timedelta(hours=args.hours)
    entries = pd.date_range(start.ceil("5min"), edge, freq="5min", tz="UTC")
    print(f"### ENGINE SIM  model={args.model_dir.name}  window {start}..{edge}  "
          f"gate: h{horizons} p_dir>={args.p_dir} opp<={args.opp_cap} | "
          f"${args.notional:.0f}/unit maxconc={args.max_concurrent} cd={args.cooldown_min}m")

    feats = build_feature_rows(symbols=syms, entries=entries, horizons=horizons, entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN)
    scored = score_ensemble(feats, args.model_dir)
    cand = add_outcomes(candidates(scored, args.p_dir), edge)
    cand = cand[cand.p_opp <= args.opp_cap]
    if cand.empty:
        print("no candidates"); return

    units = (cand.groupby(["symbol", "base_time"])
             .agg(unit_net=("net", "mean"), k=("net", "size"),
                  entry=("entry_time", "first"), dl=("deadline", "max"),
                  p_dir=("p_dir", "max")).reset_index()
             .sort_values("entry").reset_index(drop=True))

    open_dl: list[pd.Timestamp] = []
    last_sym: dict[str, pd.Timestamp] = {}
    opened, blocked = [], 0
    for r in units.itertuples(index=False):
        open_dl = [d for d in open_dl if d > r.entry]
        last = last_sym.get(r.symbol)
        if len(open_dl) >= args.max_concurrent:
            blocked += 1; continue
        if last is not None and r.entry < last + pd.Timedelta(minutes=args.cooldown_min):
            blocked += 1; continue
        open_dl.append(r.dl)
        last_sym[r.symbol] = r.entry
        opened.append(r)

    op = pd.DataFrame(opened)
    op["usd"] = args.notional * op["unit_net"] / 100.0
    op["won"] = (op["unit_net"] > 0).astype(int)
    op["hour"] = pd.to_datetime(op["entry"], utc=True).dt.tz_convert("Europe/Kiev").dt.floor("1h")

    print(f"\nSUMMARY: units_opened={len(op)} (blocked={blocked})  "
          f"win={op.won.mean():.0%}  avg_net%={op.unit_net.mean():+.3f}  "
          f"total=${op.usd.sum():+.2f}  $/day={op.usd.sum()/args.hours*24:+.2f}  "
          f"maxDD=${(op.sort_values('entry').usd.cumsum().cummin().min()):+.2f}")
    print("\nHOURLY (Kyiv):")
    h = (op.groupby("hour").agg(units=("symbol", "size"), win=("won", "mean"),
                                avg_net=("unit_net", "mean"), usd=("usd", "sum")).reset_index())
    h["cum$"] = h["usd"].cumsum()
    print(h.to_string(index=False, formatters={
        "hour": lambda t: t.strftime("%m-%d %H:%M"), "win": "{:.0%}".format,
        "avg_net": "{:+.2f}".format, "usd": "{:+.2f}".format, "cum$": "{:+.2f}".format}))
    print("\nTOP UNITS:")
    t = op.sort_values("usd", ascending=False).head(12)[["symbol", "entry", "p_dir", "k", "unit_net", "usd"]]
    t["entry"] = pd.to_datetime(t["entry"], utc=True).dt.tz_convert("Europe/Kiev").dt.strftime("%H:%M")
    print(t.to_string(index=False, formatters={"p_dir": "{:.3f}".format, "unit_net": "{:+.2f}".format, "usd": "{:+.2f}".format}))


if __name__ == "__main__":
    main()
