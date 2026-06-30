"""Live-faithful replay judge — Fix 5 of the sim plan. The single source of truth.

Replays the EXACT live decision path (HCPortfolioEngine.snapshot + decide) over a
window of real 5-min anchors on the live candle store, then scores realized
outcomes the way live actually trades:
  - entry at base_time + entry-delay (default 5 = the target Fix-1 convention),
  - exit  at entry + horizon,
  - per-instrument round-trip cost (src/hc/costs.py) instead of a flat 0.75%,
  - the portfolio's own universe + min_p_dir floor (so floor matches live, Fix 4).

It prints the gradator frontier (winrate/net per p_dir bucket) under the NEW
honest cost next to the OLD flat 0.75% so the verdict change is explicit, and can
validate that engine picks reproduce a live trades.csv.

  python -m src.run_hc_replay --portfolio <cfg.json> --from-ago-h 6 --to-ago-h 0
  python -m src.run_hc_replay --portfolio <cfg.json> --anchors 2026-06-09T17:16,2026-06-09T17:26
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .hc import config as HC
from .database import CandleStore
from .hc.costs import cost_pct
from .markets import is_equity
from .trading.hc_portfolio_engine import HCPortfolioEngine

OLD_FLAT_COST = 0.75


def _price_at(store: CandleStore, sym: str, when: pd.Timestamp) -> float | None:
    c = store.load(sym)
    if c is None or c.empty:
        return None
    idx = c.index.tz_localize("UTC") if c.index.tz is None else c.index
    sub = c[idx <= when]
    return float(sub["close"].iloc[-1]) if not sub.empty else None


def _build_engine(cfg: dict) -> tuple[HCPortfolioEngine, float]:
    notional = float(cfg.get("stake_margin", 5.0)) * int(cfg.get("leverage", 3))
    eng = HCPortfolioEngine(
        cfg["builds"], notional_usd=notional,
        universe_path=Path(cfg.get("universe", "configs/hc_universe_full.json")),
        profile=cfg.get("name", "replay"),
        min_p_dir=float(cfg.get("min_p_dir", 0.70)),
        slots_per_engine=int(cfg.get("slots_per_engine", 0)),
    )
    return eng, notional


def _anchors(args, edge: pd.Timestamp) -> list[pd.Timestamp]:
    if args.anchors:
        return [pd.Timestamp(a, tz="UTC") if pd.Timestamp(a).tzinfo is None
                else pd.Timestamp(a).tz_convert("UTC") for a in args.anchors.split(",")]
    start = (edge - pd.Timedelta(hours=args.from_ago_h)).ceil(f"{args.step_min}min")
    end = edge - pd.Timedelta(hours=args.to_ago_h)
    return list(pd.date_range(start, end, freq=f"{args.step_min}min", tz="UTC"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", type=Path, required=True)
    ap.add_argument("--from-ago-h", type=float, default=6.0)
    ap.add_argument("--to-ago-h", type=float, default=0.0)
    ap.add_argument("--step-min", type=int, default=5)
    ap.add_argument("--anchors", type=str, default="", help="comma list of explicit UTC anchors")
    ap.add_argument("--entry-delay", type=int, default=HC.EXEC_ENTRY_DELAY_MIN,
                    help="minutes base_time->entry (Fix 1 target convention = 5)")
    ap.add_argument("--universe", type=Path, default=None,
                    help="override the portfolio's universe (e.g. configs/hc_universe_liquid.json)")
    args = ap.parse_args()

    cfg = json.loads(args.portfolio.read_text(encoding="utf-8"))
    if args.universe is not None:
        cfg["universe"] = str(args.universe)
    eng, notional = _build_engine(cfg)
    store = CandleStore(C.CANDLES_DIR)
    symbols = eng.build_watchlist(store, top_n=0)
    edge = _price_at(store, HC.BTC_SYMBOL, pd.Timestamp.now(tz="UTC"))  # warm load
    edge = pd.read_parquet(C.CANDLES_DIR / f"{HC.BTC_SYMBOL}.parquet", columns=["timestamp"])["timestamp"].max()
    edge = pd.Timestamp(edge, tz="UTC") if pd.Timestamp(edge).tzinfo is None else pd.Timestamp(edge)
    anchors = _anchors(args, edge)
    print(f"engine: {eng.describe()}")
    print(f"universe={Path(cfg.get('universe','?')).name} watch={len(symbols)} floor={eng.min_p_dir} "
          f"entry_delay={args.entry_delay}m notional=${notional:.0f} | anchors={len(anchors)} "
          f"{anchors[0]}..{anchors[-1]}", flush=True)

    rows, scan_sigs = [], []
    for i, anchor in enumerate(anchors):
        snap = eng.snapshot(store, symbols, anchor)
        sigs = eng.decide(snap, top_n=int(cfg.get("top_per_scan", 12)))
        scan_sigs.append(len(sigs))
        for s in sigs:
            h = int(s.horizon.replace("m", ""))
            et = anchor + pd.Timedelta(minutes=args.entry_delay)
            xt = et + pd.Timedelta(minutes=h)
            ep = _price_at(store, s.symbol, et)
            xp = _price_at(store, s.symbol, xt) if xt <= edge else None
            gross = net_new = net_old = np.nan
            if ep and xp and ep > 0:
                gross = (xp / ep - 1.0) * 100.0 * (1 if s.side == "long" else -1)
                net_new = gross - cost_pct(s.symbol, candles=store.load(s.symbol), t=anchor)
                net_old = gross - OLD_FLAT_COST
            rows.append({"anchor": anchor, "symbol": s.symbol, "side": s.side, "h": h,
                         "p_dir": s.prob, "engine": s.engine, "eq": is_equity(s.symbol),
                         "matured": bool(ep and xp), "gross": gross,
                         "net_new": net_new, "net_old": net_old})
        if (i + 1) % 20 == 0:
            print(f"  ..{i+1}/{len(anchors)} anchors, {len(rows)} signals", flush=True)

    span_h = max(1e-9, (anchors[-1] - anchors[0]).total_seconds() / 3600.0)
    tot = int(sum(scan_sigs))
    print(f"\n==== SCAN-LEVEL ====  signals={tot} over {span_h:.1f}h = {tot/span_h:.2f}/h "
          f"| scans-with-signal {sum(1 for x in scan_sigs if x)}/{len(scan_sigs)}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no signals"); return
    mat = df[df["matured"]].copy()
    print(f"matured signals: {len(mat)}/{len(df)}\n")
    if mat.empty:
        print("nothing matured yet in window"); return

    print("==== GRADATOR (per p_dir bucket) — NEW honest cost vs OLD flat 0.75 ====")
    print(f"{'bucket':>10s} {'n':>4s} | {'win_new':>7s} {'net_new':>8s} {'$new':>7s} | {'win_old':>7s} {'net_old':>8s}")
    for lo in (0.70, 0.80, 0.85, 0.90):
        g = mat[mat["p_dir"] >= lo]
        if g.empty:
            print(f"  >={lo:.2f}   n=0"); continue
        wn = (g.net_new > 0).mean() * 100; an = g.net_new.mean()
        wo = (g.net_old > 0).mean() * 100; ao = g.net_old.mean()
        usd = (notional * g.net_new / 100).sum()
        print(f"  >={lo:.2f} {len(g):4d} | {wn:6.1f}% {an:+7.3f} {usd:+6.2f} | {wo:6.1f}% {ao:+7.3f}")

    print("\n==== per-symbol (matured) ====")
    per = (mat.groupby("symbol")
              .agg(n=("net_new", "size"), eq=("eq", "first"),
                   win_new=("net_new", lambda s: (s > 0).mean() * 100),
                   net_new=("net_new", "mean"), net_old=("net_old", "mean"))
              .sort_values("n", ascending=False))
    print(per.to_string(formatters={"win_new": "{:.0f}%".format, "net_new": "{:+.3f}".format,
                                    "net_old": "{:+.3f}".format}))

    print(f"\n==== TOTAL (matured) ==== trades={len(mat)}")
    rate = span_h >= 1.0 and not args.anchors  # $/day only meaningful over a real swept window
    for nm, col in (("NEW honest cost", "net_new"), ("OLD flat 0.75", "net_old")):
        usd = (notional * mat[col] / 100).sum()
        tail = f"  $/day={usd/span_h*24:+.2f}" if rate else ""
        print(f"  {nm:18s}: win={ (mat[col]>0).mean()*100:5.1f}%  avg_net%={mat[col].mean():+.3f}  "
              f"total=${usd:+.2f}{tail}")


if __name__ == "__main__":
    main()
