"""Backtest a SAVED PORTFOLIO of builds as ONE risk book over the last N hours.

Same decision logic as the live machine (src/trading/hc_portfolio_engine.py):
  - score every distinct model once (features built ONCE, shared),
  - per build apply level filters (side/p_dir/horizon-set/hour-of-day-Kyiv/ban),
    global p_dir floor, per-symbol best, per-engine slot quota,
  - cross-dedup across builds (1 pos/symbol/scan, higher p_dir wins),
  - then simulate the shared book over time: max-concurrent + per-symbol cooldown
    (PositionManager semantics), risk-unit net per position.

Prints SUMMARY + per-engine + HOURLY (Kyiv).

  python -m src.run_hc_portfolio_sim --portfolio configs/builds/portfolio_5x3.json --hours 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config as C
from .hc import config as HC
from .hc.costs import cost_fn_from_store
from .hc_historical_features import iter_feature_row_chunks_for_schema
from .hc_model_registry import EnsembleScorer, model_schema
from .markets import is_equity
from .run_hc_dense_eval import candidates, add_outcomes
from .trading.hc_portfolio_engine import SIM_TO_DIR, _apply, _build_horizons

COST = 0.75


def _key(model_dir: str) -> str:
    return Path(model_dir).name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", type=Path, default=Path("configs/builds/portfolio_5x3.json"))
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--scan-stride-min", type=int, default=5)
    ap.add_argument("--universe", type=Path, default=None, help="override config universe")
    ap.add_argument("--json-out", action="store_true")
    args = ap.parse_args()

    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cfg = json.loads(args.portfolio.read_text(encoding="utf-8"))
    builds = cfg.get("builds", [])
    if not builds:
        raise SystemExit("no builds in portfolio")
    stake = float(cfg.get("stake_margin", 5.0)); lev = int(cfg.get("leverage", 3))
    notional = stake * lev
    maxconc = int(cfg.get("max_concurrent", 12))
    cooldown_min = int(cfg.get("cooldown_min", 30))
    slots = int(cfg.get("slots_per_engine", 0))
    floor = float(cfg.get("min_p_dir", 0.70))
    universe_path = args.universe or Path(cfg.get("universe", "configs/hc_universe_full.json"))

    # window
    edge = pd.read_parquet(C.CANDLES_DIR / f"{HC.BTC_SYMBOL}.parquet")["timestamp"].max()
    edge = pd.Timestamp(edge, tz="UTC") if pd.Timestamp(edge).tzinfo is None else pd.Timestamp(edge)
    start = edge - pd.Timedelta(hours=args.hours)
    entries = pd.date_range(start.ceil(f"{args.scan_stride_min}min"), edge,
                            freq=f"{args.scan_stride_min}min", tz="UTC")
    hours = max(1.0, (edge - start).total_seconds() / 3600.0)

    # union horizons + universe (minus blacklist)
    union_h: set[int] = set()
    model_h: dict[str, set[int]] = {}
    for b in builds:
        hs = set(_build_horizons(b))
        union_h.update(hs)
        md = SIM_TO_DIR.get(b.get("sim", ""), b.get("sim", ""))
        model_h.setdefault(md, set()).update(hs)
    union_h = tuple(sorted(union_h))
    syms = json.loads(universe_path.read_text()); syms = syms.get("symbols", syms)
    blacklist = set(C.hc_blacklist_symbols())
    syms = [s for s in syms if s not in blacklist]

    print(f"portfolio={cfg.get('name')} builds={len(builds)} window {start}..{edge} ({hours:.1f}h) "
          f"scans={len(entries)} symbols={len(syms)} union_h={len(union_h)} "
          f"floor={floor} slots/eng={slots} maxconc={maxconc} cooldown={cooldown_min}m "
          f"${notional:.0f}/pos", flush=True)

    # score each distinct model with its own schema. V4/d12 needs 1m feature rows;
    # legacy models keep the old HC feature matrix.
    cand_by_model: dict[str, pd.DataFrame] = {}
    cost_fn = cost_fn_from_store()
    for md, hset in model_h.items():
        k = _key(md)
        schema = model_schema(Path(md))
        scorer = EnsembleScorer(Path(md))
        parts = []
        for feats in iter_feature_row_chunks_for_schema(
            schema,
            symbols=syms,
            entries=entries,
            horizons=tuple(sorted(hset)),
            entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN,
            batch_size=12,
        ):
            scored = scorer.score(feats)
            cpart = add_outcomes(candidates(scored, floor), edge, cost_fn=cost_fn)
            if not cpart.empty:
                parts.append(cpart)
        c = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        cand_by_model[k] = c
        print(f"  scored {k} ({schema}): legs(floor {floor})={len(c)}", flush=True)

    # ---- per-scan decide (mirror HCPortfolioEngine.decide) ----
    # best[(base_time, symbol)] = (leg_row_dict, engine_name)
    best: dict[tuple, tuple] = {}
    per_engine_offered = {b.get("name"): 0 for b in builds}
    for b in builds:
        md = SIM_TO_DIR.get(b.get("sim", ""), b.get("sim", ""))
        c = cand_by_model[_key(md)]
        if c.empty:
            continue
        banned = set(b.get("banned", []))
        levels = b.get("levels", [])
        bt = pd.to_datetime(c["base_time"], utc=True)
        epoch_min = ((bt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy()
        hod = ((epoch_min + 180) // 60) % 24
        sym = c["symbol"].to_numpy(); hm = c["horizon_minutes"].to_numpy()
        side = c["side"].to_numpy(); pdv = c["p_dir"].to_numpy(); pov = c["p_opp"].to_numpy()
        spv = c["spread"].to_numpy(); net = c["net"].to_numpy()
        et = pd.to_datetime(c["entry_time"], utc=True).to_numpy()
        btv = bt.to_numpy()
        # group legs by scan, build leg dicts, apply filters
        legs_by_scan: dict = {}
        for i in range(len(c)):
            s = str(sym[i])
            if s in banned:
                continue
            leg = {"sym": s, "h": int(hm[i]), "hod": int(hod[i]),
                   "eq": 1 if is_equity(s) else 0, "side": int(side[i]),
                   "pd": float(pdv[i]), "po": float(pov[i]), "sp": float(spv[i]),
                   "net": float(net[i]), "entry": et[i], "bt": btv[i]}
            legs_by_scan.setdefault(btv[i], []).append(leg)
        for scan_bt, legs in legs_by_scan.items():
            sel = _apply(legs, levels)
            if not sel:
                continue
            by_sym: dict[str, dict] = {}
            for l in sel:
                cur = by_sym.get(l["sym"])
                if cur is None or l["pd"] > cur["pd"]:
                    by_sym[l["sym"]] = l
            offered = sorted(by_sym.values(), key=lambda l: l["pd"], reverse=True)
            if slots > 0:
                offered = offered[:slots]
            per_engine_offered[b.get("name")] += len(offered)
            for l in offered:
                kkey = (scan_bt, l["sym"])
                prev = best.get(kkey)
                if prev is None or l["pd"] > prev[0]["pd"]:
                    best[kkey] = (l, b.get("name"))

    # ---- shared book sim: maxconc + per-symbol cooldown ----
    picks = [dict(leg, engine=eng) for (leg, eng) in best.values()]
    # chronological; within a scan prefer higher p_dir
    picks.sort(key=lambda l: (pd.Timestamp(l["entry"]).value, -l["pd"]))
    open_until: list[float] = []          # close-time ns of currently open positions
    last_open: dict[str, float] = {}       # symbol -> last open ns
    cd_ns = cooldown_min * 60 * 1_000_000_000
    opened: list[dict] = []
    for l in picks:
        ent_ns = pd.Timestamp(l["entry"]).value
        # free positions closed by now
        open_until = [c for c in open_until if c > ent_ns]
        if len(open_until) >= maxconc:
            continue
        lo = last_open.get(l["sym"])
        if lo is not None and ent_ns - lo < cd_ns:
            continue
        close_ns = ent_ns + l["h"] * 60 * 1_000_000_000
        open_until.append(close_ns)
        last_open[l["sym"]] = ent_ns
        opened.append(l)

    # ---- report ----
    n = len(opened)
    win = sum(1 for l in opened if l["net"] > 0) / n if n else 0.0
    avg = sum(l["net"] for l in opened) / n if n else 0.0
    usd = sum(notional * l["net"] / 100 for l in opened)
    print(f"\nSUMMARY (one risk book): trades={n} win={win*100:.1f}% avg_net%={avg:+.3f} "
          f"total=${usd:+.2f} $/day={usd/hours*24:+.2f} maxconc={maxconc}")
    print("per-engine opened:")
    from collections import Counter
    ce = Counter(l["engine"] for l in opened)
    for b in builds:
        nm = b.get("name")
        eo = [l for l in opened if l["engine"] == nm]
        ew = sum(1 for l in eo if l["net"] > 0) / len(eo) if eo else 0.0
        eu = sum(notional * l["net"] / 100 for l in eo)
        print(f"  {nm:12s} opened={ce.get(nm,0):3d} win={ew*100:4.0f}% ${eu:+7.2f} "
              f"(offered~{per_engine_offered.get(nm,0)})")

    if n:
        df = pd.DataFrame({"entry": [pd.Timestamp(l["entry"]) for l in opened],
                           "net": [l["net"] for l in opened]})
        df["hour"] = df["entry"].dt.tz_convert("Europe/Kiev").dt.floor("1h")
        h = df.groupby("hour").agg(n=("net", "size"), win=("net", lambda s: (s > 0).mean()),
                                   usd=("net", lambda s: (notional * s / 100).sum())).reset_index()
        h["cum$"] = h["usd"].cumsum()
        print("\nHOURLY (Kyiv):")
        print(h.to_string(index=False, formatters={
            "hour": lambda t: t.strftime("%m-%d %H:%M"),
            "win": "{:.0%}".format, "usd": "{:+.2f}".format, "cum$": "{:+.2f}".format}))

    if args.json_out:
        print("JSON " + json.dumps({"trades": n, "win_pct": round(win * 100, 1),
              "avg_net_pct": round(avg, 3), "total_usd": round(usd, 2),
              "usd_per_day": round(usd / hours * 24, 2)}))


if __name__ == "__main__":
    main()
