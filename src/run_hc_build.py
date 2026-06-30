"""Run a SAVED EXPLORER BUILD as a sim on fresh data (so builds are usable).

A build = the JSON the explorer exports: {sim, mode, notional, period, reverse,
banned, levels:[{conditions:[{regulator,value}]}]}. This re-creates the exact
level logic (OR within a level, AND across levels) server-side on freshly built
candidates, and prints a summary + hourly report.

  python -m src.run_hc_build --build configs/builds/MyBuild.json
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
from .hc_model_registry import EnsembleScorer, model_dir_for_sim, model_schema
from .markets import is_equity
from .run_hc_dense_eval import candidates, add_outcomes
from .run_hc_prod_train import parse_cutoff  # noqa: F401 (kept for parity)

COST = 0.75  # legacy fallback; live path uses per-instrument cost_fn below


def _cond(l, c):
    r, v = c.get("regulator") or c.get("reg"), c.get("value", c.get("val"))
    pd_, po, sp, h, side, eq, lean, hod = l["pd"], l["po"], l["sp"], l["h"], l["side"], l["eq"], l["lean"], l["hod"]
    if r == "p_dir": return pd_ >= v
    if r == "p_dir_max": return pd_ <= v
    if r == "p_dir_sides": return pd_ >= v[0] if side > 0 else pd_ >= v[1]
    if r == "p_dir_long": return side < 0 or pd_ >= v
    if r == "p_dir_short": return side > 0 or pd_ >= v
    if r == "p_opp": return po <= v
    if r == "spread": return sp >= v
    if r == "cost_max": return l["cost"] <= v
    if r == "wday": return l["wd"] in set(v or [])
    if r == "lean_min": return lean >= v
    if r == "lean_max": return lean <= v
    if r == "hmin": return h >= v
    if r == "hmax": return h <= v
    if r == "horizon": return h in set(v or [])
    if r == "hour": return hod in set(v or [])
    if r == "asset": return v == "both" or (eq == 1 if v == "equity" else eq == 0)
    if r == "side": return v == "both" or (side > 0 if v == "long" else side < 0)
    # unknown regulator must FAIL LOUDLY: silently passing would run a looser
    # filter than the saved build promises.
    raise ValueError(f"unknown build regulator: {r!r}")


def _apply(legs, levels):
    cur = legs
    for L in levels:
        conds = L.get("conditions") or L.get("conds") or []
        if not conds:
            continue
        cur = [l for l in cur if any(_cond(l, c) for c in conds)]
    return cur


def _build_horizons(levels, dense):
    hs: set[int] = set()
    for L in levels:
        for c in L.get("conditions") or L.get("conds") or []:
            r = c.get("regulator") or c.get("reg")
            if r != "horizon":
                continue
            for x in c.get("value", c.get("val")) or []:
                hs.add(int(x))
    return tuple(sorted(hs)) if hs else tuple(int(x) for x in dense)


def _live_book(rows, max_conc=12, top_per_scan=10, cooldown=30):
    """Engine-realistic book on risk-units — mirror of explorer liveBook():
    per scan top-N by p_dir into free slots, 1 pos/symbol, horizon exit frees
    the slot, per-symbol cooldown after exit."""
    by_t: dict[int, list[dict]] = {}
    for r in rows:
        by_t.setdefault(int(r["t"]), []).append(r)
    cool: dict[str, float] = {}
    book: list[tuple[int, str]] = []
    out: list[dict] = []
    for t in sorted(by_t):
        book = [(e, s) for e, s in book if e > t + 5]
        held = {s for _, s in book}
        offered = 0
        for r in sorted(by_t[t], key=lambda x: x.get("pd", 0), reverse=True):
            if offered >= top_per_scan or len(book) >= max_conc:
                break
            if r["sym"] in held:
                continue
            if t + 5 < cool.get(r["sym"], float("-inf")):
                continue
            exit_t = t + 5 + int(r["h"])
            book.append((exit_t, r["sym"]))
            held.add(r["sym"])
            cool[r["sym"]] = exit_t + cooldown
            out.append(r)
            offered += 1
    return out


def _max_concurrent(rows):
    ev = []
    for r in rows:
        ev.append((int(r["t"]) + 5, 1))
        ev.append((int(r["t"]) + 5 + int(r["h"]), -1))
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, d in ev:
        cur += d
        mx = max(mx, cur)
    return mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_full.json"))
    ap.add_argument("--dense", default=",".join(str(x) for x in range(20, 181, 5)))
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--hours", type=float, default=None, help="override saved period: last N hours")
    ap.add_argument("--from-ago-h", type=float, default=None, help="override window start, hours before edge")
    ap.add_argument("--to-ago-h", type=float, default=None, help="override window end, hours before edge")
    ap.add_argument("--batch-size", type=int, default=12, help="symbols per scoring batch")
    ap.add_argument("--json-out", action="store_true", help="print machine summary line")
    args = ap.parse_args()

    b = json.loads(args.build.read_text(encoding="utf-8"))
    sim = b.get("sim", "OLD")
    mdir = model_dir_for_sim(sim)
    schema = model_schema(mdir)
    notional = float(b.get("notional", 15))
    mode = b.get("mode", "unit")
    reverse = bool(b.get("reverse"))
    banned = set(b.get("banned", []))
    levels = b.get("levels", [])
    floor = float(args.floor if args.floor is not None else b.get("floor", 0.65))
    dense = tuple(int(x) for x in args.dense.split(","))
    horizons = _build_horizons(levels, dense)

    edge = pd.read_parquet(C.CANDLES_DIR / f"{HC.BTC_SYMBOL}.parquet")["timestamp"].max()
    edge = pd.Timestamp(edge, tz="UTC") if pd.Timestamp(edge).tzinfo is None else pd.Timestamp(edge)
    per = b.get("period")
    if args.from_ago_h is not None or args.to_ago_h is not None:
        end = edge - pd.Timedelta(hours=float(args.to_ago_h or 0.0))
        start = edge - pd.Timedelta(hours=float(args.from_ago_h if args.from_ago_h is not None else args.hours or 64.0))
    elif args.hours is not None:
        end, start = edge, edge - pd.Timedelta(hours=float(args.hours))
    elif per:
        end = edge - pd.Timedelta(hours=float(per.get("toAgoH", 0)))
        start = edge - pd.Timedelta(hours=float(per.get("fromAgoH", 64)))
    else:
        end, start = edge, edge - pd.Timedelta(hours=64)
    hours = max(1.0, (end - start).total_seconds() / 3600.0)
    syms = json.loads(args.universe.read_text()); syms = syms.get("symbols", syms)
    entries = pd.date_range(start.ceil("5min"), end, freq="5min", tz="UTC")
    print(f"build sim={sim} model={mdir.name} schema={schema} mode={mode} reverse={reverse} "
          f"window {start}..{end} ({hours:.1f}h) symbols={len(syms)} "
          f"horizons={len(horizons)} [{min(horizons)}..{max(horizons)}] floor={floor}", flush=True)

    cost_fn = cost_fn_from_store()  # Fix 2: per-instrument round-trip cost
    scorer = EnsembleScorer(mdir)
    cand_parts = []
    for feats in iter_feature_row_chunks_for_schema(
        schema,
        symbols=syms,
        entries=entries,
        horizons=horizons,
        entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN,
        batch_size=args.batch_size,
    ):
        scored = scorer.score(feats)
        c = add_outcomes(candidates(scored, floor), edge, cost_fn=cost_fn)
        if not c.empty:
            cand_parts.append(c)
    cand = pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame()
    if cand.empty:
        print("no candidates"); return

    # to leg dicts + lean per scan + ban + reverse
    legs = []
    for r in cand.itertuples(index=False):
        if r.symbol in banned:
            continue
        net = float(r.net)
        if reverse:
            net = -net - 2 * cost_fn(r.symbol)
        tmin = int(pd.Timestamp(r.base_time).value // 60_000_000_000)
        legs.append({"sym": r.symbol, "t": tmin, "hod": ((tmin + 180) // 60) % 24,
                     "wd": ((tmin + 180) // 1440 + 4) % 7, "cost": cost_fn(r.symbol),
                     "entry": pd.Timestamp(r.entry_time), "h": int(r.horizon_minutes), "side": int(r.side),
                     "pd": float(r.p_dir), "po": float(r.p_opp), "sp": float(r.spread),
                     "net": net, "eq": 1 if is_equity(r.symbol) else 0})
    # lean per scan
    byT = {}
    for l in legs:
        a = byT.setdefault(l["t"], [0, 0, 0, 0])
        if l["side"] > 0: a[0] += l["pd"]; a[1] += 1
        else: a[2] += l["pd"]; a[3] += 1
    for l in legs:
        a = byT[l["t"]]; lo = a[0] / a[1] if a[1] else 0; sh = a[2] / a[3] if a[3] else 0
        l["lean"] = lo - sh

    sel = _apply(legs, levels)
    # risk units
    if mode in ("unit", "book"):
        groups = {}
        for l in sel:
            groups.setdefault((l["sym"], l["t"]), []).append(l)
        rows = []
        for (s, t), g in groups.items():
            rows.append({"sym": s, "t": t, "h": max(x["h"] for x in g),
                         "pd": max(x["pd"] for x in g),
                         "entry": g[0]["entry"], "net": sum(x["net"] for x in g) / len(g)})
        if mode == "book":
            bk = b.get("book") or {}
            rows = _live_book(rows, int(bk.get("max_concurrent", 12)),
                              int(bk.get("top_per_scan", 10)), int(bk.get("cooldown_min", 30)))
    else:
        rows = [{"sym": l["sym"], "t": l["t"], "h": l["h"],
                 "entry": l["entry"], "net": l["net"]} for l in sel]

    n = len(rows)
    win = sum(1 for r in rows if r["net"] > 0) / n if n else 0.0
    avg = sum(r["net"] for r in rows) / n if n else 0.0
    usd = sum(notional * r["net"] / 100 for r in rows)
    maxconc = _max_concurrent(rows)
    print(f"\nSUMMARY: trades={n} win={win*100:.1f}% avg_net%={avg:+.3f} "
          f"total=${usd:+.2f} $/day={usd/hours*24:+.2f} maxconc={maxconc}")
    if n:
        df = pd.DataFrame(rows)
        df["hour"] = pd.to_datetime(df["entry"], utc=True).dt.tz_convert("Europe/Kiev").dt.floor("1h")
        h = df.groupby("hour").agg(n=("net", "size"), win=("net", lambda s: (s > 0).mean()),
                                   usd=("net", lambda s: (notional * s / 100).sum())).reset_index()
        print("\nHOURLY (Kyiv):")
        print(h.to_string(index=False, formatters={"hour": lambda t: t.strftime("%m-%d %H:%M"),
              "win": "{:.0%}".format, "usd": "{:+.2f}".format}))
    if args.json_out:
        print("JSON " + json.dumps({"trades": n, "win_pct": round(win * 100, 1),
              "avg_net_pct": round(avg, 3), "total_usd": round(usd, 2),
              "usd_per_day": round(usd / hours * 24, 2),
              "max_concurrent": maxconc}))


if __name__ == "__main__":
    main()
