"""Engine backtest WITH exits for a SET of builds (the Consensus/movie button).

Reads a spec {builds:[{sim,levels,exit_policy?}], exit_default, book, notional}.
Mirrors the live engine: score each distinct model once, each build offers 1
position per (symbol, scan), cross-dedup across builds (highest p_dir wins),
then a shared live book (slots/top/cooldown). For every held position computes
HOLD-to-horizon vs the winning build's exit policy (its own, else the default),
on the 1-min candle path. Outputs rich stats for both. Window = the last --days
up to now (one live pool, no sealed exam); the UI may narrow it with from/to.

  python -m src.run_binance_engine_exittest --spec spec.json --out engine_result.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config as C
from .binance_fetcher import norm_symbol
from .hc_model_registry import SIM_TO_DIR
from .markets import is_equity
from .run_binance_exittest import _seeds, _close_series, _cond, _apply

DATASET_NOW = C.ROOT / "data" / "binance_now" / "dataset"
DATASET_Y1 = C.ROOT / "data" / "binance_y1" / "dataset"
DATASET_V5 = C.ROOT / "data" / "binance_y1_v5" / "dataset"
COSTS = C.CONFIGS_DIR / "binance_costs.json"
LIQUID = C.CONFIGS_DIR / "binance_universe_liquid.json"
FLOOR = 0.55
ENTRY_DELAY = 5
RECHECK_MIN = 60


def _kyiv_day(bt: int) -> str:
    """Kyiv calendar day label (MM-DD) for an entry, using the +180min offset the
    rest of the file uses for hour/weekday."""
    return pd.Timestamp((bt + 180) * 60_000_000_000).strftime("%m-%d")


def _entry_desc(levels: list) -> str:
    """Human-readable entry filter of a build (first condition of each level)."""
    parts = []
    for L in levels or []:
        conds = L.get("conditions") or L.get("conds") or []
        if conds:
            c = conds[0]
            parts.append(f"{c.get('regulator', c.get('reg', '?'))}{c.get('value', c.get('val', ''))}")
    return " · ".join(parts) or "p_dir>=0.55"


def _score_model(mdir: Path, dset: Path, trade, start):
    feat = json.loads((mdir / "feature_names.json").read_text(encoding="utf-8"))
    seeds = _seeds(mdir)
    need = list(dict.fromkeys(feat + ["symbol", "base_time", "horizon_minutes", "ret_pct", "thr_pct"]))
    frames = []
    for s in trade:
        p = dset / f"{s}.parquet"
        if p.exists():
            t = pq.read_table(p, columns=need, filters=[("base_time", ">=", start)])
            if t.num_rows:
                frames.append(t.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df["base_time"] = pd.to_datetime(df["base_time"], utc=True)
    X = df[feat]
    df["up"] = np.mean([u.predict_proba(X, thread_count=6)[:, 1] for u, _ in seeds], axis=0)
    df["dn"] = np.mean([d.predict_proba(X, thread_count=6)[:, 1] for _, d in seeds], axis=0)
    return df


def _legs(df):
    em = (df["base_time"].astype("int64") // 60_000_000_000).to_numpy()
    hod = ((em + 180) // 60) % 24
    wd = ((em + 180) // 1440 + 4) % 7
    sym = df["symbol"].to_numpy(); hz = df["horizon_minutes"].to_numpy()
    up, dn = df["up"].to_numpy(), df["dn"].to_numpy(); thr = df["thr_pct"].to_numpy()
    legs = []
    for i in range(len(df)):
        for sd, pdv, po in ((1, up[i], dn[i]), (-1, dn[i], up[i])):
            if pdv < FLOOR:
                continue
            s = str(sym[i])
            legs.append({"sym": s, "bt": int(em[i]), "h": int(hz[i]), "side": sd, "pd": float(pdv),
                         "po": float(po), "sp": float(pdv - po), "hod": int(hod[i]), "wd": int(wd[i]),
                         "eq": 1 if is_equity(s) else 0, "lean": 0.0, "thr": float(thr[i])})
    return legs


def _live_book(positions, mc, top, cd):
    su = sorted(positions, key=lambda r: (r["bt"], -r["pd"]))
    cool = {}; book = []; out = []; i = 0; n = len(su)
    while i < n:
        t = su[i]["bt"]
        book = [e for e in book if e[0] > t + 5]
        held = {e[1] for e in book}
        offered = 0
        while i < n and su[i]["bt"] == t:
            r = su[i]
            if offered < top and len(book) < mc and r["sym"] not in held and t + 5 >= cool.get(r["sym"], -1e18):
                ex = t + 5 + r["h"]; book.append((ex, r["sym"])); held.add(r["sym"])
                cool[r["sym"]] = ex + cd; out.append(r); offered += 1
            i += 1
    return out


def _exit_net(cs, t0, entry_px, h, side, cost, ep, bpd, sym):
    sgn = 1.0 if side > 0 else -1.0
    et = pd.Timestamp(t0 * 60_000_000_000, tz="UTC") + pd.Timedelta(minutes=ENTRY_DELAY)
    xt = et + pd.Timedelta(minutes=h)
    path = cs[(cs.index >= et) & (cs.index <= xt)]
    if not len(path):
        return None, False
    fav = sgn * (path.to_numpy() / entry_px - 1.0) * 100
    tmins = ((path.index.astype("int64") // 60_000_000_000) - (et.value // 60_000_000_000)).to_numpy()
    trig = {}
    if ep.get("tp"):
        idx = np.where(fav >= ep["tp"]["pct"])[0]
        if len(idx): trig["tp"] = (tmins[idx[0]], ep["tp"]["pct"] - cost)
    if ep.get("crash"):
        idx = np.where(-fav >= ep["crash"]["pct"])[0]
        if len(idx): trig["crash"] = (tmins[idx[0]], -ep["crash"]["pct"] - cost)
    if ep.get("trail"):
        arm, give = ep["trail"]["arm"], ep["trail"]["give"]; peak = -1e9
        for j in range(len(fav)):
            if fav[j] > peak: peak = fav[j]
            if peak >= arm and fav[j] <= peak - give:
                trig["trail"] = (tmins[j], (peak - give) - cost); break
    if ep.get("reconfirm") and bpd is not None:
        floor = ep["reconfirm"]["floor"]; tk = RECHECK_MIN
        while tk < h:
            k = (sym, t0 + tk, side)
            if k in bpd and bpd[k] < floor:
                pidx = cs.index.searchsorted(et + pd.Timedelta(minutes=tk), side="right") - 1
                if pidx >= 0:
                    trig["reconfirm"] = (tk, sgn * (float(cs.iloc[pidx]) / entry_px - 1.0) * 100 - cost)
                break
            tk += RECHECK_MIN
    if trig:
        first = min(trig.values(), key=lambda x: x[0])
        return first[1], True
    # HOLD
    return float(fav[-1]) - cost, False


def _stats(rows, notional, hours):
    n = len(rows)
    if not n:
        return {"n": 0}
    nets = [r["net"] for r in rows]
    wins = [x for x in nets if x > 0]; losses = [x for x in nets if x < 0]
    usd = notional * sum(nets) / 100
    longs = [r for r in rows if r["side"] > 0]; shorts = [r for r in rows if r["side"] < 0]
    ev = []
    for r in rows:
        ev.append((r["t"] + 5, 1)); ev.append((r["t"] + 5 + r["h"], -1))
    ev.sort(); c = mx = 0
    for _, d in ev:
        c += d; mx = max(mx, c)
    return {
        "n": n, "win": round(100 * len(wins) / n, 1),
        "total_usd": round(usd, 2), "perday_usd": round(usd / hours * 24, 2) if hours else 0,
        "avg_net": round(sum(nets) / n, 3), "median_net": round(st.median(nets), 3),
        "loss_tail_usd": round(notional * sum(losses) / 100, 2),
        "best": round(max(nets), 2), "worst": round(min(nets), 2),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else None,
        "long_n": len(longs), "long_win": round(100 * len([r for r in longs if r["net"] > 0]) / len(longs), 0) if longs else 0,
        "short_n": len(shorts), "short_win": round(100 * len([r for r in shorts if r["net"] > 0]) / len(shorts), 0) if shorts else 0,
        "early": sum(1 for r in rows if r.get("early")), "maxconc": mx,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--days", type=int, default=12, help="window length up to now (days)")
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    builds = spec["builds"]; default_ep = spec.get("exit_default") or {}
    book = spec.get("book") or {}; mc = int(book.get("maxConc", 12)); top = int(book.get("top", 12)); cd = int(book.get("cd", 20))
    notional = float(spec.get("notional", 15))

    # one live pool: the last --days up to now (no sealed windows)
    start = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(days=args.days)
    costs = json.loads(COSTS.read_text(encoding="utf-8"))["costs"]
    liquid = [norm_symbol(s) for s in json.loads(LIQUID.read_text(encoding="utf-8"))["symbols"]]
    trade = [s for s in liquid if s in costs]

    # score each distinct model once
    scored, bpd_by_model = {}, {}
    edge = None
    for b in builds:
        sim = b["sim"]; md = str(SIM_TO_DIR.get(sim, sim))
        if md in scored:
            continue
        if "v5" in sim:
            dset = DATASET_V5
        else:
            dset = DATASET_NOW if DATASET_NOW.exists() else DATASET_Y1
        print(f"scoring {sim} ({dset.parent.name})", flush=True)
        df = _score_model(Path(md), dset, trade, start)
        edge = df["base_time"].max() if edge is None else max(edge, df["base_time"].max())
        scored[md] = _legs(df)
        # best p_dir per (sym, t, side) for re-confirm
        bpd = {}
        for l in scored[md]:
            k = (l["sym"], l["bt"], l["side"])
            if k not in bpd or l["pd"] > bpd[k]:
                bpd[k] = l["pd"]
        bpd_by_model[md] = bpd

    # each build offers 1 position per (sym, scan); cross-dedup across builds (highest pd)
    best = {}
    for b in builds:
        bname = b.get("name") or b.get("sim")
        md = str(SIM_TO_DIR.get(b["sim"], b["sim"]))
        ban = set(b.get("banned", []))
        legs = [l for l in scored[md] if l["sym"] not in ban]
        sel = _apply(legs, b.get("levels", []))
        by_sym = {}
        for l in sel:
            k = (l["sym"], l["bt"])
            if k not in by_sym or l["pd"] > by_sym[k]["pd"]:
                by_sym[k] = l
        ep = b.get("exit_policy") or default_ep
        for l in by_sym.values():
            k = (l["sym"], l["bt"])
            if k not in best or l["pd"] > best[k]["pd"]:
                best[k] = {**l, "_md": md, "_ep": ep, "_bname": bname}

    # optional sub-window inside the live pool (epoch minutes from the UI)
    positions = list(best.values())
    fm, tm = spec.get("from_min"), spec.get("to_min")
    win_start, win_end = start, edge
    hours = (edge - start).total_seconds() / 3600.0
    if fm is not None and tm is not None:
        positions = [p for p in positions if fm <= p["bt"] <= tm]
        hours = (tm - fm) / 60.0
        win_start = pd.Timestamp(int(fm) * 60_000_000_000, tz="UTC")
        win_end = pd.Timestamp(int(tm) * 60_000_000_000, tz="UTC")
    taken = _live_book(positions, mc, top, cd)
    print(f"positions: {len(taken)} (after dedup+book)", flush=True)

    closes = {}
    def cs_for(s):
        if s not in closes:
            closes[s] = _close_series(s)
        return closes[s]

    recs = []
    for p in taken:
        cs = cs_for(p["sym"])
        if cs is None:
            continue
        t0, h, side, cost = p["bt"], p["h"], p["side"], p["thr"]
        et = pd.Timestamp(t0 * 60_000_000_000, tz="UTC") + pd.Timedelta(minutes=ENTRY_DELAY)
        ei = cs.index.searchsorted(et, side="right") - 1
        if ei < 0:
            continue
        entry_px = float(cs.iloc[ei])
        sgn = 1.0 if side > 0 else -1.0
        xi = cs.index.searchsorted(et + pd.Timedelta(minutes=h), side="right") - 1
        if xi < 0:
            continue
        hold_net = sgn * (float(cs.iloc[xi]) / entry_px - 1.0) * 100 - cost
        if p["_ep"]:
            enet, early = _exit_net(cs, t0, entry_px, h, side, cost, p["_ep"], bpd_by_model.get(p["_md"]), p["sym"])
            if enet is None:
                enet, early = hold_net, False
        else:
            enet, early = hold_net, False
        recs.append({"hold": hold_net, "exit": enet, "side": side, "t": t0, "h": h,
                     "early": early, "build": p.get("_bname", "?"), "day": _kyiv_day(t0)})

    hold_rows = [{"net": r["hold"], "side": r["side"], "t": r["t"], "h": r["h"]} for r in recs]
    exit_rows = [{"net": r["exit"], "side": r["side"], "t": r["t"], "h": r["h"], "early": r["early"]} for r in recs]

    # +/- of each model (build) AFTER cross-dedup, and a per-day breakdown
    def _lite(rs):
        n = len(rs)
        if not n:
            return {"n": 0, "win": 0.0, "hold_usd": 0.0, "exit_usd": 0.0, "early": 0}
        ew = sum(1 for r in rs if r["exit"] > 0)
        return {"n": n, "win": round(100 * ew / n, 1),
                "hold_usd": round(notional * sum(r["hold"] for r in rs) / 100, 2),
                "exit_usd": round(notional * sum(r["exit"] for r in rs) / 100, 2),
                "early": sum(1 for r in rs if r.get("early"))}

    by_build = {}
    for r in recs:
        by_build.setdefault(r["build"], []).append(r)
    entry_of = {(b.get("name") or b.get("sim")): _entry_desc(b.get("levels", [])) for b in builds}
    per_build = [dict(name=name, entry=entry_of.get(name, ""),
                      per_day=round(_lite(rs)["n"] / (hours / 24), 1) if hours else 0, **_lite(rs))
                 for name, rs in sorted(by_build.items(), key=lambda kv: -_lite(kv[1])["exit_usd"])]

    by_day = {}
    for r in recs:
        by_day.setdefault(r["day"], []).append(r)
    per_day = [dict(day=day, **_lite(rs)) for day, rs in sorted(by_day.items())]

    taken_n = len(recs)
    days = hours / 24 if hours else 0
    out = {
        "builds": len(builds), "book": {"maxConc": mc, "top": top, "cd": cd}, "notional": notional,
        "floor": FLOOR, "hours": round(hours, 1), "days": round(days, 1),
        "window_start": str(win_start)[:16], "window_end": str(win_end)[:16],
        "signals": len(positions), "taken": taken_n,
        "trades_per_day": round(taken_n / days, 1) if days else 0,
        "signals_per_day": round(len(positions) / days, 1) if days else 0,
        "hold": _stats(hold_rows, notional, hours),
        "exits": _stats(exit_rows, notional, hours),
        "per_build": per_build, "per_day": per_day,
        "exit_default": default_ep,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
