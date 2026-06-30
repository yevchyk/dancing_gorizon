"""Generate a batch of high-turnover Binance builds — 6 methodology approaches
per model — into a dedicated folder, with real probe-window summaries.

Each approach is a level-set in the SAME schema the explorer/run_hc_build use
(OR within a level, AND across levels). Summaries are computed through the live
book (slots/top/cooldown) so the saved cards already show engine-realistic
trades/winrate/$ over the last DAYS up to now (one live pool).

  python -m src.run_binance_gen_builds
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

from . import config as C
from .binance_fetcher import norm_symbol
from .markets import is_equity

BUILDS = C.CONFIGS_DIR / "builds"
DATASET_NOW = C.ROOT / "data" / "binance_now" / "dataset"
DATASET_Y1 = C.ROOT / "data" / "binance_y1" / "dataset"
DATASET = DATASET_NOW if DATASET_NOW.exists() else DATASET_Y1
COSTS = C.CONFIGS_DIR / "binance_costs.json"
LIQUID = C.CONFIGS_DIR / "binance_universe_liquid.json"
FOLDER = "🔥 макс-оборот"
FLOOR = 0.55
NOTIONAL = 15.0
DAYS = 12
BOOK = {"max_concurrent": 20, "top_per_scan": 15, "cooldown_min": 20}

MODELS = [
    ("binance d8", Path("models/binance_y1_d8")),
    ("binance d10", Path("models/binance_y1_d10")),
    ("binance d12", Path("models/binance_y1_d12")),
]

# (suffix, description, levels) — levels mirror explorer regulators exactly
APPROACHES = [
    ("RAW80", "p_dir>=0.80, всі горизонти, обидва боки — макс обсяг",
     [[{"regulator": "p_dir", "value": 0.80}]]),
    ("RAW85", "p_dir>=0.85 — якісніший хвіст",
     [[{"regulator": "p_dir", "value": 0.85}]]),
    ("SPREAD70", "spread(p_dir-p_opp)>=0.70",
     [[{"regulator": "spread", "value": 0.70}]]),
    ("bdw", "p_dir>=0.80 І p_opp<=0.05 — чистий напрямок",
     [[{"regulator": "p_dir", "value": 0.80}], [{"regulator": "p_opp", "value": 0.05}]]),
    ("SHORT80", "лише шорти, p_dir>=0.80 — наш едж",
     [[{"regulator": "side", "value": "short"}], [{"regulator": "p_dir", "value": 0.80}]]),
    ("LONG88", "лише лонги, p_dir>=0.88 — вища планка",
     [[{"regulator": "side", "value": "long"}], [{"regulator": "p_dir", "value": 0.88}]]),
]


def _seeds(mdir: Path):
    out = []
    for sub in sorted(mdir.iterdir()):
        if sub.is_dir() and (sub / "metrics.json").exists():
            u = CatBoostClassifier(); u.load_model(sub / "up.cbm")
            d = CatBoostClassifier(); d.load_model(sub / "down.cbm")
            out.append((u, d))
    return out


def _cond(leg: dict, c: dict) -> bool:
    r = c["regulator"]; v = c["value"]
    if r == "p_dir":  return leg["pd"] >= v
    if r == "p_opp":  return leg["po"] <= v
    if r == "spread": return leg["sp"] >= v
    if r == "side":   return leg["side"] > 0 if v == "long" else leg["side"] < 0
    raise ValueError(r)


def _apply(legs, levels):
    cur = legs
    for lvl in levels:
        cur = [l for l in cur if any(_cond(l, c) for c in lvl)]
    return cur


def _units(legs):
    """1 risk-unit per (sym, scan): net=mean of legs, pd=max, h=max — == explorer toUnits."""
    by = {}
    for l in legs:
        k = (l["sym"], l["t"])
        u = by.get(k)
        if u is None:
            by[k] = {"sym": l["sym"], "t": l["t"], "nets": [l["net"]], "pd": l["pd"], "h": l["h"]}
        else:
            u["nets"].append(l["net"]); u["pd"] = max(u["pd"], l["pd"]); u["h"] = max(u["h"], l["h"])
    out = []
    for u in by.values():
        out.append({"sym": u["sym"], "t": u["t"], "net": sum(u["nets"]) / len(u["nets"]),
                    "pd": u["pd"], "h": u["h"]})
    return out


def _live_book(units, mc, top, cd):
    """Port of explorer liveBook: per scan take top-N by p_dir into free slots."""
    su = sorted(units, key=lambda r: (r["t"], -r["pd"]))
    cool = {}; book = []; out = []; i = 0
    while i < len(su):
        t = su[i]["t"]
        book = [e for e in book if e[0] > t + 5]
        held = {e[1] for e in book}
        offered = 0
        while i < len(su) and su[i]["t"] == t:
            r = su[i]
            if offered < top and len(book) < mc and r["sym"] not in held \
                    and t + 5 >= cool.get(r["sym"], -1e18):
                exit_t = t + 5 + r["h"]
                book.append((exit_t, r["sym"])); held.add(r["sym"])
                cool[r["sym"]] = exit_t + cd; out.append(r); offered += 1
            i += 1
    return out


def _summary(rows, hours):
    n = len(rows)
    if not n:
        return {"trades": 0, "winrate_pct": 0, "avg_net_pct": 0, "total_usd": 0,
                "usd_per_day": 0, "max_concurrent": 0}
    wins = sum(1 for r in rows if r["net"] > 0)
    net = sum(r["net"] for r in rows)
    usd = NOTIONAL * net / 100.0
    # max concurrent via sweep
    ev = []
    for r in rows:
        ev.append((r["t"] + 5, 1)); ev.append((r["t"] + 5 + r["h"], -1))
    ev.sort()
    c = mx = 0
    for _, d in ev:
        c += d; mx = max(mx, c)
    return {"trades": n, "winrate_pct": round(100 * wins / n, 1),
            "avg_net_pct": round(net / n, 3), "total_usd": round(usd, 2),
            "usd_per_day": round(usd / hours * 24, 2) if hours else 0, "max_concurrent": mx}


def main() -> None:
    start = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(days=DAYS)
    hours = (pd.Timestamp.utcnow() - start).total_seconds() / 3600.0
    costs = json.loads(COSTS.read_text(encoding="utf-8"))["costs"]
    liquid = [norm_symbol(s) for s in json.loads(LIQUID.read_text(encoding="utf-8"))["symbols"]]
    trade = [s for s in liquid if s in costs]
    BUILDS.mkdir(parents=True, exist_ok=True)

    written = 0
    for sim, mdir in MODELS:
        feat = json.loads((mdir / "feature_names.json").read_text(encoding="utf-8"))
        seeds = _seeds(mdir)
        need = list(dict.fromkeys(feat + ["symbol", "base_time", "horizon_minutes", "ret_pct", "thr_pct"]))
        frames = []
        for s in trade:
            p = DATASET / f"{s}.parquet"
            if not p.exists():
                continue
            t = pq.read_table(p, columns=need, filters=[("base_time", ">=", start)])
            if t.num_rows:
                frames.append(t.to_pandas())
        df = pd.concat(frames, ignore_index=True)
        X = df[feat]
        up = np.mean([u.predict_proba(X, thread_count=6)[:, 1] for u, _ in seeds], axis=0)
        dn = np.mean([d.predict_proba(X, thread_count=6)[:, 1] for _, d in seeds], axis=0)
        long = up >= dn
        p_dir = np.where(long, up, dn); p_opp = np.where(long, dn, up)
        ret = df["ret_pct"].to_numpy(); side = np.where(long, 1, -1)
        net = np.where(long, ret, -ret) - df["thr_pct"].to_numpy()
        tmin = (pd.to_datetime(df["base_time"], utc=True).astype("int64") // 60_000_000_000).to_numpy()
        hz = df["horizon_minutes"].to_numpy(); sym = df["symbol"].to_numpy()
        keep = p_dir >= FLOOR
        legs = [{"sym": str(sym[i]), "t": int(tmin[i]), "h": int(hz[i]), "side": int(side[i]),
                 "pd": float(p_dir[i]), "po": float(p_opp[i]), "sp": float(p_dir[i] - p_opp[i]),
                 "net": float(net[i])} for i in range(len(df)) if keep[i]]
        print(f"{sim}: {len(legs)} legs (p_dir>={FLOOR})")

        for suffix, desc, levels in APPROACHES:
            sel = _apply(legs, levels)
            rows = _live_book(_units(sel), BOOK["max_concurrent"], BOOK["top_per_scan"], BOOK["cooldown_min"])
            summ = _summary(rows, hours)
            tag = sim.replace("binance ", "").replace(" (probe)", "")
            name = f"{tag} · {suffix}"
            cfg = {
                "name": name, "folder": FOLDER, "sim": sim, "mode": "book",
                "notional": NOTIONAL, "floor": FLOOR, "period": None,
                "reverse": False, "one_pos": False, "sizing": None,
                "book": dict(BOOK), "banned": [],
                "description": f"{desc} · авто-генерований макс-оборот білд",
                "levels": [{"conditions": lvl} for lvl in levels],
                "summary": summ,
            }
            (BUILDS / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            written += 1
            print(f"  {name:22} trades={summ['trades']:5} win={summ['winrate_pct']}% "
                  f"$/day={summ['usd_per_day']} maxconc={summ['max_concurrent']}")
    print(f"\nwrote {written} builds -> folder '{FOLDER}' in {BUILDS}")


if __name__ == "__main__":
    main()
