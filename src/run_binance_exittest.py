"""Backtest an exit/management policy for ONE explorer build on PROBE -> JSON.

Called by the webapp (/api/exittest). Reads a spec {sim, levels, exit_policy},
scores the model on the probe window, selects entries with the build's levels
(1 position per (symbol, scan) = highest-p_dir leg, like live), then compares
HOLD-to-horizon against each enabled exit rule and the combined policy.

Cost per position = the dataset's honest thr_pct (parity-verified). Prices from
the 1-min candle store. Window = the last --days up to now (one live pool, no
sealed exam) — the UI may narrow it further with from_min/to_min.

  python -m src.run_binance_exittest --spec spec.json --out exit_result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

from . import config as C
from .binance_fetcher import norm_symbol
from .hc_model_registry import SIM_TO_DIR
from .markets import is_equity

DATASET_NOW = C.ROOT / "data" / "binance_now" / "dataset"
DATASET_Y1 = C.ROOT / "data" / "binance_y1" / "dataset"
DATASET_V5 = C.ROOT / "data" / "binance_y1_v5" / "dataset"
COSTS = C.CONFIGS_DIR / "binance_costs.json"
LIQUID = C.CONFIGS_DIR / "binance_universe_liquid.json"
CANDLES = C.ROOT / "data" / "binance" / "candles"
FLOOR = 0.55
NOTIONAL = 15.0
ENTRY_DELAY = 5
RECHECK_MIN = 60  # dataset cadence


def _seeds(mdir: Path):
    out = []
    for sub in sorted(mdir.iterdir()):
        if sub.is_dir() and (sub / "metrics.json").exists():
            u = CatBoostClassifier(); u.load_model(sub / "up.cbm")
            d = CatBoostClassifier(); d.load_model(sub / "down.cbm")
            out.append((u, d))
    return out


def _close_series(sym: str):
    p = CANDLES / f"{sym}.parquet"
    if not p.exists():
        return None
    try:
        df = pq.read_table(p, columns=["timestamp", "close"]).to_pandas()
    except Exception:
        return None
    s = pd.Series(df["close"].to_numpy(dtype=float), index=pd.to_datetime(df["timestamp"], utc=True)).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _cond(leg, c):
    r = c.get("regulator") or c.get("reg"); v = c.get("value", c.get("val"))
    if r == "p_dir":    return leg["pd"] >= v
    if r == "p_dir_max": return leg["pd"] <= v
    if r == "p_dir_sides": return leg["pd"] >= v[0] if leg["side"] > 0 else leg["pd"] >= v[1]
    if r == "p_opp":    return leg["po"] <= v
    if r == "spread":   return leg["sp"] >= v
    if r == "hmin":     return leg["h"] >= v
    if r == "hmax":     return leg["h"] <= v
    if r == "horizon":  return leg["h"] in set(v or [])
    if r == "hour":     return leg["hod"] in set(v or [])
    if r == "wday":     return leg["wd"] in set(v or [])
    if r == "asset":    return v == "both" or (leg["eq"] == 1 if v == "equity" else leg["eq"] == 0)
    if r == "side":     return v == "both" or (leg["side"] > 0 if v == "long" else leg["side"] < 0)
    if r == "lean_min": return leg["lean"] >= v
    if r == "lean_max": return leg["lean"] <= v
    return True  # unknown -> don't block (exit-test is advisory)


def _apply(legs, levels):
    cur = legs
    for L in levels:
        conds = L.get("conditions") or L.get("conds") or []
        if conds:
            cur = [l for l in cur if any(_cond(l, c) for c in conds)]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hmin", type=int, default=20)
    ap.add_argument("--hmax", type=int, default=480)
    ap.add_argument("--days", type=int, default=12, help="window length up to now (days)")
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    sim = spec["sim"]; levels = spec.get("levels", []); ep = spec.get("exit_policy") or {}
    mdir = Path(SIM_TO_DIR.get(sim, sim))
    if "v5" in sim:
        dset = DATASET_V5
    else:
        dset = DATASET_NOW if DATASET_NOW.exists() else DATASET_Y1
    print(f"exit-test sim={sim} -> {mdir}  dataset={dset.parent.name}", flush=True)

    # one live pool: the last --days up to now (no sealed windows). Read the FULL
    # window (so re-confirm has its future scans) and narrow ENTRIES below via from/to.
    start = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(days=args.days)
    fm, tm = spec.get("from_min"), spec.get("to_min")
    win_start = start
    costs = json.loads(COSTS.read_text(encoding="utf-8"))["costs"]
    liquid = [norm_symbol(s) for s in json.loads(LIQUID.read_text(encoding="utf-8"))["symbols"]]
    trade = [s for s in liquid if s in costs]
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
    win_end = df["base_time"].max()
    hours = (win_end - win_start).total_seconds() / 3600.0
    print(f"scored: {len(df)} rows", flush=True)
    X = df[feat]
    up = np.mean([u.predict_proba(X, thread_count=6)[:, 1] for u, _ in seeds], axis=0)
    dn = np.mean([d.predict_proba(X, thread_count=6)[:, 1] for _, d in seeds], axis=0)
    df["up"], df["dn"] = up, dn

    # Kyiv hour/weekday for hour/wday filters
    em = (df["base_time"].astype("int64") // 60_000_000_000).to_numpy()
    hod = ((em + 180) // 60) % 24
    wd = ((em + 180) // 1440 + 4) % 7

    # build legs (both sides), with lean per scan
    long = df["up"].to_numpy() >= df["dn"].to_numpy()
    legs = []
    sym = df["symbol"].to_numpy(); hz = df["horizon_minutes"].to_numpy()
    upn, dnn = df["up"].to_numpy(), df["dn"].to_numpy()
    ret = df["ret_pct"].to_numpy(); thr = df["thr_pct"].to_numpy()
    bt = df["base_time"].to_numpy()
    for i in range(len(df)):
        for sd, pd_, po in ((1, upn[i], dnn[i]), (-1, dnn[i], upn[i])):
            if pd_ < FLOOR:
                continue
            s = str(sym[i])
            legs.append({"sym": s, "bt": bt[i], "h": int(hz[i]), "side": sd, "pd": float(pd_),
                         "po": float(po), "sp": float(pd_ - po), "hod": int(hod[i]), "wd": int(wd[i]),
                         "eq": 1 if is_equity(s) else 0, "lean": 0.0, "ret": float(ret[i]), "thr": float(thr[i])})
    sel = _apply(legs, levels)
    # 1 position per (symbol, scan): highest p_dir leg (matches live open)
    best = {}
    for l in sel:
        k = (l["sym"], l["bt"])
        if k not in best or l["pd"] > best[k]["pd"]:
            best[k] = l
    entries = list(best.values())
    if fm is not None and tm is not None:
        emin = lambda ts: int(pd.Timestamp(ts).value // 60_000_000_000)
        entries = [e for e in entries if fm <= emin(e["bt"]) <= tm]
        hours = (tm - fm) / 60.0
        win_start = pd.Timestamp(int(fm) * 60_000_000_000, tz="UTC")
        win_end = pd.Timestamp(int(tm) * 60_000_000_000, tz="UTC")
    print(f"entries after levels (1/sym/scan): {len(entries)}", flush=True)

    closes = {}
    def cs_for(s):
        if s not in closes:
            closes[s] = _close_series(s)
        return closes[s]

    # accumulators
    def mk(): return {"net": 0.0, "wins": 0, "n": 0, "early": 0, "tail": 0.0}
    modes = {"HOLD": mk()}
    enabled = []
    if ep.get("trail"): enabled.append("trail")
    if ep.get("tp"): enabled.append("tp")
    if ep.get("reconfirm"): enabled.append("reconfirm")
    if ep.get("crash"): enabled.append("crash")
    for m in enabled:
        modes[m] = mk()
    if len(enabled) > 1:
        modes["combined"] = mk()

    # per (sym, scan) best p_dir over band for re-confirm
    bpd = {}
    if "reconfirm" in enabled:
        for l in legs:
            k = (l["sym"], l["bt"], l["side"])
            if k not in bpd or l["pd"] > bpd[k]:
                bpd[k] = l["pd"]

    def add(mode, net):
        a = modes[mode]; a["net"] += net; a["n"] += 1; a["wins"] += net > 0
        if net < 0: a["tail"] += net

    for e in entries:
        s = e["sym"]; t0 = pd.Timestamp(e["bt"]); h = e["h"]; sgn = 1.0 if e["side"] > 0 else -1.0
        cost = e["thr"]
        cs = cs_for(s)
        if cs is None: continue
        et = t0 + pd.Timedelta(minutes=ENTRY_DELAY); xt = et + pd.Timedelta(minutes=h)
        ep_idx = cs.index.searchsorted(et, side="right") - 1
        xp_idx = cs.index.searchsorted(xt, side="right") - 1
        if ep_idx < 0 or xp_idx < 0: continue
        entry_px = float(cs.iloc[ep_idx])
        hold_net = sgn * (float(cs.iloc[xp_idx]) / entry_px - 1.0) * 100 - cost
        add("HOLD", hold_net)

        path = cs[(cs.index >= et) & (cs.index <= xt)]
        fav = (sgn * (path.to_numpy() / entry_px - 1.0) * 100)
        tmins = ((path.index.astype("int64") // 60_000_000_000) - (et.value // 60_000_000_000)).to_numpy()

        triggers = {}  # mode -> (minute, net)
        if "tp" in enabled:
            tp = ep["tp"]["pct"]
            idx = np.where(fav >= tp)[0]
            if len(idx): triggers["tp"] = (tmins[idx[0]], tp - cost)
        if "crash" in enabled:
            cpct = ep["crash"]["pct"]
            idx = np.where(-fav >= cpct)[0]
            if len(idx): triggers["crash"] = (tmins[idx[0]], -cpct - cost)
        if "trail" in enabled:
            arm = ep["trail"]["arm"]; give = ep["trail"]["give"]; peak = -1e9
            for j in range(len(fav)):
                if fav[j] > peak: peak = fav[j]
                if peak >= arm and fav[j] <= peak - give:
                    triggers["trail"] = (tmins[j], (peak - give) - cost); break
        if "reconfirm" in enabled:
            floor = ep["reconfirm"]["floor"]; tk = RECHECK_MIN
            while tk < h:
                k = (s, t0 + pd.Timedelta(minutes=tk), e["side"])
                if k in bpd and bpd[k] < floor:
                    pidx = cs.index.searchsorted(t0 + pd.Timedelta(minutes=tk + ENTRY_DELAY), side="right") - 1
                    if pidx >= 0:
                        triggers["reconfirm"] = (tk, sgn * (float(cs.iloc[pidx]) / entry_px - 1.0) * 100 - cost)
                    break
                tk += RECHECK_MIN

        for m in enabled:
            add(m, triggers[m][1] if m in triggers else hold_net)
            if m in triggers: modes[m]["early"] += 1
        if "combined" in modes:
            if triggers:
                first = min(triggers.values(), key=lambda x: x[0])
                add("combined", first[1]); modes["combined"]["early"] += 1
            else:
                add("combined", hold_net)

    def fin(a):
        n = a["n"] or 1
        return {"n": a["n"], "win": round(100 * a["wins"] / n, 1),
                "total_usd": round(NOTIONAL * a["net"] / 100, 2),
                "tail_usd": round(NOTIONAL * a["tail"] / 100, 2), "early": a["early"]}
    order = ["HOLD"] + enabled + (["combined"] if "combined" in modes else [])
    entry_desc = " · ".join(f"{(L.get('conditions') or [{}])[0].get('regulator','?')}"
                            f"{(L.get('conditions') or [{}])[0].get('value','')}" for L in levels) or "p_dir>=0.55"
    out = {"sim": sim, "entry": entry_desc, "notional": NOTIONAL,
           "hours": round(hours, 1), "days": round(hours / 24.0, 1),
           "window_start": str(win_start)[:16], "window_end": str(win_end)[:16],
           "rows": [dict(mode=m, **fin(modes[m])) for m in order]}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
