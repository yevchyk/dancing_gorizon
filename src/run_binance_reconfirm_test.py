"""Offline backtest of the RE-CONFIRM exit on the PROBE window.

Hypothesis (user, 2026-06-13): the model over-sees upside; when a long starts
falling it does NOT bounce. So re-scoring an open position and CLOSING it early
once the model's conviction (p_dir for its side) decays should cut the fat left
tail (SIREN −43%, STG −12%) without killing the winners.

This proves it on history BEFORE any live wiring (real-money discipline):
  baseline  = hold every signal to its horizon (the dataset's own net).
  reconfirm = walk the hold hour-by-hour; at each step re-score the symbol and
              if best p_dir(side) < recheck_floor, EXIT at that minute's close
              (signed return entry->exit minus the same per-symbol RT cost).

Dataset snapshots are hourly, so re-checks land on the 60-min grid; prices come
from the 1-min candle store. Window = the last --days up to now (one live pool).

  python -m src.run_binance_reconfirm_test --model-dir models/binance_y1_d12 --side long
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

DATASET_NOW = C.ROOT / "data" / "binance_now" / "dataset"
DATASET_Y1 = C.ROOT / "data" / "binance_y1" / "dataset"
COSTS = C.CONFIGS_DIR / "binance_costs.json"
LIQUID = C.CONFIGS_DIR / "binance_universe_liquid.json"
CANDLES = C.ROOT / "data" / "binance" / "candles"
ENTRY_DELAY = 5  # EXEC_ENTRY_DELAY_MIN


def _seeds(mdir: Path):
    out = []
    for sub in sorted(mdir.iterdir()):
        if sub.is_dir() and (sub / "metrics.json").exists():
            u = CatBoostClassifier(); u.load_model(sub / "up.cbm")
            d = CatBoostClassifier(); d.load_model(sub / "down.cbm")
            out.append((u, d))
    return out


def _close_series(sym: str) -> pd.Series | None:
    p = CANDLES / f"{sym}.parquet"
    if not p.exists():
        return None
    try:
        df = pq.read_table(p, columns=["timestamp", "close"]).to_pandas()
    except Exception as e:  # corrupt parquet footer etc. — skip the symbol
        print(f"  WARN skip {sym}: {type(e).__name__}")
        return None
    ts = pd.to_datetime(df["timestamp"], utc=True)
    s = pd.Series(df["close"].to_numpy(dtype=float), index=ts).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _price_at(s: pd.Series, t: pd.Timestamp) -> float | None:
    # last close at or before t (1-min grid); None if before data starts
    idx = s.index.searchsorted(t, side="right") - 1
    if idx < 0:
        return None
    return float(s.iloc[idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("models/binance_y1_d12"))
    ap.add_argument("--side", choices=["long", "short", "both"], default="long")
    ap.add_argument("--entry-floor", type=float, default=0.85)
    ap.add_argument("--recheck-floors", default="0.55,0.65,0.75,0.80")
    ap.add_argument("--recheck-min", type=int, default=60, help="re-check cadence (dataset is hourly)")
    ap.add_argument("--crash-pcts", default="4,6,8,12", help="crash-stop adverse %% levels (1-min path)")
    ap.add_argument("--tp-pcts", default="2,3,5", help="take-profit lock levels (favorable %%)")
    ap.add_argument("--trails", default="2:1,3:1.5", help="trailing arm:give pairs, comma-sep")
    ap.add_argument("--hmin", type=int, default=95)
    ap.add_argument("--hmax", type=int, default=240)
    ap.add_argument("--days", type=int, default=12, help="window length up to now (days)")
    args = ap.parse_args()

    start = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(days=args.days)
    dataset = DATASET_NOW if DATASET_NOW.exists() else DATASET_Y1
    costs = json.loads(COSTS.read_text(encoding="utf-8"))["costs"]
    liquid = [norm_symbol(s) for s in json.loads(LIQUID.read_text(encoding="utf-8"))["symbols"]]
    trade = [s for s in liquid if s in costs]
    feat = json.loads((args.model_dir / "feature_names.json").read_text(encoding="utf-8"))
    seeds = _seeds(args.model_dir)
    print(f"window >= {start} ({dataset.parent.name})  model={args.model_dir.name} seeds={len(seeds)} side={args.side}")

    need = list(dict.fromkeys(feat + ["symbol", "base_time", "horizon_minutes", "ret_pct", "thr_pct"]))
    frames = []
    for s in trade:
        p = dataset / f"{s}.parquet"
        if not p.exists():
            continue
        t = pq.read_table(p, columns=need, filters=[("base_time", ">=", start)])
        if t.num_rows:
            frames.append(t.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df["base_time"] = pd.to_datetime(df["base_time"], utc=True)
    X = df[feat]
    up = np.mean([u.predict_proba(X, thread_count=6)[:, 1] for u, _ in seeds], axis=0)
    dn = np.mean([d.predict_proba(X, thread_count=6)[:, 1] for _, d in seeds], axis=0)
    df["up"], df["dn"] = up, dn
    print(f"scored {len(df)} probe rows, {df['symbol'].nunique()} symbols")

    # per (symbol, base_time): best long/short p_dir over the build's horizon band
    band = df[(df["horizon_minutes"] >= args.hmin) & (df["horizon_minutes"] <= args.hmax)]
    g = band.groupby(["symbol", "base_time"])
    best_long = g["up"].max()
    best_short = g["dn"].max()

    sides = ["long", "short"] if args.side == "both" else [args.side]
    # entries = signals clearing entry floor on the chosen side(s), within the band
    rows = band.copy()
    rows["pd_side_long"] = rows["up"]
    rows["pd_side_short"] = rows["dn"]

    closes: dict[str, pd.Series] = {}
    def closes_for(sym):
        if sym not in closes:
            closes[sym] = _close_series(sym)
        return closes[sym]

    def _mk(): return {"net": 0.0, "wins": 0, "n": 0, "early": 0, "tail": 0.0}
    _off = lambda s: (not s) or s.strip().lower() in ("none", "off", "-")
    floors = [] if _off(args.recheck_floors) else [float(x) for x in args.recheck_floors.split(",")]
    crashes = [] if _off(args.crash_pcts) else [float(x) for x in args.crash_pcts.split(",")]
    tps = [] if _off(args.tp_pcts) else [float(x) for x in args.tp_pcts.split(",")]
    trails = [] if _off(args.trails) else [tuple(float(y) for y in p.split(":")) for p in args.trails.split(",")]
    results = {f: _mk() for f in floors}
    cresults = {c: _mk() for c in crashes}
    tpresults = {tp: _mk() for tp in tps}
    trresults = {f"{a}/{g}": _mk() for (a, g) in trails}
    base = {"net": 0.0, "wins": 0, "n": 0, "tail": 0.0}
    parity = {"sum_abs_diff": 0.0, "n": 0}  # candle base_net vs dataset (ret_pct - thr_pct)

    # iterate entries
    for side in sides:
        sel = rows[rows[f"pd_side_{side}"] >= args.entry_floor]
        for r in sel.itertuples(index=False):
            sym = r.symbol; t0 = r.base_time; h = int(r.horizon_minutes)
            cost = float(r.thr_pct)  # dataset's honest per-symbol cost (already %, == labels)
            cs = closes_for(sym)
            if cs is None:
                continue
            entry_t = t0 + pd.Timedelta(minutes=ENTRY_DELAY)
            entry_px = _price_at(cs, entry_t)
            exit_t = entry_t + pd.Timedelta(minutes=h)
            exit_px = _price_at(cs, exit_t)
            if not entry_px or not exit_px:
                continue
            sgn = 1.0 if side == "long" else -1.0
            base_net = sgn * (exit_px / entry_px - 1.0) * 100 - cost
            # PARITY: my candle net must match the dataset's own (signed ret_pct - thr_pct)
            ds_net = sgn * float(r.ret_pct) - float(r.thr_pct)
            parity["sum_abs_diff"] += abs(base_net - ds_net); parity["n"] += 1
            base["net"] += base_net; base["n"] += 1; base["wins"] += base_net > 0
            if base_net < 0:
                base["tail"] += base_net

            # re-confirm: walk hourly; exit when conviction decays
            bb = best_long if side == "long" else best_short
            for f in floors:
                exited = False
                tk = t0 + pd.Timedelta(minutes=args.recheck_min)
                while tk < t0 + pd.Timedelta(minutes=h):
                    key = (sym, tk)
                    if key in bb.index:
                        if float(bb.loc[key]) < f:
                            px = _price_at(cs, tk + pd.Timedelta(minutes=ENTRY_DELAY))
                            if px:
                                net = sgn * (px / entry_px - 1.0) * 100 - cost
                                results[f]["net"] += net; results[f]["n"] += 1
                                results[f]["wins"] += net > 0; results[f]["early"] += 1
                                if net < 0:
                                    results[f]["tail"] += net
                                exited = True
                            break
                    tk += pd.Timedelta(minutes=args.recheck_min)
                if not exited:
                    results[f]["net"] += base_net; results[f]["n"] += 1
                    results[f]["wins"] += base_net > 0
                    if base_net < 0:
                        results[f]["tail"] += base_net

            # ---- path-based policies (crash / take-profit / trailing) ----
            if crashes or tps or trails:
                path = cs[(cs.index >= entry_t) & (cs.index <= exit_t)]
                fav = sgn * (path / entry_px - 1.0) * 100         # favorable move %
                adverse = -fav                                    # adverse move %
            # crash-stop: exit at first adverse >= crash_pct
            for c in crashes:
                net = (-c - cost) if (adverse >= c).any() else base_net
                if (adverse >= c).any():
                    cresults[c]["early"] += 1
                cresults[c]["net"] += net; cresults[c]["n"] += 1; cresults[c]["wins"] += net > 0
                if net < 0:
                    cresults[c]["tail"] += net
            # take-profit: lock the win at first favorable >= tp
            for tp in tps:
                net = (tp - cost) if (fav >= tp).any() else base_net
                if (fav >= tp).any():
                    tpresults[tp]["early"] += 1
                tpresults[tp]["net"] += net; tpresults[tp]["n"] += 1; tpresults[tp]["wins"] += net > 0
                if net < 0:
                    tpresults[tp]["tail"] += net
            # trailing: once favorable peak >= arm, exit if it gives back `give` from peak
            for (arm, give) in trails:
                fv = fav.to_numpy(); peak = -1e9; exit_net = None
                for v in fv:
                    if v > peak:
                        peak = v
                    if peak >= arm and v <= peak - give:
                        exit_net = (peak - give) - cost; break
                net = exit_net if exit_net is not None else base_net
                k = f"{arm}/{give}"
                if exit_net is not None:
                    trresults[k]["early"] += 1
                trresults[k]["net"] += net; trresults[k]["n"] += 1; trresults[k]["wins"] += net > 0
                if net < 0:
                    trresults[k]["tail"] += net

    n = base["n"]
    if parity["n"]:
        md = parity["sum_abs_diff"] / parity["n"]
        flag = "OK" if md < 0.15 else "BUG? candle price != dataset"
        print(f"\nPARITY candle-net vs dataset-net: mean|diff|={md:.4f}%  [{flag}]")
    print(f"\nentries (side={args.side}, p_dir>={args.entry_floor}, h{args.hmin}-{args.hmax}): {n}")
    print(f"{'mode':>16}  {'n':>4} {'win%':>5} {'avgnet%':>8} {'total%':>8} {'losstail%':>9} {'early':>6}")
    print(f"{'HOLD (baseline)':>16}  {n:>4} {100*base['wins']/n:>5.1f} {base['net']/n:>8.3f} "
          f"{base['net']:>8.1f} {base['tail']:>9.1f} {'-':>6}")
    for f in floors:
        rr = results[f]; nn = rr["n"]
        print(f"{'recheck<'+str(f):>16}  {nn:>4} {100*rr['wins']/nn:>5.1f} {rr['net']/nn:>8.3f} "
              f"{rr['net']:>8.1f} {rr['tail']:>9.1f} {rr['early']:>6}")
    for c in crashes:
        rr = cresults[c]; nn = rr["n"]
        print(f"{'crash-stop -'+str(c)+'%':>16}  {nn:>4} {100*rr['wins']/nn:>5.1f} {rr['net']/nn:>8.3f} "
              f"{rr['net']:>8.1f} {rr['tail']:>9.1f} {rr['early']:>6}")
    for tp in tps:
        rr = tpresults[tp]; nn = rr["n"]
        print(f"{'take-profit +'+str(tp)+'%':>16}  {nn:>4} {100*rr['wins']/nn:>5.1f} {rr['net']/nn:>8.3f} "
              f"{rr['net']:>8.1f} {rr['tail']:>9.1f} {rr['early']:>6}")
    for (a, g) in trails:
        rr = trresults[f"{a}/{g}"]; nn = rr["n"]
        print(f"{'trail '+str(a)+'/'+str(g):>16}  {nn:>4} {100*rr['wins']/nn:>5.1f} {rr['net']/nn:>8.3f} "
              f"{rr['net']:>8.1f} {rr['tail']:>9.1f} {rr['early']:>6}")


if __name__ == "__main__":
    main()
