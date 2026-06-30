"""Export Binance model legs to the browser sim explorer — ONE live data pool.

Single data source: the explorer's data.js covers the last --days up to NOW.
There is no sealed exam / probe / cutoffs split anymore — the test period is a
plain holdout chosen at TRAIN time, never a permanent fixture in this pipeline.

Window = [now_utc - days, latest labelled base_time]. Universe = liquid AND
trusted-cost. net per leg = signed ret - thr_pct, where thr_pct is the row's
honest per-symbol cost incl funding — bit-consistent with the labels.

Reads the freshest dataset available: data/binance_now/dataset (rebuilt to "now"
by the panel's "докачати + ребілд до зараз" button) when present, otherwise the
static data/binance_y1/dataset. v5 models read their v5 dataset.

Scores every COMPLETED seed (metrics.json present) under each model dir and
averages probabilities. CPU-only with a bounded thread count so a concurrent GPU
training run is not starved.

  # one model, append to data.js:
  python -m src.run_binance_export --model-dir models/binance_y1_d8 --name "binance d8"
  # the standard model set, fresh data.js up to now:
  python -m src.run_binance_export --all --fresh
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
from .markets import is_equity

OUT = C.ROOT / "reports" / "sim_explorer" / "data.js"
DATASET_NOW = C.ROOT / "data" / "binance_now" / "dataset"
DATASET_Y1 = C.ROOT / "data" / "binance_y1" / "dataset"
DATASET_V5 = C.ROOT / "data" / "binance_y1_v5" / "dataset"
COSTS = C.CONFIGS_DIR / "binance_costs.json"
LIQUID = C.CONFIGS_DIR / "binance_universe_liquid.json"

# the standard model set exported by --all (names carry NO window suffix now)
STD_MODELS = [
    ("binance d8", Path("models/binance_y1_d8")),
    ("binance d10", Path("models/binance_y1_d10")),
    ("binance d12", Path("models/binance_y1_d12")),
    ("binance d12 20k", Path("models/binance_y1_d12_it20k")),
]


def pick_dataset(name: str, explicit: Path | None) -> Path:
    """Freshest dataset matching the model schema; binance_now beats the static y1."""
    if explicit is not None:
        return explicit
    if "v5" in name:
        return DATASET_V5
    return DATASET_NOW if DATASET_NOW.exists() else DATASET_Y1


def load_completed_seeds(model_dir: Path) -> list[tuple[str, CatBoostClassifier, CatBoostClassifier]]:
    seeds = []
    for sub in sorted(model_dir.iterdir()):
        if sub.is_dir() and (sub / "metrics.json").exists():
            u = CatBoostClassifier()
            u.load_model(sub / "up.cbm")
            d = CatBoostClassifier()
            d.load_model(sub / "down.cbm")
            seeds.append((sub.name, u, d))
    if not seeds:
        raise SystemExit(f"no completed seeds (metrics.json) under {model_dir}")
    return seeds


def export_model(name: str, model_dir: Path, dataset: Path, start: pd.Timestamp,
                 trade_syms: list[str], costs: dict, floor: float, threads: int,
                 sims: dict, syms_arr: list, idx_of: dict,
                 cutoff_min: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp, set]:
    """Score one model over [start, now], append its legs to `sims`. Returns window bounds + horizons."""
    feat = json.loads((model_dir / "feature_names.json").read_text(encoding="utf-8"))
    need = list(dict.fromkeys(feat + ["symbol", "base_time", "horizon_minutes", "ret_pct", "thr_pct"]))
    frames = []
    for s in trade_syms:
        p = dataset / f"{s}.parquet"
        if not p.exists():
            continue
        t = pq.read_table(p, columns=need, filters=[("base_time", ">=", start)])
        if t.num_rows:
            frames.append(t.to_pandas())
    if not frames:
        raise SystemExit(f"no rows >= {start} in {dataset} — rebuild the dataset to now first")
    df = pd.concat(frames, ignore_index=True)
    df["base_time"] = pd.to_datetime(df["base_time"], utc=True)
    print(f"  {name}: {len(df)} rows from {df['symbol'].nunique()} symbols ({dataset.parent.name})")

    seeds = load_completed_seeds(model_dir)
    X = df[feat]
    up = np.mean([u.predict_proba(X, thread_count=threads)[:, 1] for _, u, _ in seeds], axis=0)
    dn = np.mean([d.predict_proba(X, thread_count=threads)[:, 1] for _, _, d in seeds], axis=0)

    long = up >= dn
    p_dir = np.where(long, up, dn)
    p_opp = np.where(long, dn, up)
    ret = df["ret_pct"].to_numpy()
    side = np.where(long, 1, -1)
    net = np.where(long, ret, -ret) - df["thr_pct"].to_numpy()

    keep = p_dir >= float(floor)
    sub = df[keep]
    p_dir, p_opp, net, side = p_dir[keep], p_opp[keep], net[keep], side[keep]
    syms = sub["symbol"].to_numpy()
    tmin = (sub["base_time"].astype("int64") // 60_000_000_000).to_numpy()
    hz = sub["horizon_minutes"].to_numpy()

    def sym_idx(s: str) -> int:
        if s not in idx_of:
            idx_of[s] = len(syms_arr)
            syms_arr.append(s)
        return idx_of[s]

    legs = []
    for i in range(len(sub)):
        s = str(syms[i])
        leg = [sym_idx(s), int(tmin[i]), int(hz[i]), int(side[i]),
               round(float(p_dir[i]), 4), round(float(p_opp[i]), 4),
               round(float(net[i]), 3), 1 if is_equity(s) else 0]
        if cutoff_min is not None:                      # 9th field = holdout (base_time >= train cutoff)
            leg.append(1 if int(tmin[i]) >= cutoff_min else 0)
        legs.append(leg)
    sims[name] = {"legs": legs, "thr_in_net": True}

    for lo in (0.7, 0.8, 0.85, 0.9):
        m = p_dir >= lo
        if m.sum():
            print(f"    p_dir>={lo:.2f}: n={int(m.sum()):6d}  win(net>0)={float((net[m] > 0).mean()):.1%}  avg_net={float(net[m].mean()):+.3f}%")
    return df["base_time"].min(), df["base_time"].max(), {int(x) for x in np.unique(hz)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("models/binance_y1_d8"))
    ap.add_argument("--name", default="binance d8")
    ap.add_argument("--all", action="store_true", help="export the standard model set (ignores --model-dir/--name)")
    ap.add_argument("--fresh", action="store_true", help="start a clean data.js (drop any existing sims)")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="override the dataset shards (default: binance_now if present, else binance_y1; v5 models use the v5 dataset)")
    ap.add_argument("--days", type=int, default=12, help="window length up to now (days)")
    ap.add_argument("--floor", type=float, default=0.55)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--holdout-cutoff", default=None,
                    help="ISO UTC; mark legs with base_time >= it as holdout (9th leg field) + cutoff_min in meta (holdout view)")
    args = ap.parse_args()

    start = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(days=args.days)
    cutoff_min = (int(pd.Timestamp(args.holdout_cutoff).value // 60_000_000_000)
                  if args.holdout_cutoff else None)
    print(f"LIVE window: base_time >= {start} (last {args.days}d up to now)"
          + (f"  | HOLDOUT cutoff {args.holdout_cutoff}" if cutoff_min else ""))

    costs = json.loads(COSTS.read_text(encoding="utf-8"))["costs"]
    liquid = [norm_symbol(s) for s in json.loads(LIQUID.read_text(encoding="utf-8"))["symbols"]]
    trade_syms = [s for s in liquid if s in costs]
    print(f"tradeable universe = liquid AND trusted-cost = {len(trade_syms)} symbols")

    if args.fresh:
        syms_arr, sims, old_liquid, old_cost = [], {}, [], {}
    else:
        syms_arr, sims, old_liquid, old_cost = _load_existing()
    idx_of = {s: i for i, s in enumerate(syms_arr)}

    models = STD_MODELS if args.all else [(args.name, args.model_dir)]
    win_min = win_max = None
    horizons_seen: set = set()
    for name, mdir in models:
        if not mdir.exists():
            print(f"  skip {name}: no model dir {mdir}")
            continue
        ds = pick_dataset(name, args.dataset)
        lo, hi, hz = export_model(name, mdir, ds, start, trade_syms, costs,
                                  args.floor, args.threads, sims, syms_arr, idx_of, cutoff_min)
        win_min = lo if win_min is None else min(win_min, lo)
        win_max = hi if win_max is None else max(win_max, hi)
        horizons_seen |= hz
    if win_min is None:
        raise SystemExit("no models exported")

    used_costs = [costs[s] for s in trade_syms if s in idx_of]
    hours = (win_max - win_min).total_seconds() / 3600.0
    meta = {"cost_pct": round(float(np.median(used_costs)), 3) if used_costs else 0.126,
            "cost_model": "binance per-symbol thr_pct (fee+spread+impact+funding)",
            "window_start": str(win_min), "window_end": str(win_max),
            "hours": hours, "horizons": sorted(horizons_seen), "floor": args.floor,
            "cutoff_min": cutoff_min, "tz": "Europe/Kiev"}

    liquid_out = sorted(set(old_liquid) | {s for s in trade_syms if s in idx_of})
    cost_out = dict(old_cost)
    cost_out.update({s: round(float(costs[s]), 3) for s in trade_syms if s in idx_of})

    if cutoff_min is not None:                       # holdout view -> SEPARATE file, never clobbers the live data.js
        out_hold = OUT.parent / "data_holdout.js"
        bundle = {"SIMS": sims, "SYMS": syms_arr, "SIM_META": meta, "COST": cost_out, "LIQUID": liquid_out}
        out_hold.parent.mkdir(parents=True, exist_ok=True)
        out_hold.write_text("window.HOLD=" + json.dumps(bundle, separators=(",", ":")) + ";\n", encoding="utf-8")
        print(f"wrote {out_hold} (window.HOLD bundle; data.js UNTOUCHED)  sims={list(sims)}  "
              f"window {meta['window_start']} .. {meta['window_end']}  size={out_hold.stat().st_size // 1024}KB")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("window.SIM_META=" + json.dumps(meta) + ";\n")
        f.write("window.SYMS=" + json.dumps(syms_arr) + ";\n")
        f.write("window.SIMS=" + json.dumps(sims, separators=(",", ":")) + ";\n")
        f.write("window.LIQUID=" + json.dumps(liquid_out) + ";\n")
        f.write("window.COST=" + json.dumps(cost_out, separators=(",", ":")) + ";\n")

    print(f"wrote {OUT}  sims={list(sims)}  window {meta['window_start']} .. {meta['window_end']} "
          f"({hours/24:.1f}d)  syms={len(syms_arr)} size={OUT.stat().st_size // 1024}KB")


def _load_existing() -> tuple[list, dict, list, dict]:
    """Parse existing data.js -> (SYMS, SIMS, LIQUID, COST); empties if missing."""
    if not OUT.exists():
        return [], {}, [], {}
    syms, sims, liq, cost = [], {}, [], {}
    try:
        for line in OUT.read_text(encoding="utf-8").splitlines():
            line = line.strip().rstrip(";")
            for key, slot in (("window.SYMS=", "syms"), ("window.SIMS=", "sims"),
                              ("window.LIQUID=", "liq"), ("window.COST=", "cost")):
                if line.startswith(key):
                    val = json.loads(line[len(key):])
                    if slot == "syms":
                        syms = val
                    elif slot == "sims":
                        sims = val
                    elif slot == "liq":
                        liq = val
                    else:
                        cost = val
    except Exception:
        return [], {}, [], {}
    return list(syms), dict(sims), list(liq), dict(cost)


if __name__ == "__main__":
    main()
