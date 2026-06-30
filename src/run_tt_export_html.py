"""Export the ТТ curve model -> `window.TT` in data.js for the dedicated 'ТТ' tab.

ТТ predicts a forward CURVE, not P(up)/P(down) — so it gets its OWN tab/data
instead of being wedged into the prob-model explorer. Scores the curve over
data/tt_now/dataset (covers up to NOW; the training set tt_curve stays sterile).
Each leg carries a `holdout` flag (base_time >= the training cutoff) so the tab
splits IN-SAMPLE vs the UNSEEN HOLDOUT (= the user's OOS test). Conviction =
|predicted move|, per-horizon percentile in [0,1] (the tab's gate). net =
side*realized_% − per-instrument Binance cost. Live-zone horizons only (≥30).

Preserves the other (binance) sims in data.js; drops any legacy 'ТТ крива' sim.

  python -m src.run_tt_export_html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from . import config as C
from .hc.data import load_dataset, to_ns
from .markets import is_equity

OUT = C.ROOT / "reports" / "sim_explorer" / "data.js"
COSTS = C.ROOT / "configs" / "binance_costs.json"
TRAIN_SUMMARY = C.ROOT / "data" / "tt_curve" / "dataset_summary.json"
LIVE_HORIZONS = (30, 45, 60, 90, 120, 180, 240, 270, 300)
MEDIAN_COST = 0.126
PRESERVE = ("window.SIM_META", "window.SYMS", "window.LIQUID", "window.COST")


def _existing_lines() -> dict[str, str]:
    out: dict[str, str] = {}
    if not OUT.exists():
        return out
    for line in OUT.read_text(encoding="utf-8").splitlines():
        s = line.strip().rstrip(";")
        for key in PRESERVE + ("window.SIMS", "window.TT"):
            if s.startswith(key + "="):
                out[key] = s[len(key) + 1:]
    return out


def _costs() -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(COSTS.read_text())["costs"].items()}
    except Exception:
        return {}


def _train_cutoff_min() -> int | None:
    try:
        return int(pd.Timestamp(json.loads(TRAIN_SUMMARY.read_text())["cutoff"]).value // 60_000_000_000)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("models/tt_curve"))
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/tt_now/dataset"))
    ap.add_argument("--hours", type=float, default=480.0)
    ap.add_argument("--horizons", default=",".join(str(h) for h in LIVE_HORIZONS))
    ap.add_argument("--holdout-cutoff", default=None, help="UTC ISO; rows >= it are holdout. default=tt_curve cutoff")
    a = ap.parse_args()

    md = a.model_dir
    if not (md / "feature_names.json").exists():
        raise SystemExit(f"no ТТ model at {md} — train it first (src.run_tt_train)")
    feat = json.loads((md / "feature_names.json").read_text(encoding="utf-8"))
    std = json.loads((md / "standardizer.json").read_text(encoding="utf-8"))
    mu = np.asarray(std["mu"], dtype="float64"); sd = np.asarray(std["sd"], dtype="float64")
    models = [CatBoostRegressor().load_model(str(p)) for p in sorted(md.glob("curve_seed*/curve.cbm"))]
    if not models:
        raise SystemExit(f"no curve_seed*/curve.cbm under {md}")
    horizons = [int(x) for x in a.horizons.split(",") if x.strip()]

    need = list(dict.fromkeys(feat + ["symbol", "base_time", "sigma"] + [f"y_{h}" for h in horizons]))
    df = load_dataset(a.dataset_dir, columns=need)
    df["base_time"] = pd.to_datetime(df["base_time"], utc=True)
    edge = df["base_time"].max(); start = edge - pd.Timedelta(hours=a.hours)
    df = df[df["base_time"] >= start].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no rows in window")
    cut_min = (int(pd.Timestamp(a.holdout_cutoff).value // 60_000_000_000)
               if a.holdout_cutoff else _train_cutoff_min())

    X = df[feat]
    preds = np.stack([m.predict(X) for m in models])       # [S, n, 240] per-seed
    pz = preds.mean(axis=0); pstd = preds.std(axis=0)       # ensemble mean + cross-seed dispersion (std space)
    pred = pz * sd + mu                                     # vol-norm cumret curve
    pstd_vn = pstd * sd                                     # cross-seed dispersion, vol-norm units
    sig = df["sigma"].to_numpy("float64")
    syms_col = df["symbol"].to_numpy()
    tmin = to_ns(pd.DatetimeIndex(df["base_time"])) // 60_000_000_000   # robust to us/ns parquet resolution
    cost_map = _costs()
    cst = np.array([cost_map.get(str(s), MEDIAN_COST) for s in syms_col])
    hold_all = (tmin >= cut_min).astype(int) if cut_min is not None else np.zeros(len(df), dtype=int)

    tt_syms: list[str] = []; idx_of: dict[str, int] = {}
    def sidx(s: str) -> int:
        if s not in idx_of:
            idx_of[s] = len(tt_syms); tt_syms.append(s)
        return idx_of[s]

    # CURVE-NATIVE fields (impossible with a scalar-prob model):
    #   move   = predicted signed % move at h (the EV/magnitude gate, vs cost)
    #   snr    = |pred| / cross-seed dispersion = POINTWISE clarity (the spread analog)
    #   persist= fraction of live horizons >= h that keep the same sign (curve holds direction)
    hidx = [h - 1 for h in horizons]
    signmat = np.sign(pred[:, hidx])                        # [n, H] over the live horizons
    EPS = 1e-9
    legs: list[list] = []
    for j, h in enumerate(horizons):
        ph = pred[:, h - 1]
        yreal = df[f"y_{h}"].to_numpy("float64")
        side = np.where(ph >= 0, 1, -1)
        absp = np.abs(ph); n = len(absp)
        order = absp.argsort(); ranks = np.empty(n); ranks[order] = np.arange(n)
        conv = ranks / max(n - 1, 1)                        # відносний: per-horizon percentile [0,1]
        net = side * ((np.exp(yreal * sig) - 1.0) * 100.0) - cst
        move = (np.exp(ph * sig) - 1.0) * 100.0             # predicted signed % move
        snr = absp / (pstd_vn[:, h - 1] + EPS)              # поточечний: clarity / seed-agreement
        persist = (signmat[:, j:] == signmat[:, j:j + 1]).mean(axis=1)
        for i in range(n):
            legs.append([sidx(str(syms_col[i])), int(tmin[i]), int(h), int(side[i]),
                         round(float(conv[i]), 4), round(float(net[i]), 3), int(hold_all[i]),
                         round(float(move[i]), 3), round(float(snr[i]), 2), round(float(persist[i]), 3)])

    hours = max(1.0, (edge - start).total_seconds() / 3600.0)
    tt = {"meta": {"cutoff_min": cut_min, "window_start": str(start), "window_end": str(edge),
                   "hours": round(hours, 1), "horizons": horizons, "tz": "Europe/Kiev",
                   "cost": "per-instrument binance (already in net)",
                   "conv": "per-horizon percentile of |predicted move|"},
          "syms": tt_syms, "eq": [1 if is_equity(s) else 0 for s in tt_syms], "legs": legs}

    ex = _existing_lines()
    sims = json.loads(ex.get("window.SIMS", "{}")); sims.pop("ТТ крива", None)   # drop the legacy wedged sim
    lines = []
    for key in PRESERVE:
        if key in ex:
            lines.append(f"{key}={ex[key]};")
    lines.append("window.SIMS=" + json.dumps(sims, separators=(",", ":")) + ";")
    lines.append("window.TT=" + json.dumps(tt, separators=(",", ":")) + ";")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hold_n = int(sum(l[6] for l in legs))
    print(f"wrote {OUT}  window.TT legs={len(legs)} syms={len(tt_syms)} holdout_legs={hold_n} "
          f"horizons={horizons} window={hours:.0f}h end={edge} cutoff_min={cut_min} size={OUT.stat().st_size//1024}KB")


if __name__ == "__main__":
    main()
