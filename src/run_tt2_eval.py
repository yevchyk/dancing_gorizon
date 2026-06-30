"""TT v2 eval — score a ratio-curve model on a holdout-inclusive dataset, ratio-native.

Two target modes share ONE eval dataset (ratioA = cumulative truth):
  ratioA: model predicts cumulative price ratio (exit/entry) directly.
  ratioB: model predicts per-minute STEP ratios -> cumprod along time = cumulative.
Realized cumulative move = the eval dataset's y_h (= exit/entry). For a leg:
  move%   = (pred_cumulative[h] - 1) * 100        (predicted)
  side    = sign(pred_cumulative[h] - 1)
  net%    = side * (y_h - 1)*100  -  per-instrument binance cost
  conv    = per-horizon percentile of |move| (the gate)
Reports IS (base_time < train cutoff) vs HOLDOUT (>= cutoff) win%/net per horizon.

  .venv/Scripts/python -m src.run_tt2_eval --model-dir models/tt2_ratioA --mode ratioA
  .venv/Scripts/python -m src.run_tt2_eval --model-dir models/tt2_ratioB --mode ratioB
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .hc.data import load_dataset

COSTS = Path("configs/binance_costs.json")
MEDIAN_COST = 0.126


def _costs() -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(COSTS.read_text())["costs"].items()}
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["ratioA", "ratioB"], required=True)
    ap.add_argument("--eval-dir", type=Path, default=Path("data/tt2_now_ratioA/dataset"))
    ap.add_argument("--train-summary", type=Path, default=None,
                    help="dataset_summary.json of the TRAIN set (for the cutoff). default = model_dir/../<derive>")
    ap.add_argument("--cutoff", default=None, help="UTC ISO; overrides train-summary cutoff")
    ap.add_argument("--horizons", default="30,60,120,180,240,300")
    ap.add_argument("--conv", type=float, default=0.9)
    a = ap.parse_args()

    md = a.model_dir
    feat = json.loads((md / "feature_names.json").read_text(encoding="utf-8"))
    std = json.loads((md / "standardizer.json").read_text(encoding="utf-8"))
    mu = np.asarray(std["mu"], dtype="float64")
    sd = np.asarray(std["sd"], dtype="float64")
    models = [CatBoostRegressor().load_model(str(p)) for p in sorted(md.glob("curve_seed*/curve.cbm"))]
    if not models:
        raise SystemExit(f"no curve_seed*/curve.cbm under {md}")

    if a.cutoff:
        cut = pd.Timestamp(a.cutoff)
    else:
        ts = a.train_summary or (Path(str(md).replace("models", "data")) / "dataset_summary.json")
        cut = pd.Timestamp(json.loads(Path(ts).read_text(encoding="utf-8"))["cutoff"])
    cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")

    horizons = [int(x) for x in a.horizons.split(",") if x.strip()]
    need = list(dict.fromkeys(feat + ["symbol", "base_time"] + [f"y_{h}" for h in horizons]))
    df = load_dataset(a.eval_dir, columns=need)
    bt = pd.to_datetime(df["base_time"], utc=True)
    hold = (bt >= cut).to_numpy()

    P = np.stack([m.predict(df[feat]) for m in models]).mean(axis=0)   # [n, H] standardized
    pred = P * sd + mu                                                 # denorm to ratio space
    if a.mode == "ratioB":
        pred = np.cumprod(pred, axis=1)                                # per-step -> cumulative

    syms = df["symbol"].to_numpy()
    cost_map = _costs()
    cst = np.array([cost_map.get(str(s), MEDIAN_COST) for s in syms])

    def s(mask: np.ndarray, net: np.ndarray):
        n = int(mask.sum())
        if n == 0:
            return n, None, None
        return n, round(float((net[mask] > 0).mean() * 100), 1), round(float(net[mask].mean()), 3)

    def f(x):
        return "  —  " if x is None else f"{x:>6}"

    print(f"\nmodel={md.name} mode={a.mode}  cutoff={cut}  rows={len(df)}  holdout_rows={int(hold.sum())}  conv>={a.conv}")
    print(f"{'h':>5} | {'IS n':>7} {'IS win':>7} {'IS net':>7} | {'HO n':>7} {'HO win':>7} {'HO net':>7}")
    for h in horizons:
        pc = pred[:, h - 1]
        move = (pc - 1.0) * 100.0
        side = np.where(pc >= 1.0, 1, -1)
        yreal = df[f"y_{h}"].to_numpy("float64")
        net = side * ((yreal - 1.0) * 100.0) - cst
        absm = np.abs(move); n = len(absm)
        order = absm.argsort(); ranks = np.empty(n); ranks[order] = np.arange(n)
        conv = ranks / max(n - 1, 1)
        gate = conv >= a.conv
        isn, isw, isnet = s(gate & ~hold, net)
        hon, how, honet = s(gate & hold, net)
        print(f"{h:>5} | {isn:>7} {f(isw)} {f(isnet)} | {hon:>7} {f(how)} {f(honet)}")


if __name__ == "__main__":
    main()
