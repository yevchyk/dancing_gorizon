"""Export TT v2 models (ratioA + ratioB) -> window.TT_MODELS in data.js for the ТТ tab.

Ratio-native scoring on the holdout-inclusive eval dataset (data/tt2_now_ratioA):
  move% = (pred_cumulative[h]-1)*100 ; side = sign(move) ; net = side*(y_h-1)*100 - cost.
  ratioA: model predicts cumulative ratio directly; ratioB: cumprod of per-step preds.
Each model -> 10-field legs [sym,tmin,h,side,conv,net,hold,move,snr,persist] (same layout
as window.TT). snr is set to 0 = N/A (seed-dispersion needs >=2 seeds; v2 is 1-seed).
Writes window.TT_MODELS={"ratioA":{...},"ratioB":{...}} + window.TT=ratioA (default); the
tab's model <select> swaps window.TT. Preserves the other (binance) window vars.

  python -m src.run_tt2_export_html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .hc.data import load_dataset, to_ns
from .markets import is_equity

OUT = Path("reports/sim_explorer/data.js")
COSTS = Path("configs/binance_costs.json")
MEDIAN_COST = 0.126
LIVE_HORIZONS = (30, 45, 60, 90, 120, 180, 240, 270, 300)
PRESERVE = ("window.SIM_META", "window.SYMS", "window.LIQUID", "window.COST")
MODELS = [("ratioA", "A · vs входу (кумулятив)", "models/tt2_ratioA"),
          ("ratioB", "B · vs попередньої (покроково)", "models/tt2_ratioB")]


def _existing_lines() -> dict[str, str]:
    out: dict[str, str] = {}
    if not OUT.exists():
        return out
    for line in OUT.read_text(encoding="utf-8").splitlines():
        s = line.strip().rstrip(";")
        for key in PRESERVE + ("window.SIMS",):
            if s.startswith(key + "="):
                out[key] = s[len(key) + 1:]
    return out


def _costs() -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(COSTS.read_text())["costs"].items()}
    except Exception:
        return {}


def _score(md: str, mode: str, df, feat, horizons, cst, cut_min, tmin, sidx) -> list[list]:
    std = json.loads((Path(md) / "standardizer.json").read_text(encoding="utf-8"))
    mu = np.asarray(std["mu"], dtype="float64"); sd = np.asarray(std["sd"], dtype="float64")
    models = [CatBoostRegressor().load_model(str(p)) for p in sorted(Path(md).glob("curve_seed*/curve.cbm"))]
    pred = np.stack([m.predict(df[feat]) for m in models]).mean(axis=0) * sd + mu
    if mode == "ratioB":
        pred = np.cumprod(pred, axis=1)                       # per-step -> cumulative
    hidx = [h - 1 for h in horizons]
    signmat = np.sign(pred[:, hidx] - 1.0)
    hold_all = (tmin >= cut_min).astype(int) if cut_min is not None else np.zeros(len(df), dtype=int)
    legs: list[list] = []
    for j, h in enumerate(horizons):
        pc = pred[:, h - 1]
        move = (pc - 1.0) * 100.0
        side = np.where(pc >= 1.0, 1, -1)
        yreal = df[f"y_{h}"].to_numpy("float64")
        net = side * ((yreal - 1.0) * 100.0) - cst
        absm = np.abs(move); n = len(absm)
        order = absm.argsort(); ranks = np.empty(n); ranks[order] = np.arange(n)
        conv = ranks / max(n - 1, 1)
        persist = (signmat[:, j:] == signmat[:, j:j + 1]).mean(axis=1)
        for i in range(n):
            legs.append([int(sidx[i]), int(tmin[i]), int(h), int(side[i]), round(float(conv[i]), 4),
                         round(float(net[i]), 3), int(hold_all[i]), round(float(move[i]), 3),
                         0, round(float(persist[i]), 3)])
    return legs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("data/tt2_now_ratioA/dataset"))
    ap.add_argument("--train-summary", type=Path, default=Path("data/tt2_ratioA/dataset_summary.json"))
    a = ap.parse_args()

    feat = json.loads(Path("models/tt2_ratioA/feature_names.json").read_text(encoding="utf-8"))
    cut = pd.Timestamp(json.loads(a.train_summary.read_text(encoding="utf-8"))["cutoff"])
    cut_min = int(cut.value // 60_000_000_000)
    horizons = list(LIVE_HORIZONS)

    need = list(dict.fromkeys(feat + ["symbol", "base_time"] + [f"y_{h}" for h in horizons]))
    df = load_dataset(a.eval_dir, columns=need)
    df["base_time"] = pd.to_datetime(df["base_time"], utc=True)
    tmin = to_ns(pd.DatetimeIndex(df["base_time"])) // 60_000_000_000
    syms_col = df["symbol"].to_numpy()
    cost_map = _costs()
    cst = np.array([cost_map.get(str(s), MEDIAN_COST) for s in syms_col])

    tt_syms: list[str] = []; idx_of: dict[str, int] = {}
    sidx = np.empty(len(df), dtype="int64")
    for i, s in enumerate(syms_col):
        s = str(s)
        if s not in idx_of:
            idx_of[s] = len(tt_syms); tt_syms.append(s)
        sidx[i] = idx_of[s]
    eq = [1 if is_equity(s) else 0 for s in tt_syms]

    models_out: dict[str, dict] = {}
    for key, label, md in MODELS:
        legs = _score(md, key, df, feat, horizons, cst, cut_min, tmin, sidx)
        models_out[key] = {"label": label,
                           "meta": {"cutoff_min": cut_min, "horizons": horizons, "tz": "Europe/Kiev",
                                    "model": key, "cost": "per-instrument binance (already in net)",
                                    "conv": "per-horizon percentile of |predicted move|", "snr": "N/A (1 seed)"},
                           "syms": tt_syms, "eq": eq, "legs": legs}
        print(f"  scored {key}: legs={len(legs)} holdout={sum(l[6] for l in legs)}")

    ex = _existing_lines()
    sims = json.loads(ex.get("window.SIMS", "{}")); sims.pop("ТТ крива", None)
    lines = [f"{k}={ex[k]};" for k in PRESERVE if k in ex]
    lines.append("window.SIMS=" + json.dumps(sims, separators=(",", ":")) + ";")
    lines.append("window.TT_MODELS=" + json.dumps(models_out, separators=(",", ":")) + ";")
    lines.append("window.TT=window.TT_MODELS.ratioA;")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}  models={list(models_out)} syms={len(tt_syms)} "
          f"rows={len(df)} cutoff_min={cut_min} size={OUT.stat().st_size//1024}KB")


if __name__ == "__main__":
    main()
