"""TT v2 walk-forward — honest multi-window OOS for the per-step (ratioB) model.

ONE full-year dataset (cutoff=edge, no reservation). Slides K weekly windows; for each:
  train on  base_time < T_k - embargo   (embargo = h_max + entry_delay, no target leak)
  test  on  [T_k, T_k + test_days)       (unseen forward week)
Fit MultiRMSE (center-only, NO scale = raw-ratio magnitude), predict the 300-step curve,
cumprod -> cumulative, score ratio-native at conv>=gate per horizon. NEVER tunes on the
test — each window is pure forward OOS. Resumable: appends per-window to --out json and
skips windows already present. Final: % of windows GREEN (+net) per horizon.

  .venv/Scripts/python -m src.run_tt2_walkforward --windows 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .hc.data import load_dataset
from .tt import schema_tt as STT

COSTS = Path("configs/binance_costs.json")
MEDIAN_COST = 0.126


def _costs() -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(COSTS.read_text())["costs"].items()}
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/tt2_wf_ratioB/dataset"))
    ap.add_argument("--mode", choices=["ratioA", "ratioB"], default="ratioB")
    ap.add_argument("--windows", type=int, default=15)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--h-max", type=int, default=300)
    ap.add_argument("--entry-delay-min", type=int, default=5)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.045)
    ap.add_argument("--l2-leaf-reg", type=float, default=4.0)
    ap.add_argument("--task-type", default="GPU")
    ap.add_argument("--gpu-ram-part", type=float, default=0.8)
    ap.add_argument("--horizons", default="120,180,240,300")
    ap.add_argument("--conv", type=float, default=0.9)
    ap.add_argument("--out", type=Path, default=Path("reports/tt2_wf_ratioB.json"))
    a = ap.parse_args()

    feat = STT.feature_names_tt(include_regime=False)        # 543 (no regime)
    tgt = STT.target_columns_tt(a.h_max, 1)                  # y_1..y_300
    eval_h = [int(x) for x in a.horizons.split(",") if x.strip()]
    embargo = pd.Timedelta(minutes=a.h_max + a.entry_delay_min)
    cost_map = _costs()

    print(f"loading {a.dataset_dir} ...", flush=True)
    need = list(dict.fromkeys(feat + tgt + ["symbol", "base_time"]))
    df = load_dataset(a.dataset_dir, columns=need).sort_values("base_time").reset_index(drop=True)
    base = pd.to_datetime(df["base_time"], utc=True)
    X = df[feat]
    Y = df[tgt].to_numpy("float32")
    syms = df["symbol"].to_numpy()
    cst = np.array([cost_map.get(str(s), MEDIAN_COST) for s in syms])
    data_end = base.max()
    print(f"  rows={len(df)} span {base.min()} .. {data_end}  feat={len(feat)} tgt={len(tgt)}", flush=True)

    # weekly window cutoffs: last test = [data_end - test_days, data_end]
    cutoffs = [data_end - pd.Timedelta(days=a.test_days) - pd.Timedelta(days=a.step_days) * (a.windows - 1 - k)
               for k in range(a.windows)]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if a.out.exists():
        try:
            done = {r["cutoff"]: r for r in json.loads(a.out.read_text())}
        except Exception:
            done = {}
    results = list(done.values())

    for k, T in enumerate(cutoffs):
        key = T.isoformat()
        if key in done:
            print(f"[{k+1}/{a.windows}] {key} — already done, skip", flush=True)
            continue
        tr = (base < (T - embargo)).to_numpy()
        te = ((base >= T) & (base < T + pd.Timedelta(days=a.test_days))).to_numpy()
        if tr.sum() < 5000 or te.sum() < 200:
            print(f"[{k+1}/{a.windows}] {key} — too few rows (tr={int(tr.sum())} te={int(te.sum())}), skip", flush=True)
            continue
        mu = Y[tr].mean(axis=0).astype("float64")            # center only, NO scale (magnitude kept)
        Ytr = (Y[tr] - mu).astype("float32")
        p = dict(loss_function="MultiRMSE", iterations=a.iterations, depth=a.depth,
                 learning_rate=a.learning_rate, l2_leaf_reg=a.l2_leaf_reg, random_seed=42,
                 task_type=a.task_type, allow_writing_files=False, verbose=False)
        if a.task_type == "GPU":
            p.update(devices="0", border_count=32, gpu_ram_part=a.gpu_ram_part)
        m = CatBoostRegressor(**p)
        m.fit(Pool(X.loc[tr], Ytr, feature_names=feat))
        pred = m.predict(X.loc[te]) + mu                     # back to ratio space
        if a.mode == "ratioB":
            pred = np.cumprod(pred, axis=1)
        Yte = Y[te]; cste = cst[te]
        real_cum = np.cumprod(Yte, axis=1) if a.mode == "ratioB" else Yte   # steps -> realized CUMULATIVE move
        per_h = {}
        for h in eval_h:
            pc = pred[:, h - 1]
            move = (pc - 1.0) * 100.0
            side = np.where(pc >= 1.0, 1, -1)
            net = side * ((real_cum[:, h - 1] - 1.0) * 100.0) - cste
            absm = np.abs(move); n = len(absm)
            order = absm.argsort(); ranks = np.empty(n); ranks[order] = np.arange(n)
            gate = (ranks / max(n - 1, 1)) >= a.conv
            ng = int(gate.sum())
            per_h[str(h)] = {"n": ng,
                             "win": round(float((net[gate] > 0).mean() * 100), 1) if ng else None,
                             "net": round(float(net[gate].mean()), 3) if ng else None}
        rec = {"idx": k, "cutoff": key, "train_n": int(tr.sum()), "test_n": int(te.sum()), "per_h": per_h}
        results.append(rec)
        a.out.write_text(json.dumps(results, indent=1), encoding="utf-8")     # resumable checkpoint
        row = " ".join(f"h{h}:{per_h[str(h)]['win']}%/{per_h[str(h)]['net']}" for h in eval_h)
        print(f"[{k+1}/{a.windows}] {key[:10]} tr={int(tr.sum())} te={int(te.sum())}  {row}", flush=True)

    # ---- aggregate ----
    print(f"\n=== WALK-FORWARD SUMMARY ({a.mode}, conv>={a.conv}, {len([r for r in results])} windows) ===", flush=True)
    print(f"{'h':>5} {'green/N':>9} {'med win':>8} {'med net':>8} {'mean net':>9}")
    for h in eval_h:
        nets = [r["per_h"][str(h)]["net"] for r in results if r["per_h"].get(str(h), {}).get("net") is not None]
        wins = [r["per_h"][str(h)]["win"] for r in results if r["per_h"].get(str(h), {}).get("win") is not None]
        if not nets:
            continue
        green = sum(1 for x in nets if x > 0)
        print(f"{h:>5} {f'{green}/{len(nets)}':>9} {np.median(wins):>7.1f}% {np.median(nets):>8.3f} {np.mean(nets):>9.3f}")
    print(f"\nGREEN/N = у скількох вікнах net додатний. Хочемо >=~3/4 зелених + додатний mean.", flush=True)


if __name__ == "__main__":
    main()
