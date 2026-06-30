"""Evaluate a trained HC model dir on the locked last-24h test from the dataset.

Works for both the symbol-blind model (302 feats) and the tagged model
(302 + symbol). Auto-detects from feature_names.json. Reports the winrate/net
frontier, per-day, and last-day, all on rows the model never saw (base>=cutoff).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .hc import config as HC
from .hc.data import load_dataset
from .run_hc_prod_train import parse_cutoff, target_end

COST = 0.75  # fee 0.15 + slip 0.60, matches scorecard "all coins"


def load_pair(model_dir: Path):
    snap = model_dir / "config_snapshot.json"
    folds = json.loads(snap.read_text())["folds"]
    fd = model_dir / folds[0]["name"]
    up = CatBoostClassifier(); up.load_model(fd / "up.cbm")
    dn = CatBoostClassifier(); dn.load_model(fd / "down.cbm")
    feats = json.loads((fd / "feature_names.json").read_text())
    return up, dn, feats


def make_legs(df, up, dn, feats):
    X = df[[c for c in feats if c != "symbol"]].copy()
    if "symbol" in feats:
        X["symbol"] = df["symbol"].astype(str)
        X = X[feats]
    up_p = up.predict_proba(X)[:, 1]
    dn_p = dn.predict_proba(X)[:, 1]
    base = df[["symbol", "base_time", "horizon_minutes", "ret_pct"]].reset_index(drop=True)
    parts = []
    for side, pdir, popp, sgn in (("long", up_p, dn_p, 1), ("short", dn_p, up_p, -1)):
        d = base.copy()
        d["side"] = side
        d["p_dir"] = pdir
        d["p_opp"] = popp
        d["spread"] = d["p_dir"] - d["p_opp"]
        d["net"] = sgn * d["ret_pct"] - COST
        d["won"] = (d["net"] > 0).astype(int)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def dedup(legs):  # one best bet per (symbol, base_time) by p_dir
    return legs.sort_values("p_dir", ascending=False).drop_duplicates(["symbol", "base_time"], keep="first")


def frontier(legs, label):
    print(f"\n=== {label}: winrate/net frontier (locked 24h, dedup one bet/scan) ===")
    gates = [("RAW90", legs.p_dir >= 0.90), ("RAW85", legs.p_dir >= 0.85),
             ("RAW80", legs.p_dir >= 0.80), ("SPREAD80", legs.spread >= 0.80),
             ("SPREAD70", legs.spread >= 0.70), ("SPREAD60", legs.spread >= 0.60),
             ("bdw .80&opp.05", (legs.p_dir >= 0.80) & (legs.p_opp <= 0.05))]
    for nm, m in gates:
        d = dedup(legs[m])
        if len(d):
            print(f"  {nm:16s} n={len(d):4d} win={d.won.mean():.0%} net={d.net.mean():+.3f} sum={d.net.sum():+.1f}")
        else:
            print(f"  {nm:16s} n=0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/hc_tagged/dataset"))
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--cutoff-local", required=True)
    args = ap.parse_args()

    cutoff = parse_cutoff(args.cutoff_local)
    df = load_dataset(args.dataset_dir)
    df["base_time"] = pd.to_datetime(df["base_time"], utc=True)
    te = df[df["base_time"] >= cutoff].copy()
    up, dn, feats = load_pair(args.model_dir)
    tagged = "symbol" in feats
    print(f"model={args.model_dir.name}  tagged={tagged}  feats={len(feats)}  "
          f"test rows={len(te)} ({te.base_time.min()} -> {te.base_time.max()})")

    legs = make_legs(te, up, dn, feats)
    legs["date"] = legs["base_time"].dt.tz_convert("Europe/Kiev").dt.date
    frontier(legs, args.model_dir.name)

    # per-day at SPREAD80 + RAW85
    print("  -- per-day (RAW85 tail) --")
    for d, g in legs.groupby("date"):
        t = dedup(g[g.p_dir >= 0.85])
        w = f"{t.won.mean():.0%}" if len(t) else "-"
        n = f"{t.net.mean():+.2f}" if len(t) else "-"
        print(f"     {d}  n={len(t):3d} win={w:>4} net={n:>6}")


if __name__ == "__main__":
    main()
