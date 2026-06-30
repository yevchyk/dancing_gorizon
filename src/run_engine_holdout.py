"""The combined engine AND each model separately, on the LAST 10 DAYS only,
using PRODUCTION up/down models (trained up to PROD_TRAIN_CUTOFF -> unseen window).

Answers two things:
  - ENGINE: side = argmax(p_up, p_down), conf = that prob, take if conf >= thr.
  - SEPARATE: each up_H (long when p_up>=thr) / down_H (short when p_down>=thr)
    on its own -- so we can see if the argmax merge loses anything.

Usage:
  python -m src.run_engine_holdout --slip 0.05
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
DIR = C.MODELS_DIR / "dir_prob"
THRS = (0.70, 0.75, 0.78, 0.80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip", type=float, default=0.05)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0

    ds = pd.read_parquet(C.DATASETS_DIR / "master_reg.parquet")
    t = pd.to_datetime(ds["anchor_time"], utc=True)
    hold = ds[t > pd.Timestamp(C.PROD_TRAIN_CUTOFF, tz="UTC")].copy()
    days = pd.to_datetime(hold["anchor_time"], utc=True).dt.strftime("%Y-%m-%d")
    print(f"HOLDOUT last-10d (unseen): {len(hold)} anchors, {days.nunique()} days, "
          f"cost/trade={cost*100:.3f}%\n")

    rows = []
    for h in C.HORIZONS:
        up_m = joblib.load(DIR / f"up_{h.label}.joblib")
        up_c = joblib.load(DIR / f"up_{h.label}_columns.joblib")
        dn_m = joblib.load(DIR / f"down_{h.label}.joblib")
        dn_c = joblib.load(DIR / f"down_{h.label}_columns.joblib")
        rows.append(pd.DataFrame({
            "day": days.to_numpy(), "horizon": h.label,
            "p_up": up_m.predict_proba(hold[up_c])[:, 1],
            "p_down": dn_m.predict_proba(hold[dn_c])[:, 1],
            "ret": hold[f"ret_{h.label}"].to_numpy()}))
    R = pd.concat(rows, ignore_index=True)

    # ENGINE (argmax merge)
    print("=== ENGINE (argmax p_up/p_down, all horizons) ===")
    side = np.where(R.p_up >= R.p_down, 1, -1)
    conf = np.maximum(R.p_up, R.p_down)
    epnl = side * R.ret.to_numpy() - cost
    E = pd.DataFrame({"day": R.day, "conf": conf, "pnl": epnl, "won": (epnl > 0).astype(int)})
    for thr in THRS:
        g = E[E.conf >= thr]
        if len(g) < 10:
            continue
        d = g.groupby("day")["pnl"].mean() * 100
        print(f"  conf>={thr:.2f}: n={len(g):>4} win={g.won.mean():.3f} "
              f"avg_pnl={g.pnl.mean()*100:+.4f}%  green={int((d>0).sum())}/{len(d)}")

    # SEPARATE per model
    print("\n=== EACH MODEL SEPARATELY @ conf>=0.78 (vs engine) ===")
    print(f"  {'model':<9} {'n':>4} {'win':>5} {'avg_pnl':>9}")
    for h in C.HORIZONS:
        sub = R[R.horizon == h.label]
        up = sub[sub.p_up >= 0.78]
        up_pnl = up.ret.to_numpy() - cost
        dn = sub[sub.p_down >= 0.78]
        dn_pnl = -dn.ret.to_numpy() - cost
        for nm, n, pnl in (("up_" + h.label, len(up), up_pnl),
                           ("down_" + h.label, len(dn), dn_pnl)):
            if n >= 5:
                print(f"  {nm:<9} {n:>4} {(pnl>0).mean():>5.2f} {pnl.mean()*100:>+8.4f}%")
            else:
                print(f"  {nm:<9} {n:>4}   (too few)")

    # how often both fire (where argmax has to choose)
    print("\n=== OVERLAP: where BOTH p_up>=0.78 AND p_down>=0.78 (argmax must pick) ===")
    for h in C.HORIZONS:
        sub = R[R.horizon == h.label]
        both = ((sub.p_up >= 0.78) & (sub.p_down >= 0.78)).sum()
        anyfire = ((sub.p_up >= 0.78) | (sub.p_down >= 0.78)).sum()
        print(f"  {h.label:>3}: both={both}  any={anyfire}  "
              f"({both/anyfire*100:.0f}% conflicted)" if anyfire else f"  {h.label}: none")


if __name__ == "__main__":
    main()
