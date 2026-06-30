"""Agreement filter on last-10d: does requiring the REGRESSION (E[r], 'synthesis')
to agree with the simple CLASSIFIER (P_up / P_down) raise win-rate?

For longs:  p_up >= thr        (classifier alone)
        vs  p_up >= thr AND pred_ret > 0   (both agree up)
Shorts mirror with p_down and pred_ret < 0. Swept over probability thresholds.

Usage:
  python -m src.run_agreement_holdout --slip 0.05
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
REG = C.MODELS_DIR / "reg"
DIRP = C.MODELS_DIR / "dir_prob"
THRS = (0.65, 0.70, 0.75, 0.78, 0.80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip", type=float, default=0.05)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0

    ds = pd.read_parquet(C.DATASETS_DIR / "master_reg.parquet")
    t = pd.to_datetime(ds["anchor_time"], utc=True)
    hold = ds[t > pd.Timestamp(C.PROD_TRAIN_CUTOFF, tz="UTC")].copy()
    print(f"last-10d holdout: {len(hold)} anchors, cost={cost*100:.3f}%\n")

    rows = []
    for h in C.HORIZONS:
        rc = joblib.load(REG / f"ret_{h.label}_columns.joblib")
        rm = joblib.load(REG / f"ret_{h.label}.joblib")
        uc = joblib.load(DIRP / f"up_{h.label}_columns.joblib")
        um = joblib.load(DIRP / f"up_{h.label}.joblib")
        dm = joblib.load(DIRP / f"down_{h.label}.joblib")
        rows.append(pd.DataFrame({
            "horizon": h.label, "pred_ret": rm.predict(hold[rc]),
            "p_up": um.predict_proba(hold[uc])[:, 1],
            "p_down": dm.predict_proba(hold[uc])[:, 1],
            "ret": hold[f"ret_{h.label}"].to_numpy()}))
    R = pd.concat(rows, ignore_index=True)

    def stats(mask, sidesign):
        pnl = sidesign * R.ret.to_numpy()[mask] - cost
        return len(pnl), (pnl > 0).mean() if len(pnl) else 0, pnl.mean() * 100 if len(pnl) else 0

    print("=== LONGS: classifier alone  vs  classifier AND regression>0 ===")
    print(f"  {'thr':>5} | {'P_up alone n/win/pnl':>26} | {'P_up & E[r]>0 n/win/pnl':>28}")
    for thr in THRS:
        a = R.p_up >= thr
        b = a & (R.pred_ret > 0)
        na, wa, pa = stats(a.to_numpy(), 1)
        nb, wb, pb = stats(b.to_numpy(), 1)
        print(f"  {thr:>5.2f} | n={na:>5} win={wa:.3f} pnl={pa:+.3f}% | "
              f"n={nb:>5} win={wb:.3f} pnl={pb:+.3f}%")

    print("\n=== SHORTS: classifier alone  vs  classifier AND regression<0 ===")
    print(f"  {'thr':>5} | {'P_down alone n/win/pnl':>26} | {'P_down & E[r]<0 n/win/pnl':>28}")
    for thr in THRS:
        a = R.p_down >= thr
        b = a & (R.pred_ret < 0)
        na, wa, pa = stats(a.to_numpy(), -1)
        nb, wb, pb = stats(b.to_numpy(), -1)
        print(f"  {thr:>5.2f} | n={na:>5} win={wa:.3f} pnl={pa:+.3f}% | "
              f"n={nb:>5} win={wb:.3f} pnl={pb:+.3f}%")

    print("\n=== BOTH-AGREE @0.75, per horizon (long: p_up>=.75 & E[r]>0) ===")
    for h in C.HORIZONS:
        sub = R[R.horizon == h.label]
        m = (sub.p_up >= 0.75) & (sub.pred_ret > 0)
        if m.sum() >= 5:
            pnl = sub.ret.to_numpy()[m.to_numpy()] - cost
            print(f"  {h.label:>3} long: n={m.sum():>3} win={(pnl>0).mean():.2f} "
                  f"pnl={pnl.mean()*100:+.3f}%")


if __name__ == "__main__":
    main()
