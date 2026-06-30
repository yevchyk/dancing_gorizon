"""Hour-of-day analysis: are some UTC hours systematically better/worse for the
engine? (Is the recent weakness about WHEN we trade?)

Scores the independent-anchor master with the production dir_prob models, builds
the v4 clean+agree signals, and buckets realized PnL by the anchor's UTC hour.
~200 days of anchors -> enough per hour. (Partly in-sample on the train portion,
so read the RELATIVE hour pattern, not absolute edge.)

Usage:
  python -m src.run_hour_analysis
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import CurveBuilder
from .training.horizon_slicer import HorizonSlicer

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
FLOOR, OPP, AGREE = 0.82, 0.30, 2
EXCL = {"down_1h", "down_2h"}
DIRP = C.MODELS_DIR / "dir_prob"


def main() -> None:
    ds = pd.read_parquet(C.DATASETS_DIR / "master_reg.parquet")
    slicer = HorizonSlicer(CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN))
    cost = FEE + 0.0005
    t = pd.to_datetime(ds["anchor_time"], utc=True)
    hour = t.dt.hour.to_numpy()

    # score every horizon's up/down
    P = {}
    for h in C.HORIZONS:
        X = ds[slicer.columns_for(h)]
        P[("up", h.label)] = joblib.load(DIRP / f"up_{h.label}.joblib").predict_proba(X)[:, 1]
        P[("down", h.label)] = joblib.load(DIRP / f"down_{h.label}.joblib").predict_proba(X)[:, 1]

    # per-anchor: clean signals per side per horizon, require agreement>=2, take best spread
    n = len(ds)
    sig_side = np.full(n, 0)        # +1 long / -1 short / 0 none
    sig_pnl = np.full(n, np.nan)
    for i in range(n):
        best = None
        for side, sgn in (("up", 1), ("down", -1)):
            agree = 0; bestlab = None; bestspread = -1
            for h in C.HORIZONS:
                if f"{side}_{h.label}" in EXCL:
                    continue
                p = P[(side, h.label)][i]
                opp = P[("down" if side == "up" else "up", h.label)][i]
                if p >= FLOOR and opp <= OPP:
                    agree += 1
                    if p - opp > bestspread:
                        bestspread = p - opp; bestlab = h.label
            if agree >= AGREE and bestspread > (best[1] if best else -1):
                best = (sgn, bestspread, bestlab)
        if best:
            sgn, _, lab = best
            sig_side[i] = sgn
            sig_pnl[i] = sgn * ds[f"ret_{lab}"].iloc[i] - cost

    m = sig_side != 0
    df = pd.DataFrame({"hour": hour[m], "pnl": sig_pnl[m], "won": (sig_pnl[m] > 0)})
    print(f"total clean signals: {len(df)} over {ds['anchor_time'].nunique()} anchors\n")
    print("=== BY UTC HOUR ===")
    print(f"  {'hour':>4} {'n':>5} {'win':>6} {'avg_pnl':>9}")
    for hr, g in df.groupby("hour"):
        if len(g) >= 20:
            print(f"  {hr:>4} {len(g):>5} {g.won.mean():>6.3f} {g.pnl.mean()*100:>+8.4f}%")
    # day (08-20 UTC) vs night (20-08 UTC)
    night = df[(df.hour >= 20) | (df.hour < 8)]
    day = df[(df.hour >= 8) & (df.hour < 20)]
    print(f"\n  DAY  (08-20 UTC): n={len(day):>5} win={day.won.mean():.3f} avg={day.pnl.mean()*100:+.4f}%")
    print(f"  NIGHT(20-08 UTC): n={len(night):>5} win={night.won.mean():.3f} avg={night.pnl.mean()*100:+.4f}%")


if __name__ == "__main__":
    main()
