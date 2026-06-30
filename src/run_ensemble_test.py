"""Cross-model-set ensemble test: does averaging the 'scream' of the OLD crypto
models (fast_v3 up_20m) and the NEW ones (bluechip up_18m) on the same symbol+anchor
beat either alone? Fires on consensus; reports win/avg at matched ~19m horizon.

  python -m src.run_ensemble_test
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .fast.curve import FastCurve
from .trading.timeutil import index_to_ns

sys.stdout.reconfigure(encoding="utf-8")
COST = 0.0012


def main():
    bc = pd.read_parquet("outputs/analysis/fast_bluechip/bluechip/holdout_scores.parquet")
    fv3_syms = set(pd.read_parquet("outputs/analysis/fast_v3/holdout_scores.parquet",
                                   columns=["symbol"]).symbol.unique())
    overlap = sorted(set(bc.symbol) & fv3_syms)
    bc = bc[bc.symbol.isin(overlap)].reset_index(drop=True)
    print(f"overlap symbols={len(overlap)}, rows={len(bc)}", flush=True)

    # build fast_v3 curve + load its up_20m model, score on the bluechip anchors
    curve = FastCurve(FC.CURVE_POINTS, FC.CURVE_MIN_STEP_MIN, FC.CURVE_MAX_DEPTH_MIN, FC.CURVE_SEGMENTS)
    model = joblib.load(C.MODELS_DIR / "fast_v3" / "base" / "up_20m.joblib")
    cols = joblib.load(C.MODELS_DIR / "fast_v3" / "base" / "up_20m_columns.joblib")
    store = CandleStore(C.DATA_DIR / "bluechip" / "candles_1m")
    allcols = curve.columns()

    pv3 = np.full(len(bc), np.nan)
    for sym, g in bc.groupby("symbol"):
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        ts = index_to_ns(c.index); close = c["close"].to_numpy("float64")
        a_ns = pd.DatetimeIndex(pd.to_datetime(g["anchor_time"], utc=True)).as_unit("ns").asi8
        feats, valid = curve.build_matrix(ts, close, a_ns)
        X = pd.DataFrame(feats, columns=allcols)
        p = model.predict_proba(X[cols])[:, 1]
        p = np.where(valid, p, np.nan)
        pv3[g.index.to_numpy()] = p
    bc["p_fv3_20m"] = pv3
    bc = bc.dropna(subset=["p_fv3_20m"]).reset_index(drop=True)
    print(f"scored fast_v3 on {len(bc)} rows ({bc.symbol.nunique()} symbols)\n", flush=True)

    real = bc["real_ret_18m"].to_numpy()
    pb = bc["p_up_18m"].to_numpy()
    pf = bc["p_fv3_20m"].to_numpy()
    avg = (pb + pf) / 2.0
    days = bc["day"].nunique()

    def stat(fire):
        n = int(fire.sum())
        if n < days:
            return None
        pnl = real[fire] - COST
        return n, n // max(days, 1), float((pnl > 0).mean()), float(pnl.mean() * 100)

    print(f"corr(bluechip_18m, fast_v3_20m) = {np.corrcoef(pb, pf)[0,1]:.3f}\n")
    print(f"{'config':<22}{'thr':>5}{'n':>7}{'/day':>6}{'win':>8}{'avg%':>9}")
    for thr in (0.75, 0.80, 0.85, 0.90):
        for name, fire in [
            ("bluechip_18m alone", pb >= thr),
            ("fast_v3_20m alone", pf >= thr),
            ("ENSEMBLE avg", avg >= thr),
            ("BOTH agree (each)", (pb >= thr) & (pf >= thr)),
        ]:
            r = stat(fire)
            if r:
                print(f"{name:<22}{thr:>5.2f}{r[0]:>7}{r[1]:>6}{r[2]:>8.3f}{r[3]:>+9.4f}")
        print()


if __name__ == "__main__":
    main()
