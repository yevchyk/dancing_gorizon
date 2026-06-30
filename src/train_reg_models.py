"""Train the next-version regression models (ret/mfe/mae x horizon) and validate
on the last-10-day holdout.

Core question: does a higher PREDICTED expected return mean a higher REALIZED
return? We check correlation + a decile monotonicity table + an EV-entry backtest
(go long when predicted ret > fees), plus reward/risk from predicted mae.

Usage:
  python -m src.train_reg_models
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from .features import CurveBuilder
from .training.horizon_slicer import HorizonSlicer
from .training.reg_model_spec import all_reg_specs
from .training.reg_trainer import RegTrainer

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0   # round-trip fee as a fraction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--no-save", action="store_true", help="evaluate only, keep saved models")
    args = p.parse_args()

    ds = pd.read_parquet(C.DATASETS_DIR / "master_reg.parquet")
    t = pd.to_datetime(ds["anchor_time"], utc=True)
    cutoff = t.max() - pd.Timedelta(days=C.HOLDOUT_DAYS)
    train, hold = ds[t < cutoff].copy(), ds[t >= cutoff].copy()
    print(f"train={len(train)}  holdout={len(hold)}  (cutoff {cutoff.date()})  "
          f"iterations={args.iterations} depth={args.depth}")

    trainer = RegTrainer(HorizonSlicer(CurveBuilder(
        C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)),
        iterations=args.iterations, depth=args.depth)

    preds = {}
    print("\n=== TRAIN (RMSE, pred-vs-real corr on val) ===")
    for spec in all_reg_specs():
        model, m = trainer.train(train, spec)
        if not args.no_save:
            trainer.save(model, spec, m)
        cols = m["columns"]
        preds[spec.name] = model.predict(hold[cols])
        bi = model.get_best_iteration()
        print(f"  {spec.name:<8} rmse={m['rmse']:.4f} corr={m['pred_corr']:+.3f} best_iter={bi}")

    print("\n=== HOLDOUT: does higher predicted E[r] => higher realized r? ===")
    for h in C.HORIZONS:
        lab = h.label
        p = preds[f"ret_{lab}"]
        real = hold[f"ret_{lab}"].to_numpy()
        corr = np.corrcoef(p, real)[0, 1]
        # decile monotonicity
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        dec = pd.DataFrame({"q": q, "real": real}).groupby("q")["real"].mean()
        print(f"  ret_{lab}: oos_corr={corr:+.3f}  "
              f"low-decile={dec.iloc[0]*100:+.3f}%  high-decile={dec.iloc[-1]*100:+.3f}%")

    print("\n=== HOLDOUT EV-ENTRY (long if pred_ret>fee, short if <-fee) ===")
    for h in C.HORIZONS:
        lab = h.label
        p = preds[f"ret_{lab}"]
        real = hold[f"ret_{lab}"].to_numpy()
        mae_pred = preds[f"mae_{lab}"]
        longs = p > FEE
        shorts = p < -FEE
        pnl = np.concatenate([real[longs] - FEE, -real[shorts] - FEE])
        if len(pnl):
            # reward/risk on longs: predicted ret over predicted dip
            rr = np.nanmean(p[longs] / np.abs(mae_pred[longs] + 1e-9)) if longs.any() else float("nan")
            print(f"  {lab:>3}: n={len(pnl):>5} win={np.mean(pnl>0):.3f} "
                  f"avg_pnl={pnl.mean()*100:+.4f}%  long_RR~{rr:.2f}")
    print("\nmodels -> models/reg/")


if __name__ == "__main__":
    main()
