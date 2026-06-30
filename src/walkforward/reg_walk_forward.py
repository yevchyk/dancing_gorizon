"""Walk-forward for the regression (ret/mfe/mae) models over the independent
master. Per fold: retrain in-memory on the train slice, predict on the strictly
later test slice, and record predicted vs realized ret/mfe/mae per anchor.

Realized targets are already columns in the dataset (close-return / excursions),
so no candle re-resolution is needed -- the test rows carry ground truth.
"""

from __future__ import annotations

import pandas as pd

from .. import config as C
from ..training.reg_model_spec import all_reg_specs
from ..training.reg_trainer import RegTrainer

HZ = [h.label for h in C.HORIZONS]


class RegWalkForward:
    def __init__(self, trainer: RegTrainer, train_days: int = 90,
                 test_days: int = 14, n_folds: int = 4, min_train_rows: int = 1500):
        self.trainer = trainer
        self.train_days = train_days
        self.test_days = test_days
        self.n_folds = n_folds
        self.min_train_rows = min_train_rows

    def fold_plan(self, t_min, t_max) -> list[dict]:
        folds, test_end = [], t_max
        for _ in range(self.n_folds):
            test_start = test_end - pd.Timedelta(days=self.test_days)
            train_start = test_start - pd.Timedelta(days=self.train_days)
            if train_start < t_min:
                break
            folds.append({"train_start": train_start, "test_start": test_start,
                          "test_end": test_end})
            test_end = test_start
        return list(reversed(folds))

    def run(self, master: pd.DataFrame) -> pd.DataFrame:
        t = pd.to_datetime(master["anchor_time"], utc=True)
        master = master.assign(_t=t)
        folds = self.fold_plan(t.min(), t.max())
        print(f"reg walk-forward: {len(folds)} folds "
              f"(train {self.train_days}d / test {self.test_days}d)")

        out: list[pd.DataFrame] = []
        for i, f in enumerate(folds):
            tr = master[(master["_t"] >= f["train_start"]) & (master["_t"] < f["test_start"])]
            te = master[(master["_t"] >= f["test_start"]) & (master["_t"] < f["test_end"])]
            if len(tr) < self.min_train_rows or te.empty:
                print(f"  fold {i}: SKIP (train={len(tr)} test={len(te)})")
                continue
            base = pd.DataFrame({
                "fold": i, "symbol": te["symbol"].to_numpy(),
                "day": pd.to_datetime(te["anchor_time"], utc=True).dt.strftime("%Y-%m-%d").to_numpy(),
            }, index=te.index)
            for spec in all_reg_specs():
                model, m = self.trainer.train(tr.drop(columns="_t"), spec)
                base[f"pred_{spec.name}"] = model.predict(te[m["columns"]])
                base[f"real_{spec.name}"] = te[spec.target_column].to_numpy()
            out.append(base)
            print(f"  fold {i}: {f['test_start'].date()}..{f['test_end'].date()}  "
                  f"train={len(tr)} test={len(te)}", flush=True)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
