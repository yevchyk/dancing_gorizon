"""Rolling walk-forward over a pre-built independent-anchor master dataset.

For each fold: train all 15 models IN MEMORY on the train slice (never touching
the production models the live loop uses), score the strictly-later test slice,
and resolve the corrected target/stop/timeout PnL per trade. The concatenated
per-trade table is honest out-of-sample, independent-anchor statistics.
"""

from __future__ import annotations

import pandas as pd

from .. import config as C
from ..database import CandleStore
from ..training import ModelRegistry
from ..training.model_spec import all_model_specs
from ..training.model_trainer import ModelTrainer
from ..trading.optimizer import _resolve_one, HORIZON_MIN
from ..trading.timeutil import index_to_ns, anchors_to_ns


class WalkForward:
    def __init__(self, store: CandleStore, trainer: ModelTrainer,
                 train_days: int = 90, test_days: int = 14, n_folds: int = 4,
                 min_train_rows: int = 1500):
        self.store = store
        self.trainer = trainer
        self.train_days = train_days
        self.test_days = test_days
        self.n_folds = n_folds
        self.min_train_rows = min_train_rows

    def fold_plan(self, t_min: pd.Timestamp, t_max: pd.Timestamp) -> list[dict]:
        folds = []
        test_end = t_max
        for _ in range(self.n_folds):
            test_start = test_end - pd.Timedelta(days=self.test_days)
            train_start = test_start - pd.Timedelta(days=self.train_days)
            if train_start < t_min:
                break
            folds.append({"train_start": train_start, "test_start": test_start,
                          "test_end": test_end})
            test_end = test_start
        return list(reversed(folds))

    def _train_fold(self, train_df: pd.DataFrame) -> ModelRegistry:
        models: dict[str, tuple[object, list[str]]] = {}
        for spec in all_model_specs():
            if train_df[spec.target_column].nunique() < 2:
                continue   # single-class target in this fold -> can't train
            try:
                model, metrics = self.trainer.train(train_df, spec)
            except Exception as e:
                print(f"    train {spec.name} skipped: {e}")
                continue
            models[spec.name] = (model, metrics["columns"])
        return ModelRegistry(models, all_model_specs())

    def _resolve_fold(self, scored: pd.DataFrame, registry: ModelRegistry,
                      fold_id: int) -> list[dict]:
        dir_models = [n for n in registry.names
                      if registry.spec(n).kind == "direction"]
        rows: list[dict] = []
        for symbol, g in scored.groupby("symbol"):
            candles = self.store.load(symbol)
            if candles is None:
                continue
            ts = index_to_ns(candles.index)
            high, low, close = (candles[c].to_numpy(float) for c in ("high", "low", "close"))
            anchors_ns = anchors_to_ns(g["anchor_time"])
            days = pd.to_datetime(g["anchor_time"], utc=True).dt.strftime("%Y-%m-%d").to_numpy()
            for name in dir_models:
                spec = registry.spec(name)
                side = "long" if spec.direction == "up" else "short"
                move, hmin = spec.horizon.move_pct, HORIZON_MIN[spec.horizon.label]
                probs = g[f"prob_{name}"].to_numpy(float)
                for a_ns, pr, day in zip(anchors_ns, probs, days):
                    res = _resolve_one(ts, high, low, close, int(a_ns), side, move,
                                       hmin, C.STOP_PCT_RATIO, C.OKX_FEE_PER_SIDE)
                    if res is None:
                        continue
                    rows.append({"fold": fold_id, "day": day, "symbol": symbol,
                                 "model": name, "prob": float(pr),
                                 "won": res[0], "pnl_pct": res[1]})
        return rows

    def run(self, master: pd.DataFrame) -> pd.DataFrame:
        t = pd.to_datetime(master["anchor_time"], utc=True)
        master = master.assign(_t=t)
        folds = self.fold_plan(t.min(), t.max())
        print(f"walk-forward: {len(folds)} folds "
              f"(train {self.train_days}d / test {self.test_days}d)")

        all_rows: list[dict] = []
        for i, f in enumerate(folds):
            train = master[(master["_t"] >= f["train_start"]) & (master["_t"] < f["test_start"])]
            test = master[(master["_t"] >= f["test_start"]) & (master["_t"] < f["test_end"])]
            if len(train) < self.min_train_rows or test.empty:
                print(f"  fold {i}: SKIP (train={len(train)} test={len(test)})")
                continue
            registry = self._train_fold(train.drop(columns="_t"))
            scored = registry.score(test.drop(columns="_t"))
            rows = self._resolve_fold(scored, registry, i)
            all_rows.extend(rows)
            print(f"  fold {i}: {f['test_start'].date()}..{f['test_end'].date()}  "
                  f"train={len(train)} test={len(test)} trades={len(rows)}", flush=True)
        return pd.DataFrame(all_rows)
