"""Trains one CatBoost *regressor* per RegModelSpec (predicts expected ret/mfe/mae).

Same horizon-scoped feature slicing and time-based split as the classifier
trainer. Metric: RMSE, plus the correlation between predicted and realized value
on validation (does a higher predicted E[r] really mean a higher realized r?).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .. import config as C
from .reg_model_spec import RegModelSpec
from .horizon_slicer import HorizonSlicer

REG_MODELS_DIR = C.MODELS_DIR / "reg"


class RegTrainer:
    def __init__(self, slicer: HorizonSlicer, iterations: int = 600, depth: int = 6,
                 test_size: float = 0.2, random_state: int = 42):
        self.slicer = slicer
        self.iterations = iterations
        self.depth = depth
        self.test_size = test_size
        self.random_state = random_state

    def _split(self, X, y):
        cut = int(len(X) * (1 - self.test_size))
        return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

    def train(self, dataset: pd.DataFrame, spec: RegModelSpec):
        df = dataset.sort_values("anchor_time")
        cols = self.slicer.columns_for(spec.horizon)
        X = df[cols]
        y = df[spec.target_column].astype(float)
        X_tr, X_val, y_tr, y_val = self._split(X, y)

        model = CatBoostRegressor(
            iterations=self.iterations, learning_rate=0.03, depth=self.depth,
            l2_leaf_reg=5, min_data_in_leaf=20, subsample=0.8, colsample_bylevel=0.5,
            loss_function="RMSE", random_seed=self.random_state, verbose=0,
            allow_writing_files=False,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                  use_best_model=True, early_stopping_rounds=50)

        pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((pred - y_val.to_numpy()) ** 2)))
        corr = float(np.corrcoef(pred, y_val.to_numpy())[0, 1]) if y_val.nunique() > 1 else 0.0
        metrics = {"name": spec.name, "kind": spec.kind, "target": spec.target_column,
                   "n_features": len(cols), "n_train": int(len(X_tr)),
                   "n_val": int(len(X_val)), "rmse": rmse, "pred_corr": corr,
                   "y_mean": float(y.mean()), "columns": cols}
        return model, metrics

    def save(self, model, spec: RegModelSpec, metrics: dict) -> Path:
        REG_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, REG_MODELS_DIR / f"{spec.name}.joblib")
        joblib.dump(metrics["columns"], REG_MODELS_DIR / f"{spec.name}_columns.joblib")
        (REG_MODELS_DIR / f"{spec.name}_metadata.json").write_text(
            json.dumps({k: v for k, v in metrics.items() if k != "columns"}, indent=2),
            encoding="utf-8")
        return REG_MODELS_DIR / f"{spec.name}.joblib"
