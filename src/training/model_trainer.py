"""Trains a single CatBoost classifier for one ModelSpec.

CatBoost params migrated from old src/train.py. Time-based split (sort by
anchor_time, last fraction = validation) to avoid future leakage. Balanced
class weights because targets are imbalanced.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from .. import config as C
from .model_spec import ModelSpec
from .horizon_slicer import HorizonSlicer


class ModelTrainer:
    def __init__(self, slicer: HorizonSlicer, iterations: int = 500, depth: int = 5,
                 test_size: float = 0.2, random_state: int = 42):
        self.slicer = slicer
        self.iterations = iterations
        self.depth = depth
        self.test_size = test_size
        self.random_state = random_state

    def _split(self, X: pd.DataFrame, y: pd.Series):
        cut = int(len(X) * (1 - self.test_size))
        return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

    def train(self, dataset: pd.DataFrame, spec: ModelSpec) -> tuple[CatBoostClassifier, dict]:
        # time order so validation is strictly the most recent anchors
        df = dataset.sort_values("anchor_time")
        cols = self.slicer.columns_for(spec.horizon)
        X = df[cols]
        y = df[spec.target_column].astype(int)

        X_tr, X_val, y_tr, y_val = self._split(X, y)
        model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=0.03,
            depth=self.depth,
            l2_leaf_reg=5,
            min_data_in_leaf=10,
            subsample=0.8,
            colsample_bylevel=0.5,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=self.random_state,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                  use_best_model=True, early_stopping_rounds=50)

        proba = model.predict_proba(X_val)[:, 1]
        pred = model.predict(X_val)
        try:
            auc = float(roc_auc_score(y_val, proba)) if y_val.nunique() > 1 else None
        except ValueError:
            auc = None
        metrics = {
            "name": spec.name,
            "kind": spec.kind,
            "target": spec.target_column,
            "n_features": len(cols),
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_val)),
            "pos_rate": float(y.mean()),
            "accuracy": float(accuracy_score(y_val, pred)),
            "roc_auc": auc,
            "report": classification_report(y_val, pred, zero_division=0, output_dict=True),
            "columns": cols,
        }
        return model, metrics

    def save(self, model: CatBoostClassifier, spec: ModelSpec, metrics: dict) -> Path:
        out_dir = C.DIRECTION_MODELS_DIR if spec.kind == "direction" else C.STABILITY_MODELS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_dir / f"{spec.name}.joblib")
        joblib.dump(metrics["columns"], out_dir / f"{spec.name}_columns.joblib")
        (out_dir / f"{spec.name}_metadata.json").write_text(
            json.dumps({k: v for k, v in metrics.items() if k != "columns"}, indent=2),
            encoding="utf-8",
        )
        return out_dir / f"{spec.name}.joblib"
