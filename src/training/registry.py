"""Loads trained models and scores feature rows. Shared by testing and trading
so both use the exact same model set and column handling.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .. import config as C
from .model_spec import ModelSpec, all_model_specs


class ModelRegistry:
    def __init__(self, models: dict[str, tuple[object, list[str]]], specs: list[ModelSpec]):
        self._models = models                 # name -> (estimator, columns)
        self._specs = {s.name: s for s in specs}

    @classmethod
    def load_default(cls) -> "ModelRegistry":
        models: dict[str, tuple[object, list[str]]] = {}
        for spec in all_model_specs():
            d = C.DIRECTION_MODELS_DIR if spec.kind == "direction" else C.STABILITY_MODELS_DIR
            model_path = d / f"{spec.name}.joblib"
            cols_path = d / f"{spec.name}_columns.joblib"
            if not model_path.exists():
                continue
            models[spec.name] = (joblib.load(model_path), joblib.load(cols_path))
        return cls(models, all_model_specs())

    @property
    def names(self) -> list[str]:
        return list(self._models)

    def spec(self, name: str) -> ModelSpec:
        return self._specs[name]

    def score(self, df: pd.DataFrame, prefix: str = "prob_") -> pd.DataFrame:
        """Return df with one probability column per model (prob_<name>)."""
        out = df.copy()
        for name, (model, cols) in self._models.items():
            out[f"{prefix}{name}"] = model.predict_proba(df[cols])[:, 1]
        return out
