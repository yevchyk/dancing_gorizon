"""Loads one old model group (8 models: up/down x 30/90/180/240m) and scores
the v2 holdout anchors, producing prob_<name> columns aligned with each model's
own feature order.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

OLD_MODELS_ROOT = Path(r"C:\ml\ml_predictor\models")
OLD_HORIZONS = (30, 90, 180, 240)
OLD_DIRECTIONS = ("up", "down")

# directional touch threshold each group was trained for (for PnL via ExitSimulator)
GROUP_MOVE_PCT = {
    "directional_p03": 0.03, "directional_p03_close": 0.03,
    "directional_p05": 0.05, "directional_p05_close": 0.05,
}


class LegacyModelGroup:
    def __init__(self, group: str, root: Path = OLD_MODELS_ROOT):
        self.group = group
        self.dir = root / group
        self.models: dict[str, object] = {}
        self.feature_names: dict[str, list[str]] = {}
        for d in OLD_DIRECTIONS:
            for h in OLD_HORIZONS:
                name = f"{d}_{h}m"
                path = self.dir / f"target_{name}.joblib"
                if not path.exists():
                    continue
                m = joblib.load(path)
                self.models[name] = m
                self.feature_names[name] = list(m.feature_names_)

    @property
    def names(self) -> list[str]:
        return list(self.models)

    @staticmethod
    def move_pct(group: str) -> float:
        return GROUP_MOVE_PCT.get(group, 0.05)

    def score_rows(self, feat_rows: list[dict]) -> pd.DataFrame:
        """feat_rows: legacy feature dicts (from OldFeatureBuilder.build_rows)."""
        X = pd.DataFrame(feat_rows)
        out = pd.DataFrame(index=X.index)
        for name, model in self.models.items():
            cols = self.feature_names[name]
            Xm = X.reindex(columns=cols, fill_value=0.0)
            out[f"prob_{name}"] = model.predict_proba(Xm)[:, 1]
        return out
