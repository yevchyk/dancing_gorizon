"""Shared HC model registry and schema-aware scorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hc import config as HC


SIM_TO_DIR = {
    "d7 (hc_final)": "models/hc_final",
    "d8 (hc_final_d8)": "models/hc_final_d8",
    "OLD": "models/hc_exec_stride120_nonoverlap",
    "NEW": "models/hc_exec_to20260604_prod",
    "min1 2-120": "models/min1_2to120",
    "min1 flat d9": "models/min1_flat_d9",
    "min1 flat d10": "models/min1_flat_d10",
    "min1 flat d12": "models/min1_flat_d12",
    "3mo flat d12": "models/min1_3mo_d12",
    "binance d8": "models/binance_y1_d8",
    "binance d10": "models/binance_y1_d10",
    "binance d12": "models/binance_y1_d12",
    "binance d8 20k": "models/binance_y1_d8_it20k",
    "binance d12 20k": "models/binance_y1_d12_it20k",
    "binance d8 v5": "models/binance_y1_v5_d8",
    # legacy "(probe)" names: builds saved before the single-pool consolidation
    # carry these; keep resolving them to the same model dirs.
    "binance d8 (probe)": "models/binance_y1_d8",
    "binance d10 (probe)": "models/binance_y1_d10",
    "binance d12 (probe)": "models/binance_y1_d12",
    "binance d8 20k (probe)": "models/binance_y1_d8_it20k",
    "binance d12 20k (probe)": "models/binance_y1_d12_it20k",
    "binance d8 v5 (probe)": "models/binance_y1_v5_d8",
}


def model_dir_for_sim(sim: str | None) -> Path:
    key = str(sim or "OLD")
    return Path(SIM_TO_DIR.get(key, key))


def feature_cols(model_dir: Path) -> list[str]:
    model_dir = Path(model_dir)
    fn = model_dir / "feature_names.json"
    if fn.exists():
        return json.loads(fn.read_text(encoding="utf-8"))
    snap = model_dir / "config_snapshot.json"
    if snap.exists():
        data = json.loads(snap.read_text(encoding="utf-8"))
        cols = data.get("feature_columns")
        if cols:
            return list(cols)
    return list(HC.FEATURE_COLUMNS)


def model_schema(model_dir: Path) -> str:
    cols = feature_cols(Path(model_dir))
    # v5 = v4 curves + market-regime block; MUST be detected before v4 or a live
    # v4 scorer would silently feed 305 features to a 323-feature model.
    if "funding_level" in cols:
        return "v5"
    if any(str(c).startswith("c1m_") for c in cols):
        return "v4"
    if "hour_sin" in cols:
        return "v2"
    return "legacy"


def _fold_dirs(model_dir: Path) -> list[Path]:
    model_dir = Path(model_dir)
    snap = model_dir / "config_snapshot.json"
    if snap.exists():
        try:
            data = json.loads(snap.read_text(encoding="utf-8"))
            dirs = [model_dir / str(f["name"]) for f in data.get("folds", []) if f.get("name")]
            dirs = [d for d in dirs if (d / "up.cbm").exists() and (d / "down.cbm").exists()]
            if dirs:
                return dirs
        except Exception:
            pass
    return sorted(p for p in model_dir.iterdir() if (p / "up.cbm").exists() and (p / "down.cbm").exists())


class EnsembleScorer:
    def __init__(self, model_dir: Path) -> None:
        from catboost import CatBoostClassifier

        self.model_dir = Path(model_dir)
        self.cols = feature_cols(self.model_dir)
        self.folds = []
        for fold_dir in _fold_dirs(self.model_dir):
            up = CatBoostClassifier()
            down = CatBoostClassifier()
            up.load_model(fold_dir / "up.cbm")
            down.load_model(fold_dir / "down.cbm")
            self.folds.append((fold_dir.name, up, down))
        if not self.folds:
            raise FileNotFoundError(f"No HC fold models found under {self.model_dir}")

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            shown = ", ".join(missing[:8])
            raise KeyError(f"{self.model_dir}: missing feature columns: {shown}")
        x = df[self.cols]
        up_preds: list[np.ndarray] = []
        down_preds: list[np.ndarray] = []
        for name, up, down in self.folds:
            print(f"  score ensemble fold {name}", flush=True)
            up_preds.append(up.predict_proba(x)[:, 1].astype("float32"))
            down_preds.append(down.predict_proba(x)[:, 1].astype("float32"))
        out = df.copy()
        out["up_prob"] = np.vstack(up_preds).mean(axis=0).astype("float32")
        out["down_prob"] = np.vstack(down_preds).mean(axis=0).astype("float32")
        out["model_vote_count"] = len(self.folds)
        return out


def score_ensemble(df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    return EnsembleScorer(model_dir).score(df)
