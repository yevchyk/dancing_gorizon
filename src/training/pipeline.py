"""Trains all 15 models from the single shared dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import config as C
from .model_spec import all_model_specs
from .model_trainer import ModelTrainer


class MultiModelPipeline:
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer

    def run(self, dataset_path: Path) -> dict:
        dataset = pd.read_parquet(dataset_path)
        print(f"dataset: {len(dataset)} rows, {dataset.shape[1]} cols")

        summary: list[dict] = []
        for spec in all_model_specs():
            model, metrics = self.trainer.train(dataset, spec)
            self.trainer.save(model, spec, metrics)
            auc = metrics["roc_auc"]
            print(f"  {spec.name:<12} feats={metrics['n_features']:>3} "
                  f"pos={metrics['pos_rate']:.3f} "
                  f"auc={auc:.4f}" if auc is not None else
                  f"  {spec.name:<12} feats={metrics['n_features']:>3} "
                  f"pos={metrics['pos_rate']:.3f} auc=N/A")
            summary.append({k: metrics[k] for k in
                            ("name", "kind", "n_features", "pos_rate", "accuracy", "roc_auc")})

        out = C.MODELS_DIR / "summary.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary -> {out}")
        return {"models": summary}
