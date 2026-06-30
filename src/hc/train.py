"""Training helpers for HC UP/DOWN CatBoost models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from . import config as HC
from .folds import FoldSpec, choose_exec_v2_folds, choose_folds, split_masks


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def _params(
    task_type: str,
    devices: str,
    iterations: int,
    depth: int,
    verbose: int,
    *,
    learning_rate: float | None = None,
    l2_leaf_reg: float | None = None,
    border_count: int | None = None,
    gpu_ram_part: float | None = None,
    od_wait: int | None = None,
) -> dict:
    params = dict(HC.MODEL_PARAMS)
    params["task_type"] = task_type
    params["iterations"] = iterations
    params["depth"] = depth
    params["verbose"] = verbose
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    if l2_leaf_reg is not None:
        params["l2_leaf_reg"] = l2_leaf_reg
    if border_count is not None:
        params["border_count"] = border_count
    if gpu_ram_part is not None:
        params["gpu_ram_part"] = gpu_ram_part
    if od_wait is not None:
        params["od_wait"] = od_wait
    if task_type.upper() == "GPU":
        params["devices"] = devices
    else:
        params.pop("devices", None)
        params.pop("gpu_ram_part", None)  # GPU-only; CatBoost rejects it on CPU
    return params


def train_fold(
    df: pd.DataFrame,
    fold: FoldSpec,
    out_dir: Path,
    *,
    task_type: str = "GPU",
    devices: str = "0",
    iterations: int = 4000,
    depth: int = 6,
    verbose: int = 100,
    learning_rate: float | None = None,
    l2_leaf_reg: float | None = None,
    border_count: int | None = None,
    gpu_ram_part: float | None = None,
    od_wait: int | None = None,
) -> dict:
    masks = split_masks(df, fold)
    train_mask = masks["train"]
    val_mask = masks["val"]
    test_mask = masks["test"]
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0 or int(test_mask.sum()) == 0:
        raise RuntimeError(
            f"{fold.name}: empty split train={int(train_mask.sum())} "
            f"val={int(val_mask.sum())} test={int(test_mask.sum())}"
        )

    fold_dir = out_dir / fold.name
    fold_dir.mkdir(parents=True, exist_ok=True)
    X_train = df.loc[train_mask, HC.FEATURE_COLUMNS]
    X_val = df.loc[val_mask, HC.FEATURE_COLUMNS]
    w_train = df.loc[train_mask, "weight"].to_numpy("float64")
    w_val = df.loc[val_mask, "weight"].to_numpy("float64")
    params = _params(
        task_type,
        devices,
        iterations,
        depth,
        verbose,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        gpu_ram_part=gpu_ram_part,
        od_wait=od_wait,
    )

    metrics = {
        "fold": fold.to_dict(),
        "val_start": pd.Timestamp(masks["val_start"]).isoformat(),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_purged_between_train_val": int(masks["purged"].sum()),
        "models": {},
    }

    for side, label_col in (("up", "up_label"), ("down", "down_label")):
        y_train = df.loc[train_mask, label_col].astype("int8").to_numpy()
        y_val = df.loc[val_mask, label_col].astype("int8").to_numpy()
        if len(np.unique(y_train)) < 2:
            raise RuntimeError(f"{fold.name} {side}: training labels have one class only")
        model = CatBoostClassifier(**params)
        train_pool = Pool(X_train, y_train, weight=w_train, feature_names=HC.FEATURE_COLUMNS)
        val_pool = Pool(X_val, y_val, weight=w_val, feature_names=HC.FEATURE_COLUMNS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        val_prob = model.predict_proba(X_val)[:, 1]
        model.save_model(fold_dir / f"{side}.cbm")
        metrics["models"][side] = {
            "label": label_col,
            "val_auc": _auc(y_val, val_prob),
            "train_pos_rate": float(y_train.mean()),
            "val_pos_rate": float(y_val.mean()),
            "best_iteration": int(model.get_best_iteration() or 0),
        }
        print(
            f"  {fold.name} {side}: val_auc={metrics['models'][side]['val_auc']} "
            f"train_pos={metrics['models'][side]['train_pos_rate']:.4f} "
            f"val_pos={metrics['models'][side]['val_pos_rate']:.4f}",
            flush=True,
        )

    (fold_dir / "feature_names.json").write_text(
        json.dumps(HC.FEATURE_COLUMNS, indent=2), encoding="utf-8"
    )
    (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_all(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    folds: list[FoldSpec] | None = None,
    max_folds: int = 3,
    task_type: str = "GPU",
    devices: str = "0",
    iterations: int = 4000,
    depth: int = 6,
    verbose: int = 100,
    fold_plan: str = "default",
    primary_days: int = 7,
    spring_days: int = 14,
    learning_rate: float | None = None,
    l2_leaf_reg: float | None = None,
    border_count: int | None = None,
    gpu_ram_part: float | None = None,
    od_wait: int | None = None,
) -> list[dict]:
    df = df.sort_values("base_time").reset_index(drop=True)
    if folds:
        selected = folds
    elif fold_plan == "exec_v2":
        selected = choose_exec_v2_folds(
            df,
            primary_days=primary_days,
            spring_days=spring_days,
            max_folds=max_folds,
        )
    else:
        selected = choose_folds(df, max_folds=max_folds)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_names.json").write_text(
        json.dumps(HC.FEATURE_COLUMNS, indent=2), encoding="utf-8"
    )
    snapshot = HC.config_snapshot(
        {
            "folds": [f.to_dict() for f in selected],
            "fold_plan": fold_plan,
            "actual_model_params": _params(
                task_type,
                devices,
                iterations,
                depth,
                verbose,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                border_count=border_count,
                gpu_ram_part=gpu_ram_part,
                od_wait=od_wait,
            ),
            "rows": int(len(df)),
            "symbols": int(df["symbol"].nunique()),
            "base_time_min": pd.to_datetime(df["base_time"], utc=True).min().isoformat(),
            "base_time_max": pd.to_datetime(df["base_time"], utc=True).max().isoformat(),
        }
    )
    (out_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    all_metrics = []
    for fold in selected:
        print(f"\nTraining {fold.name}: {fold.test_start} -> {fold.test_end}", flush=True)
        all_metrics.append(
            train_fold(
                df,
                fold,
                out_dir,
                task_type=task_type,
                devices=devices,
                iterations=iterations,
                depth=depth,
                verbose=verbose,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                border_count=border_count,
                gpu_ram_part=gpu_ram_part,
                od_wait=od_wait,
            )
        )
    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    return all_metrics
