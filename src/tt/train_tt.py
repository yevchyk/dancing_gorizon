"""ТТ Phase 1 — the CURVE model (CatBoost MultiRMSE, multi-output).

Predicts the whole vol-normalized cumulative-return curve (h=1..h_max nodes) in
ONE pass — horizon is the output axis, no re-query (kills §9 off-anchor bias).

Per-node standardization (B8): each of the H target nodes is z-scored on the TRAIN
split before MultiRMSE so the √h growth doesn't let late horizons dominate the loss
and every horizon contributes equally. The standardizer (mu/sd per node) is saved so
predictions invert back to the vol-normalized cumret curve; multiply by the row's
`sigma` to get expected % move, then compare to cost at decision time.

3-seed ensemble. The dataset already excludes the user's 4-day holdout, so the final
fit uses ALL rows; a tail (or random) val slice only picks the best iteration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from ..hc.data import load_dataset
from . import schema_tt as STT


def _resolve_cols(dataset_dir: Path) -> tuple[list[str], list[str]]:
    fp = dataset_dir / "feature_names.json"
    if not fp.exists():
        fp = dataset_dir.parent / "feature_names.json"
    feat_cols = json.loads(fp.read_text(encoding="utf-8"))
    tp = dataset_dir / "target_names.json"
    tgt_cols = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else STT.TARGET_COLUMNS_TT
    return feat_cols, tgt_cols


def _split(base: pd.Series, *, val_fraction: float, embargo_min: int,
           random_val: bool, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    n = len(base)
    if random_val:
        rng = np.random.default_rng(seed)
        k = max(1, int(n * val_fraction))
        val = np.zeros(n, dtype=bool)
        val[rng.choice(n, size=k, replace=False)] = True
        return ~val, val, "random"
    times = np.array(sorted(pd.unique(base)))
    cut = max(1, int(len(times) * (1.0 - val_fraction)))
    val_start = pd.Timestamp(times[cut])
    if val_start.tzinfo is None:
        val_start = val_start.tz_localize("UTC")
    # embargo by the curve span so a TRAIN row's forward target doesn't overlap val.
    # Compare in pandas (tz-aware throughout) then hand back plain numpy bool masks.
    embargo = pd.Timedelta(minutes=int(embargo_min))
    train = (base < (val_start - embargo)).to_numpy()
    val = (base >= val_start).to_numpy()
    return train, val, val_start.isoformat()


def _params(*, task_type: str, devices: str, iterations: int, depth: int, learning_rate: float,
            l2_leaf_reg: float, border_count: int, gpu_ram_part: float, seed: int, verbose: int) -> dict:
    p = dict(loss_function="MultiRMSE", eval_metric="MultiRMSE", iterations=iterations,
             depth=depth, learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
             random_seed=int(seed), task_type=task_type, allow_writing_files=False, verbose=verbose)
    if task_type == "GPU":
        p.update(devices=devices, border_count=border_count, gpu_ram_part=gpu_ram_part)
    return p


def train_curve(*, dataset_dir: Path, model_dir: Path, seeds: list[int], val_fraction: float,
                random_val: bool, task_type: str, devices: str, iterations: int, depth: int,
                learning_rate: float, l2_leaf_reg: float, border_count: int, gpu_ram_part: float,
                od_wait: int, verbose: int, no_early_stop: bool, sample_frac: float,
                embargo_min: int | None, no_scale: bool = False,
                continue_from: Path | None = None) -> dict:
    feat_cols, tgt_cols = _resolve_cols(dataset_dir)
    print(f"loading ТТ curve dataset from {dataset_dir}", flush=True)
    df = load_dataset(dataset_dir)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    df = df.sort_values("base_time").reset_index(drop=True)
    missing = [c for c in feat_cols + tgt_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"dataset missing {len(missing)} cols, e.g. {missing[:3]}")
    if embargo_min is None:
        embargo_min = len(tgt_cols) + 5     # ~entry_delay + h_max span of one row's target

    base = pd.to_datetime(df["base_time"], utc=True)
    X = df[feat_cols]
    Y = df[tgt_cols].to_numpy("float32")
    print(f"rows={len(df)} symbols={df['symbol'].nunique()} feats={len(feat_cols)} "
          f"targets={len(tgt_cols)} base={base.min()}..{base.max()}", flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    mu = sd = None
    if continue_from is not None:
        # довчування: REUSE the original per-node standardizer (mu/sd) so the added
        # trees fit the same target scale; never recompute it from a new split.
        std = json.loads((continue_from / "standardizer.json").read_text(encoding="utf-8"))
        if std.get("target_names") and std["target_names"] != tgt_cols:
            raise SystemExit("continue-from standardizer target_names mismatch the dataset")
        mu = np.asarray(std["mu"], dtype="float64")
        sd = np.asarray(std["sd"], dtype="float64")
        print(f"CONTINUE from {continue_from} (+{iterations} trees/seed, reusing standardizer)", flush=True)
        if task_type == "GPU":
            # CatBoost cannot continue (init_model) on GPU -> top-up runs on CPU.
            print("  note: GPU continuation unsupported by CatBoost -> using CPU for the top-up", flush=True)
            task_type = "CPU"
    for si, seed in enumerate(seeds):
        train_mask, val_mask, val_start = _split(base, val_fraction=val_fraction,
                                                  embargo_min=embargo_min, random_val=random_val, seed=seed)
        if mu is None:
            # fresh run: per-node standardizer from the FIRST seed's train split (shared across seeds)
            mu = Y[train_mask].mean(axis=0).astype("float64")
            if no_scale:                                      # raw-ratio: center only, keep magnitude across nodes/coins
                sd = np.ones(Y.shape[1], dtype="float64")
            else:
                sd = np.where(Y[train_mask].std(axis=0) < 1e-8, 1.0, Y[train_mask].std(axis=0)).astype("float64")
        Yz = ((Y - mu) / sd).astype("float32")

        p = _params(task_type=task_type, devices=devices, iterations=iterations, depth=depth,
                    learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg, border_count=border_count,
                    gpu_ram_part=gpu_ram_part, seed=seed, verbose=verbose)
        fold_dir = model_dir / f"curve_seed{seed}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nTraining curve seed {seed} (train={int(train_mask.sum())} val={int(val_mask.sum())} "
              f"val_start={val_start})", flush=True)

        if continue_from is not None:
            init_path = continue_from / f"curve_seed{seed}" / "curve.cbm"
            if not init_path.exists():
                raise SystemExit(f"continue-from missing {init_path}")
            init = CatBoostRegressor()
            init.load_model(str(init_path))
            fp = dict(p); fp.pop("od_type", None); fp.pop("od_wait", None)
            model = CatBoostRegressor(**fp)   # adds `iterations` more trees on top of init
            model.fit(Pool(X, Yz, feature_names=feat_cols), init_model=init)
            best = int(model.tree_count_)
            val_rmse = None
        elif no_early_stop:
            fp = dict(p)
            fp["allow_writing_files"] = True       # enable train_dir logs (learn_error.tsv) + resumable snapshot
            fp["train_dir"] = str(fold_dir)
            def _fit_snap():
                m = CatBoostRegressor(**fp)
                m.fit(Pool(X, Yz, feature_names=feat_cols),
                      save_snapshot=True, snapshot_file="snapshot.cbsnap",   # RELATIVE to train_dir (abs path got doubled by CatBoost)
                      snapshot_interval=60)         # checkpoint /60s -> re-running the SAME cmd resumes from here
                return m
            try:
                model = _fit_snap()
            except Exception as e:                  # snapshot from a DIFFERENT config -> CatBoost refuses; clear & restart fresh
                snap = fold_dir / "snapshot.cbsnap"
                if snap.exists() and ("params are different" in str(e) or "snapshot" in str(e).lower()):
                    print(f"  incompatible snapshot -> clearing {snap.name}, restarting fresh", flush=True)
                    snap.unlink()
                    model = _fit_snap()
                else:
                    raise
            best = iterations
            val_rmse = None
        else:
            p.update(od_type="Iter", od_wait=od_wait)
            probe = CatBoostRegressor(**p)
            probe.fit(Pool(X.loc[train_mask], Yz[train_mask], feature_names=feat_cols),
                      eval_set=Pool(X.loc[val_mask], Yz[val_mask], feature_names=feat_cols),
                      use_best_model=True)
            gbi = probe.get_best_iteration()   # may legitimately be 0 -> keep it, don't `or`
            best = max(1, int(gbi if gbi is not None else iterations))
            try:
                val_rmse = float(probe.get_best_score()["validation"]["MultiRMSE"])
            except Exception:
                val_rmse = None
            fp = dict(p); fp["iterations"] = best; fp.pop("od_type", None); fp.pop("od_wait", None)
            model = CatBoostRegressor(**fp)
            model.fit(Pool(X, Yz, feature_names=feat_cols))

        model.save_model(str(fold_dir / "curve.cbm"))
        m = {"seed": int(seed), "best_iteration": int(best), "val_multirmse": val_rmse,
             "val_start": val_start, "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
             "n_final_fit": int(len(df))}
        (fold_dir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        all_metrics.append(m)
        print(f"  seed {seed}: best_iter={best} val_multirmse={val_rmse}", flush=True)

    # shared artifacts for the ensemble + inference
    (model_dir / "feature_names.json").write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")
    (model_dir / "target_names.json").write_text(json.dumps(tgt_cols, indent=2), encoding="utf-8")
    (model_dir / "standardizer.json").write_text(json.dumps({
        "target_names": tgt_cols, "mu": list(map(float, mu)), "sd": list(map(float, sd)),
        "note": "pred_z * sd + mu = vol-normalized cumret; * row sigma = expected log-return at that node",
    }, indent=2), encoding="utf-8")
    snapshot = {
        "schema": "tt_curve", "kind": "MultiRMSE_curve", "seeds": seeds,
        "feature_columns": len(feat_cols), "target_nodes": len(tgt_cols),
        "depth": depth, "iterations": iterations, "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg, "task_type": task_type, "no_early_stop": no_early_stop,
        "val_fraction": val_fraction, "random_val": random_val, "embargo_min": embargo_min,
        "rows": int(len(df)), "symbols": int(df["symbol"].nunique()),
        "base_time_min": base.min().isoformat(), "base_time_max": base.max().isoformat(),
        "per_seed": all_metrics,
    }
    (model_dir / "config_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"\ncurve models -> {model_dir}", flush=True)
    return snapshot
