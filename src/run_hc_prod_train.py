"""Train production HC models up to an explicit cutoff.

Unlike walk-forward HC training, this script is for live deployment: it uses all
rows whose targets are known by the cutoff (`exit_time <= cutoff`) for the final
fit. A time-validation slice is used only to choose the best iteration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from .hc import config as HC
from .hc.data import load_dataset
from .hc.train import _params


LOCAL_TZ = "Europe/Kiev"


def parse_cutoff(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize(LOCAL_TZ)
    return ts.tz_convert("UTC")


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds must not be empty")
    return seeds


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def target_end(df: pd.DataFrame) -> pd.Series:
    if "exit_time" in df.columns:
        return pd.to_datetime(df["exit_time"], utc=True)
    base = pd.to_datetime(df["base_time"], utc=True)
    return base + pd.to_timedelta(df["horizon_minutes"].astype("int64"), unit="min")


def split_for_iteration_pick(
    df: pd.DataFrame,
    *,
    cutoff_utc: pd.Timestamp,
    val_fraction: float,
    random_val: bool = False,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Timestamp | None]:
    base = pd.to_datetime(df["base_time"], utc=True)
    end = target_end(df)
    eligible = end <= cutoff_utc
    if random_val:
        # RANDOM val (not the time tail): pick best_iteration on a representative
        # random slice of the training pool. The real test stays the future holdout.
        elig_idx = np.flatnonzero(eligible.to_numpy())
        if len(elig_idx) < 20:
            raise RuntimeError("not enough eligible rows before cutoff")
        rng = np.random.default_rng(seed)
        k = max(1, int(len(elig_idx) * float(val_fraction)))
        val_idx = rng.choice(elig_idx, size=k, replace=False)
        val_arr = np.zeros(len(df), dtype=bool)
        val_arr[val_idx] = True
        val = pd.Series(val_arr, index=df.index)
        train = eligible & ~val
        if int(train.sum()) == 0 or int(val.sum()) == 0:
            raise RuntimeError("empty random split")
        return eligible, train, val, None
    times = np.array(sorted(base[eligible].unique()))
    if len(times) < 20:
        raise RuntimeError("not enough eligible timestamps before cutoff")
    cut_idx = max(1, int(len(times) * (1.0 - float(val_fraction))))
    val_start = pd.Timestamp(times[cut_idx])
    if val_start.tzinfo is None:
        val_start = val_start.tz_localize("UTC")
    embargo = pd.Timedelta(minutes=HC.EMBARGO_MIN)
    train = eligible & (end < (val_start - embargo))
    val = eligible & (base >= val_start)
    purged = eligible & ~(train | val)
    if int(train.sum()) == 0 or int(val.sum()) == 0:
        raise RuntimeError(f"empty split train={int(train.sum())} val={int(val.sum())}")
    return eligible, train, val, val_start


def fit_with_best_iteration(
    *,
    side: str,
    label_col: str,
    df: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    eligible_mask: pd.Series,
    out_path: Path,
    params: dict,
    feat_cols: list[str],
    no_early_stop: bool = False,
) -> dict:
    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, label_col].astype("int8").to_numpy()
    w_train = df.loc[train_mask, "weight"].to_numpy("float64")
    X_val = df.loc[val_mask, feat_cols]
    y_val = df.loc[val_mask, label_col].astype("int8").to_numpy()
    w_val = df.loc[val_mask, "weight"].to_numpy("float64")
    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"{side}: training labels have one class only")

    if no_early_stop:
        # Recent val window is a low-signal regime that trivially early-stops the
        # probe (best_iter ~3). Train the full requested iterations instead and
        # fit the final model on ALL eligible rows.
        fp = dict(params)
        fp.pop("od_type", None)
        fp.pop("od_wait", None)
        X_all = df.loc[eligible_mask, feat_cols]
        y_all = df.loc[eligible_mask, label_col].astype("int8").to_numpy()
        w_all = df.loc[eligible_mask, "weight"].to_numpy("float64")
        final = CatBoostClassifier(**fp)
        final.fit(Pool(X_all, y_all, weight=w_all, feature_names=feat_cols))
        final.save_model(out_path)
        val_prob = final.predict_proba(X_val)[:, 1]
        return {
            "label": label_col, "val_auc": auc(y_val, val_prob),
            "best_iteration": int(fp.get("iterations", 0)),
            "n_final_fit": int(eligible_mask.sum()), "forced_iters": True,
        }

    probe = CatBoostClassifier(**params)
    probe.fit(
        Pool(X_train, y_train, weight=w_train, feature_names=feat_cols),
        eval_set=Pool(X_val, y_val, weight=w_val, feature_names=feat_cols),
        use_best_model=True,
    )
    val_prob = probe.predict_proba(X_val)[:, 1]
    best_iter = int(probe.get_best_iteration() or params.get("iterations", 1))
    best_iter = max(1, best_iter)

    final_params = dict(params)
    final_params["iterations"] = best_iter
    final_params.pop("od_type", None)
    final_params.pop("od_wait", None)

    X_all = df.loc[eligible_mask, feat_cols]
    y_all = df.loc[eligible_mask, label_col].astype("int8").to_numpy()
    w_all = df.loc[eligible_mask, "weight"].to_numpy("float64")
    final = CatBoostClassifier(**final_params)
    final.fit(Pool(X_all, y_all, weight=w_all, feature_names=feat_cols))
    final.save_model(out_path)

    return {
        "label": label_col,
        "val_auc": auc(y_val, val_prob),
        "train_pos_rate": float(y_train.mean()),
        "val_pos_rate": float(y_val.mean()),
        "eligible_pos_rate": float(y_all.mean()),
        "best_iteration": best_iter,
        "n_train_for_iteration": int(train_mask.sum()),
        "n_val_for_iteration": int(val_mask.sum()),
        "n_final_fit": int(eligible_mask.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/hc_exec_to20260604/dataset"))
    ap.add_argument("--model-dir", type=Path, default=Path("models/hc_exec_to20260604_prod"))
    ap.add_argument("--cutoff-local", default="2026-06-05 00:00",
                    help="exclusive local cutoff; default is end of 2026-06-04 Kyiv")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--iterations", type=int, default=6000)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.045)
    ap.add_argument("--l2-leaf-reg", type=float, default=4.0)
    ap.add_argument("--border-count", type=int, default=32)
    ap.add_argument("--gpu-ram-part", type=float, default=0.85)
    ap.add_argument("--od-wait", type=int, default=300)
    ap.add_argument("--verbose", type=int, default=200)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--no-early-stop", action="store_true",
                    help="train full --iterations (skip the probe/early-stop on the recent val window)")
    ap.add_argument("--feature-names", type=Path, default=None,
                    help="JSON list of feature cols; default auto-detect dataset feature_names.json else legacy 302")
    ap.add_argument("--horizon-min", type=int, default=None, help="band filter: keep horizon_minutes >= this")
    ap.add_argument("--horizon-max", type=int, default=None, help="band filter: keep horizon_minutes <= this")
    ap.add_argument("--random-val", action="store_true",
                    help="pick best_iteration on a RANDOM val slice of the training pool (not the time tail)")
    ap.add_argument("--exclude-days", default="",
                    help="comma Kyiv dates YYYY-MM-DD dropped ENTIRELY from training (e.g. green days)")
    args = ap.parse_args()

    cutoff_utc = parse_cutoff(args.cutoff_local)
    seeds = parse_seeds(args.seeds)
    print(f"loading HC dataset from {args.dataset_dir}", flush=True)
    df = load_dataset(args.dataset_dir)
    if args.sample_frac < 1.0:
        before = len(df)
        df = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
        print(f"subsampled rows {before} -> {len(df)}", flush=True)

    # resolve feature schema: explicit > dataset feature_names.json > legacy 302
    if args.feature_names is not None:
        feat_cols = json.loads(args.feature_names.read_text(encoding="utf-8"))
    elif (args.dataset_dir / "feature_names.json").exists():
        feat_cols = json.loads((args.dataset_dir / "feature_names.json").read_text(encoding="utf-8"))
    elif (args.dataset_dir.parent / "feature_names.json").exists():
        feat_cols = json.loads((args.dataset_dir.parent / "feature_names.json").read_text(encoding="utf-8"))
    else:
        feat_cols = list(HC.FEATURE_COLUMNS)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"dataset missing {len(missing)} feature cols, e.g. {missing[:3]}")
    schema = "v2" if "hour_sin" in feat_cols else "legacy"
    print(f"feature schema: {len(feat_cols)} cols ({schema})", flush=True)

    # optional horizon-band filter (train a band-specialist from a union dataset)
    if args.horizon_min is not None or args.horizon_max is not None:
        lo = args.horizon_min if args.horizon_min is not None else -1
        hi = args.horizon_max if args.horizon_max is not None else 10 ** 9
        before = len(df)
        df = df[df["horizon_minutes"].between(lo, hi)].reset_index(drop=True)
        print(f"horizon filter [{lo},{hi}]: rows {before} -> {len(df)}", flush=True)

    if args.exclude_days.strip():
        days = {d.strip() for d in args.exclude_days.split(",") if d.strip()}
        kd = pd.to_datetime(df["base_time"], utc=True).dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d")
        before = len(df)
        df = df[~kd.isin(days)].reset_index(drop=True)
        print(f"exclude-days {sorted(days)}: rows {before} -> {len(df)}", flush=True)

    df = df.sort_values("base_time").reset_index(drop=True)
    eligible, train_mask, val_mask, val_start = split_for_iteration_pick(
        df,
        cutoff_utc=cutoff_utc,
        val_fraction=args.val_fraction,
        random_val=args.random_val,
    )
    val_start_str = "random" if val_start is None else val_start.isoformat()
    print(
        f"dataset rows={len(df)} symbols={df['symbol'].nunique()} "
        f"base={df['base_time'].min()}..{df['base_time'].max()}",
        flush=True,
    )
    print(
        f"cutoff_utc={cutoff_utc.isoformat()} eligible={int(eligible.sum())} "
        f"val_start={val_start_str} train_pick={int(train_mask.sum())} "
        f"val_pick={int(val_mask.sum())}",
        flush=True,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    (args.model_dir / "feature_names.json").write_text(
        json.dumps(feat_cols, indent=2), encoding="utf-8"
    )

    folds = []
    all_metrics = []
    for seed in seeds:
        name = f"prod_to_20260604_seed{seed}"
        fold_dir = args.model_dir / name
        fold_dir.mkdir(parents=True, exist_ok=True)
        params = _params(
            args.task_type,
            args.devices,
            args.iterations,
            args.depth,
            args.verbose,
            learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2_leaf_reg,
            border_count=args.border_count,
            gpu_ram_part=args.gpu_ram_part,
            od_wait=args.od_wait,
        )
        params["random_seed"] = int(seed)
        print(f"\nTraining production seed {seed}", flush=True)
        metrics = {
            "fold": {
                "name": name,
                "test_start": cutoff_utc.isoformat(),
                "test_end": cutoff_utc.isoformat(),
                "purpose": "production_live",
                "reason": "final fit uses all rows with known targets up to cutoff",
                "btc_return_pct": None,
                "btc_range_pct": None,
            },
            "cutoff_utc": cutoff_utc.isoformat(),
            "val_start": val_start_str,
            "n_eligible": int(eligible.sum()),
            "n_train_for_iteration": int(train_mask.sum()),
            "n_val_for_iteration": int(val_mask.sum()),
            "models": {},
        }
        for side, label_col in (("up", "up_label"), ("down", "down_label")):
            metrics["models"][side] = fit_with_best_iteration(
                side=side,
                label_col=label_col,
                df=df,
                train_mask=train_mask,
                val_mask=val_mask,
                eligible_mask=eligible,
                out_path=fold_dir / f"{side}.cbm",
                params=params,
                feat_cols=feat_cols,
                no_early_stop=args.no_early_stop,
            )
            m = metrics["models"][side]
            print(
                f"  {name} {side}: val_auc={m['val_auc']} "
                f"best_iter={m['best_iteration']} final_rows={m['n_final_fit']}",
                flush=True,
            )
        (fold_dir / "feature_names.json").write_text(
            json.dumps(feat_cols, indent=2), encoding="utf-8"
        )
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        all_metrics.append(metrics)
        folds.append(metrics["fold"])

    base_time = pd.to_datetime(df.loc[eligible, "base_time"], utc=True)
    end_time = target_end(df.loc[eligible])
    snapshot = HC.config_snapshot(
        {
            "folds": folds,
            "fold_plan": "production_cutoff",
            "cutoff_utc": cutoff_utc.isoformat(),
            "validation_used_for_iteration_only": True,
            "actual_model_params": params,
            "feature_columns": feat_cols,
            "feature_count": len(feat_cols),
            "schema": schema,
            "horizon_min": args.horizon_min,
            "horizon_max": args.horizon_max,
            "rows": int(eligible.sum()),
            "symbols": int(df.loc[eligible, "symbol"].nunique()),
            "base_time_min": base_time.min().isoformat(),
            "base_time_max": base_time.max().isoformat(),
            "target_end_max": end_time.max().isoformat(),
        }
    )
    (args.model_dir / "config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )
    (args.model_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"models -> {args.model_dir}", flush=True)


if __name__ == "__main__":
    main()
