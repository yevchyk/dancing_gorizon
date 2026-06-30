"""Evaluation tables and reports for HC models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from . import config as HC
from .folds import FoldSpec, choose_folds


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is None:
        columns = list(df.columns)
    if df.empty:
        return "_No rows._"
    rows = [[_fmt(v) for v in row] for row in df[columns].itertuples(index=False, name=None)]
    widths = [len(c) for c in columns]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(columns, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _load_model(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(path)
    return model


def score_fold(df: pd.DataFrame, model_dir: Path, fold: FoldSpec) -> pd.DataFrame:
    start = fold.start_ts()
    end = fold.end_ts()
    test = df[(df["base_time"] >= start) & (df["base_time"] < end)].copy()
    if test.empty:
        raise RuntimeError(f"{fold.name}: no test rows in dataset")
    fold_dir = model_dir / fold.name
    up = _load_model(fold_dir / "up.cbm")
    down = _load_model(fold_dir / "down.cbm")
    X = test[HC.FEATURE_COLUMNS]
    test["up_prob"] = up.predict_proba(X)[:, 1].astype("float32")
    test["down_prob"] = down.predict_proba(X)[:, 1].astype("float32")
    test["fold"] = fold.name
    return test


def table_a_by_horizon(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, fdf in scored.groupby("fold", sort=False):
        for side, label_col, prob_col in (
            ("UP", "up_label", "up_prob"),
            ("DOWN", "down_label", "down_prob"),
        ):
            for horizon in HC.HORIZON_ANCHORS:
                d = fdf[fdf["horizon_minutes"] == horizon]
                if d.empty:
                    continue
                y = d[label_col].to_numpy("int8")
                p = d[prob_col].to_numpy("float64")
                sig = p >= HC.DECISION_PROB_HIGH
                rows.append(
                    {
                        "fold": fold,
                        "side": side,
                        "horizon": horizon,
                        "n": int(len(d)),
                        "base_rate": float(y.mean()),
                        "auc": _auc(y, p),
                        "signals_070": int(sig.sum()),
                        "precision_070": float(y[sig].mean()) if sig.any() else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def table_b_calibration(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.000001)]
    for fold, fdf in scored.groupby("fold", sort=False):
        for side, label_col, prob_col in (
            ("UP", "up_label", "up_prob"),
            ("DOWN", "down_label", "down_prob"),
        ):
            y_all = fdf[label_col].to_numpy("int8")
            p_all = fdf[prob_col].to_numpy("float64")
            for lo, hi in buckets:
                mask = (p_all >= lo) & (p_all < hi)
                rows.append(
                    {
                        "fold": fold,
                        "side": side,
                        "bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                        "n": int(mask.sum()),
                        "realized_win_rate": float(y_all[mask].mean()) if mask.any() else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def table_c_by_symbol(scored: pd.DataFrame, min_signals: int = HC.MIN_SYMBOL_SIGNALS) -> pd.DataFrame:
    rows = []
    for fold, fdf in scored.groupby("fold", sort=False):
        for side, label_col, prob_col, sign in (
            ("UP", "up_label", "up_prob", 1.0),
            ("DOWN", "down_label", "down_prob", -1.0),
        ):
            base_rate = float(fdf[label_col].mean())
            sig = fdf[fdf[prob_col] >= HC.DECISION_PROB_HIGH].copy()
            if sig.empty:
                continue
            sig["signed_ret_pct"] = sig["ret_pct"] * sign
            grouped = sig.groupby("symbol").agg(
                n=(label_col, "size"),
                precision=(label_col, "mean"),
                avg_prob=(prob_col, "mean"),
                avg_signed_ret_pct=("signed_ret_pct", "mean"),
            )
            grouped = grouped[grouped["n"] >= min_signals]
            if grouped.empty:
                continue
            grouped["edge_vs_base"] = grouped["precision"] - base_rate
            top = grouped.sort_values("edge_vs_base", ascending=False).head(15).assign(rank_group="top")
            bottom = grouped.sort_values("edge_vs_base", ascending=True).head(15).assign(rank_group="bottom")
            out = pd.concat([top, bottom]).reset_index()
            out.insert(0, "fold", fold)
            out.insert(1, "side", side)
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def table_d_decisions(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fee = HC.ROUND_TRIP_FEE_PCT / 100.0
    for fold, fdf in scored.groupby("fold", sort=False):
        up_hi = fdf["up_prob"] >= HC.DECISION_PROB_HIGH
        down_hi = fdf["down_prob"] >= HC.DECISION_PROB_HIGH
        up_lo = fdf["up_prob"] <= HC.DECISION_PROB_LOW
        down_lo = fdf["down_prob"] <= HC.DECISION_PROB_LOW
        decisions = {
            "LONG": up_hi & down_lo,
            "SHORT": down_hi & up_lo,
            "BOTH_HIGH_SKIP": up_hi & down_hi,
            "NO_TRADE": ~(up_hi & down_lo) & ~(down_hi & up_lo) & ~(up_hi & down_hi),
        }
        trade_frames = []
        for action, mask in decisions.items():
            d = fdf[mask].copy()
            if action == "LONG":
                net = d["ret"].to_numpy("float64") - fee
            elif action == "SHORT":
                net = -d["ret"].to_numpy("float64") - fee
            else:
                net = np.array([], dtype="float64")
            if action in {"LONG", "SHORT"}:
                trade_frames.append(pd.DataFrame({"net": net}))
            rows.append(
                {
                    "fold": fold,
                    "action": action,
                    "count": int(mask.sum()),
                    "win_rate": float((net > 0).mean()) if len(net) else float("nan"),
                    "avg_net_ret_pct": float(net.mean() * 100.0) if len(net) else float("nan"),
                }
            )
        if trade_frames:
            all_net = pd.concat(trade_frames, ignore_index=True)["net"].to_numpy("float64")
        else:
            all_net = np.array([], dtype="float64")
        rows.append(
            {
                "fold": fold,
                "action": "ALL_TRADES",
                "count": int(len(all_net)),
                "win_rate": float((all_net > 0).mean()) if len(all_net) else float("nan"),
                "avg_net_ret_pct": float(all_net.mean() * 100.0) if len(all_net) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def edge_survival_statement(table_d: pd.DataFrame) -> str:
    trades = table_d[table_d["action"] == "ALL_TRADES"].copy()
    stress = trades[trades["fold"].str.contains("fold2|fold3", regex=True)]
    if stress.empty:
        return "Stress folds were not evaluated."
    ok = (stress["count"] > 0) & (stress["win_rate"] > 0.5) & (stress["avg_net_ret_pct"] > 0)
    return "YES" if bool(ok.all()) else "NO"


def build_report(
    scored: pd.DataFrame,
    folds: list[FoldSpec],
    *,
    out_md: Path,
    out_csv: Path,
    smoke: bool = False,
) -> dict[str, pd.DataFrame]:
    tables = {
        "A_by_horizon": table_a_by_horizon(scored),
        "B_calibration": table_b_calibration(scored),
        "C_by_symbol": table_c_by_symbol(scored),
        "D_decision_level": table_d_decisions(scored),
    }
    csv_frames = []
    for name, table in tables.items():
        t = table.copy()
        t.insert(0, "table", name)
        csv_frames.append(t)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(csv_frames, ignore_index=True, sort=False).to_csv(out_csv, index=False)

    lines = [
        "# HC Model Results" + (" (Smoke)" if smoke else ""),
        "",
        f"Generated: {pd.Timestamp.utcnow().isoformat()}",
        "",
        "## Folds",
        "",
    ]
    fold_rows = pd.DataFrame([f.to_dict() for f in folds])
    lines.append(markdown_table(fold_rows))
    lines.extend(
        [
            "",
            "## Table A - By Horizon",
            "",
            markdown_table(tables["A_by_horizon"]),
            "",
            "## Table B - Calibration",
            "",
            markdown_table(tables["B_calibration"]),
            "",
            "## Table C - By Symbol",
            "",
            markdown_table(tables["C_by_symbol"]),
            "",
            "## Table D - Decision Level",
            "",
            markdown_table(tables["D_decision_level"]),
            "",
            "## Edge Survival",
            "",
            f"Edge survives folds 2-3: **{edge_survival_statement(tables['D_decision_level'])}**",
            "",
            f"CSV: `{out_csv}`",
            "",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return tables


def load_folds(model_dir: Path, df: pd.DataFrame, max_folds: int) -> list[FoldSpec]:
    snapshot = model_dir / "config_snapshot.json"
    if snapshot.exists():
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        folds = []
        for raw in data.get("folds", []):
            folds.append(FoldSpec(**raw))
        if folds:
            return folds[:max_folds]
    return choose_folds(df, max_folds=max_folds)

