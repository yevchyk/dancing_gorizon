"""Theoretical signal-bus tests for standard_* and fast_v2_* models.

This is intentionally not a live engine. It builds one row per symbol+time on
the untouched fast_v2 3-day holdout, attaches all fast_v2 and standard direction
probabilities, then tests combinations:
  - how often multiple models scream at the same place
  - whether old+new same-direction agreement helps
  - what happens when up and down both scream
  - simple up/down math: counts, normalized headroom sums, family agreement
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .fast.curve import FastCurve

OUT = FC.FAST_ANALYSIS_DIR / "combined_signal_math"

FAST_THRESHOLDS = {
    "fast_v2_up_2m": 0.92,
    "fast_v2_down_2m": 0.92,
    "fast_v2_up_5m": 0.55,
    "fast_v2_down_5m": 0.85,
    "fast_v2_up_8m": 0.77,
    "fast_v2_down_8m": 0.83,
    "fast_v2_up_10m": 0.77,
    "fast_v2_down_10m": 0.82,
}
STD_FLOORS = (0.75, 0.80, 0.82, 0.85, 0.90)
EXIT_HORIZONS = ("2m", "5m", "8m", "10m")


@dataclass(frozen=True)
class ModelSig:
    family: str
    name: str
    side: int
    horizon: str
    prob_col: str
    threshold: float


def _to_ns(index: pd.DatetimeIndex) -> np.ndarray:
    return index.as_unit("ns").asi8


def _stat(df: pd.DataFrame, side: np.ndarray | int, exit_horizon: str, score_col: str = "") -> dict:
    if len(df) == 0:
        return {
            "n": 0, "win": np.nan, "dir_correct": np.nan, "event_hit": np.nan,
            "avg_pnl": np.nan, "median_pnl": np.nan, "touch_green": np.nan,
            "green_days": 0, "days": 0, "total": 0.0, "avg_score": np.nan,
        }
    side_arr = np.full(len(df), side, dtype=float) if np.isscalar(side) else np.asarray(side, dtype=float)
    ret = df[f"real_ret_{exit_horizon}"].to_numpy(float)
    pnl = side_arr * ret - FC.EVAL_COST
    event = side_arr * ret > FC.TARGET_EDGE
    touch = np.where(
        side_arr == 1,
        df[f"real_mfe_{exit_horizon}"].to_numpy(float) > FC.EVAL_COST,
        -df[f"real_mae_{exit_horizon}"].to_numpy(float) > FC.EVAL_COST,
    )
    daily = pd.DataFrame({"day": df["day"].to_numpy(), "pnl": pnl}).groupby("day")["pnl"].mean() * 100
    return {
        "n": int(len(df)),
        "win": float((pnl > 0).mean()),
        "dir_correct": float((side_arr * ret > 0).mean()),
        "event_hit": float(event.mean()),
        "avg_pnl": float(pnl.mean() * 100),
        "median_pnl": float(np.median(pnl) * 100),
        "touch_green": float(touch.mean()),
        "green_days": int((daily > 0).sum()),
        "days": int(len(daily)),
        "total": float(pnl.sum() * 100),
        "avg_score": float(df[score_col].mean()) if score_col and score_col in df else np.nan,
    }


def _fast_wide() -> pd.DataFrame:
    scores = pd.read_parquet(FC.FAST_ANALYSIS_DIR / "holdout_scores.parquet")
    scores["anchor_time"] = pd.to_datetime(scores["anchor_time"], utc=True)
    scores["day"] = scores["anchor_time"].dt.strftime("%Y-%m-%d")
    base = scores[["symbol", "anchor_time", "day"]].drop_duplicates().reset_index(drop=True)
    for h in FC.HORIZONS:
        lab = h.label
        d = scores[scores["horizon"] == lab][[
            "symbol", "anchor_time", "p_up", "p_down",
            "real_ret", "real_mfe", "real_mae",
        ]].rename(columns={
            "p_up": f"fast_v2_p_up_{lab}",
            "p_down": f"fast_v2_p_down_{lab}",
            "real_ret": f"real_ret_{lab}",
            "real_mfe": f"real_mfe_{lab}",
            "real_mae": f"real_mae_{lab}",
        })
        base = base.merge(d, on=["symbol", "anchor_time"], how="inner")
    return base


def _score_standard(grid: pd.DataFrame) -> pd.DataFrame:
    store = CandleStore(C.CANDLES_DIR)
    curve = FastCurve(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    cols = curve.columns()
    model_dir = C.MODELS_DIR / "dir_prob"
    models = {}
    for h in C.HORIZONS:
        lab = h.label
        models[lab] = {
            "up": joblib.load(model_dir / f"up_{lab}.joblib"),
            "down": joblib.load(model_dir / f"down_{lab}.joblib"),
            "up_cols": joblib.load(model_dir / f"up_{lab}_columns.joblib"),
            "down_cols": joblib.load(model_dir / f"down_{lab}_columns.joblib"),
        }

    recs = []
    for i, (sym, d) in enumerate(grid.groupby("symbol", sort=False), 1):
        candles = store.load(sym)
        if candles is None or candles.empty:
            continue
        candles = candles.sort_index()
        anchors = pd.DatetimeIndex(pd.to_datetime(d["anchor_time"], utc=True))
        anchors_ns = anchors.as_unit("ns").asi8
        feats, valid = curve.build_matrix(
            _to_ns(candles.index),
            candles["close"].to_numpy("float64"),
            anchors_ns,
        )
        if valid.sum() == 0:
            continue
        idx = np.where(valid)[0]
        X = pd.DataFrame(feats[idx], columns=cols)
        part = d.iloc[idx].copy().reset_index(drop=True)
        for h in C.HORIZONS:
            lab = h.label
            m = models[lab]
            part[f"standard_p_up_{lab}"] = m["up"].predict_proba(X[m["up_cols"]])[:, 1]
            part[f"standard_p_down_{lab}"] = m["down"].predict_proba(X[m["down_cols"]])[:, 1]
        recs.append(part)
        if i % 20 == 0 or i == grid["symbol"].nunique():
            print(f"  standard scored {i}/{grid['symbol'].nunique()}", flush=True)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def build_grid(fresh: bool = False) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "combined_signal_grid.parquet"
    if path.exists() and not fresh:
        return pd.read_parquet(path)
    fast = _fast_wide()
    grid = _score_standard(fast)
    grid.to_parquet(path, index=False)
    print(f"combined grid -> {path} rows={len(grid)} cols={grid.shape[1]}")
    return grid


def model_list(std_floor: float) -> list[ModelSig]:
    out: list[ModelSig] = []
    for h in FC.HORIZONS:
        lab = h.label
        for kind, side in (("up", 1), ("down", -1)):
            name = f"fast_v2_{kind}_{lab}"
            out.append(ModelSig("fast_v2", name, side, lab, f"fast_v2_p_{kind}_{lab}",
                                FAST_THRESHOLDS[name]))
    for h in C.HORIZONS:
        lab = h.label
        for kind, side in (("up", 1), ("down", -1)):
            name = f"standard_{kind}_{lab}"
            out.append(ModelSig("standard", name, side, lab, f"standard_p_{kind}_{lab}", std_floor))
    return out


def add_vote_columns(grid: pd.DataFrame, std_floor: float) -> pd.DataFrame:
    g = grid.copy()
    for fam in ("fast_v2", "standard", "all"):
        for side in ("up", "down"):
            g[f"{fam}_{side}_count"] = 0
            g[f"{fam}_{side}_score"] = 0.0
            g[f"{fam}_{side}_max"] = 0.0

    for m in model_list(std_floor):
        p = g[m.prob_col].astype(float)
        active = p >= m.threshold
        side_name = "up" if m.side == 1 else "down"
        headroom = ((p - m.threshold) / max(1e-9, 1.0 - m.threshold)).clip(lower=0)
        for fam in (m.family, "all"):
            g[f"{fam}_{side_name}_count"] += active.astype(int)
            g[f"{fam}_{side_name}_score"] += headroom
            g[f"{fam}_{side_name}_max"] = np.maximum(g[f"{fam}_{side_name}_max"], np.where(active, p, 0.0))

    for fam in ("fast_v2", "standard", "all"):
        g[f"{fam}_net_score"] = g[f"{fam}_up_score"] - g[f"{fam}_down_score"]
        g[f"{fam}_abs_score"] = np.abs(g[f"{fam}_net_score"])
        g[f"{fam}_side"] = np.where(g[f"{fam}_net_score"] >= 0, 1, -1)
        g[f"{fam}_side_count"] = np.where(g[f"{fam}_side"] == 1, g[f"{fam}_up_count"], g[f"{fam}_down_count"])
        g[f"{fam}_opp_count"] = np.where(g[f"{fam}_side"] == 1, g[f"{fam}_down_count"], g[f"{fam}_up_count"])
    return g


def frequency_report(g: pd.DataFrame, std_floor: float) -> pd.DataFrame:
    rows = []
    n = len(g)
    cases = {
        "any_fast": g["fast_v2_up_count"] + g["fast_v2_down_count"] > 0,
        "any_standard": g["standard_up_count"] + g["standard_down_count"] > 0,
        "any_both_families": ((g["fast_v2_up_count"] + g["fast_v2_down_count"] > 0)
                              & (g["standard_up_count"] + g["standard_down_count"] > 0)),
        "same_up_both_families": (g["fast_v2_up_count"] > 0) & (g["standard_up_count"] > 0),
        "same_down_both_families": (g["fast_v2_down_count"] > 0) & (g["standard_down_count"] > 0),
        "fast_conflict_up_down": (g["fast_v2_up_count"] > 0) & (g["fast_v2_down_count"] > 0),
        "standard_conflict_up_down": (g["standard_up_count"] > 0) & (g["standard_down_count"] > 0),
        "all_conflict_up_down": (g["all_up_count"] > 0) & (g["all_down_count"] > 0),
    }
    for name, mask in cases.items():
        rows.append({"std_floor": std_floor, "case": name, "n": int(mask.sum()),
                     "pct_places": float(mask.mean() * 100)})
    for k in range(1, 8):
        rows.append({"std_floor": std_floor, "case": f"all_up_count>={k}",
                     "n": int((g["all_up_count"] >= k).sum()),
                     "pct_places": float((g["all_up_count"] >= k).mean() * 100)})
        rows.append({"std_floor": std_floor, "case": f"all_down_count>={k}",
                     "n": int((g["all_down_count"] >= k).sum()),
                     "pct_places": float((g["all_down_count"] >= k).mean() * 100)})
    rows.append({"std_floor": std_floor, "case": "total_places", "n": n, "pct_places": 100.0})
    return pd.DataFrame(rows)


def evaluate_rules(g: pd.DataFrame, std_floor: float) -> pd.DataFrame:
    rows = []

    def add(rule: str, mask, side, exit_h: str, score_col: str = "", **extra) -> None:
        d = g[mask].copy()
        side_arr = side[mask] if hasattr(side, "__len__") and not np.isscalar(side) else side
        r = _stat(d, side_arr, exit_h, score_col)
        r.update({"std_floor": std_floor, "rule": rule, "exit": exit_h})
        r.update(extra)
        rows.append(r)

    for exit_h in EXIT_HORIZONS:
        fast_any = g["fast_v2_side_count"] > 0
        std_any = g["standard_side_count"] > 0
        all_any = g["all_side_count"] > 0
        add("fast_score_side_any", fast_any, g["fast_v2_side"].to_numpy(), exit_h, "fast_v2_abs_score")
        add("standard_score_side_any", std_any, g["standard_side"].to_numpy(), exit_h, "standard_abs_score")
        add("combined_score_side_any", all_any, g["all_side"].to_numpy(), exit_h, "all_abs_score")

        agree = fast_any & std_any & (g["fast_v2_side"] == g["standard_side"])
        add("family_agree_score_side", agree, g["all_side"].to_numpy(), exit_h, "all_abs_score")

        conflict = (g["all_up_count"] > 0) & (g["all_down_count"] > 0)
        add("conflict_choose_stronger", conflict, g["all_side"].to_numpy(), exit_h, "all_abs_score")

        no_conflict = all_any & ~conflict
        add("no_conflict_any_scream", no_conflict, g["all_side"].to_numpy(), exit_h, "all_abs_score")

        for k in (2, 3, 4, 5, 6):
            up = g["all_up_count"] >= k
            down = g["all_down_count"] >= k
            add(f"multi_up_count>={k}", up, 1, exit_h, "all_up_score", k=k)
            add(f"multi_down_count>={k}", down, -1, exit_h, "all_down_score", k=k)
            clean_up = up & (g["all_down_count"] == 0)
            clean_down = down & (g["all_up_count"] == 0)
            add(f"clean_multi_up_count>={k}", clean_up, 1, exit_h, "all_up_score", k=k)
            add(f"clean_multi_down_count>={k}", clean_down, -1, exit_h, "all_down_score", k=k)

        for f in (1, 2, 3):
            for s in (1, 2, 3):
                up = (g["fast_v2_up_count"] >= f) & (g["standard_up_count"] >= s)
                down = (g["fast_v2_down_count"] >= f) & (g["standard_down_count"] >= s)
                add(f"both_families_up_fast{f}_std{s}", up, 1, exit_h, "all_up_score",
                    fast_min=f, standard_min=s)
                add(f"both_families_down_fast{f}_std{s}", down, -1, exit_h, "all_down_score",
                    fast_min=f, standard_min=s)

        for k in (20, 50, 100, 200):
            d = g[all_any].copy()
            if len(d):
                d["_rank"] = d.groupby("day")["all_abs_score"].rank(ascending=False, method="first")
                mask = d["_rank"] <= k
                r = _stat(d[mask], d.loc[mask, "all_side"].to_numpy(), exit_h, "all_abs_score")
            else:
                r = _stat(d, 1, exit_h, "all_abs_score")
            r.update({"std_floor": std_floor, "rule": f"top{k}/day_combined_score",
                      "exit": exit_h, "top_per_day": k})
            rows.append(r)

    return pd.DataFrame(rows)


def top_rules_table(rules: pd.DataFrame) -> pd.DataFrame:
    x = rules[(rules["n"] >= 20) & (rules["days"] >= 3)].copy()
    return x.sort_values(["avg_pnl", "win", "n"], ascending=[False, False, False]).head(40)


def print_summary(freq: pd.DataFrame, rules: pd.DataFrame) -> None:
    print("=== FREQUENCY std_floor=0.82 ===")
    f = freq[freq["std_floor"] == 0.82]
    keep = [
        "total_places", "any_fast", "any_standard", "any_both_families",
        "same_up_both_families", "same_down_both_families",
        "all_conflict_up_down", "all_up_count>=2", "all_down_count>=2",
        "all_up_count>=3", "all_down_count>=3", "all_up_count>=4", "all_down_count>=4",
    ]
    print(f[f["case"].isin(keep)][["case", "n", "pct_places"]].to_string(
        index=False, formatters={"pct_places": "{:.2f}%".format}))

    print("\n=== BEST THEORETICAL RULES (n>=20, days>=3) ===")
    best = top_rules_table(rules)
    print(best[["std_floor", "rule", "exit", "n", "win", "dir_correct", "event_hit",
                "avg_pnl", "touch_green", "green_days", "days", "total", "avg_score"]]
          .to_string(index=False, formatters={
              "std_floor": "{:.2f}".format,
              "win": "{:.3f}".format,
              "dir_correct": "{:.3f}".format,
              "event_hit": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
              "total": "{:+.2f}".format,
              "avg_score": "{:.3f}".format,
          }))

    print("\n=== COMBINED SCORE TOP/DAY ===")
    top = rules[rules["rule"].str.contains("top", regex=False)].copy()
    top = top[(top["std_floor"].isin([0.80, 0.82, 0.85])) & (top["exit"].isin(["5m", "8m", "10m"]))]
    top = top.sort_values(["avg_pnl", "win"], ascending=[False, False]).head(30)
    print(top[["std_floor", "rule", "exit", "n", "win", "avg_pnl", "touch_green",
               "green_days", "days", "total", "avg_score"]].to_string(index=False, formatters={
                   "std_floor": "{:.2f}".format,
                   "win": "{:.3f}".format,
                   "avg_pnl": "{:+.4f}".format,
                   "touch_green": "{:.3f}".format,
                   "total": "{:+.2f}".format,
                   "avg_score": "{:.3f}".format,
               }))

    print("\n=== FAMILY AGREEMENT / CONFLICT ===")
    focus = rules[rules["rule"].isin([
        "family_agree_score_side", "conflict_choose_stronger",
        "no_conflict_any_scream", "combined_score_side_any",
    ])].copy()
    focus = focus[(focus["std_floor"].isin([0.80, 0.82, 0.85])) & (focus["exit"].isin(["5m", "8m", "10m"]))]
    print(focus[["std_floor", "rule", "exit", "n", "win", "avg_pnl", "touch_green",
                 "green_days", "days", "total"]].to_string(index=False, formatters={
                     "std_floor": "{:.2f}".format,
                     "win": "{:.3f}".format,
                     "avg_pnl": "{:+.4f}".format,
                     "touch_green": "{:.3f}".format,
                     "total": "{:+.2f}".format,
                 }))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    grid = build_grid(fresh=args.fresh)
    print(f"grid rows={len(grid)} symbols={grid['symbol'].nunique()} "
          f"{grid['anchor_time'].min()} -> {grid['anchor_time'].max()}")

    freqs = []
    rules = []
    for std_floor in STD_FLOORS:
        g = add_vote_columns(grid, std_floor)
        freqs.append(frequency_report(g, std_floor))
        rules.append(evaluate_rules(g, std_floor))
    freq = pd.concat(freqs, ignore_index=True)
    rule = pd.concat(rules, ignore_index=True)
    freq.to_csv(OUT / "scream_frequency.csv", index=False)
    rule.to_csv(OUT / "theoretical_rules.csv", index=False)
    top_rules_table(rule).to_csv(OUT / "best_theoretical_rules.csv", index=False)
    print_summary(freq, rule)
    print(f"\nreports -> {OUT}")


if __name__ == "__main__":
    main()
