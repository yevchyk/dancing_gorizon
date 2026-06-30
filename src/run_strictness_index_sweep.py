"""Global strictness-index sweep for worthy fast_v2 models.

Each worthy model has a base probability threshold. A global index moves every
threshold closer to 1.0:

    adjusted = base + index * (1 - base)

index=0.00 keeps base thresholds, index=0.50 moves each threshold halfway to
1.0. This gives one knob for signal quantity vs quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fast import config as FC

OUT = FC.FAST_ANALYSIS_DIR / "strictness_index"
GRID = FC.FAST_ANALYSIS_DIR / "combined_signal_math" / "combined_signal_grid.parquet"

WORTHY = {
    "fast_v2_up_10m": ("fast_v2_p_up_10m", 1, 0.77),
    "fast_v2_up_8m": ("fast_v2_p_up_8m", 1, 0.77),
    "fast_v2_up_2m": ("fast_v2_p_up_2m", 1, 0.92),
    "fast_v2_down_10m": ("fast_v2_p_down_10m", -1, 0.82),
    "fast_v2_down_8m": ("fast_v2_p_down_8m", -1, 0.83),
    "fast_v2_down_2m": ("fast_v2_p_down_2m", -1, 0.92),
}
EXIT_HORIZONS = ("2m", "5m", "8m", "10m")
INDEXES = np.round(np.arange(0.0, 0.801, 0.05), 2)


def stat(d: pd.DataFrame, side, exit_h: str, score_col: str = "") -> dict:
    if len(d) == 0:
        return {"n": 0, "signals_per_24h": 0.0, "win": np.nan, "avg_pnl": np.nan,
                "touch_green": np.nan, "green_days": 0, "days": 0, "total": 0.0,
                "avg_score": np.nan}
    side_arr = np.asarray(side, dtype=float)
    ret = d[f"real_ret_{exit_h}"].to_numpy(float)
    pnl = side_arr * ret - FC.EVAL_COST
    touch = np.where(
        side_arr == 1,
        d[f"real_mfe_{exit_h}"].to_numpy(float) > FC.EVAL_COST,
        -d[f"real_mae_{exit_h}"].to_numpy(float) > FC.EVAL_COST,
    )
    daily = pd.DataFrame({"day": d["day"].to_numpy(), "pnl": pnl}).groupby("day")["pnl"].mean() * 100
    hours = max(1.0, (pd.to_datetime(d["anchor_time"], utc=True).max()
                      - pd.to_datetime(d["anchor_time"], utc=True).min()).total_seconds() / 3600)
    return {
        "n": int(len(d)),
        "signals_per_24h": float(len(d) / hours * 24),
        "win": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean() * 100),
        "touch_green": float(touch.mean()),
        "green_days": int((daily > 0).sum()),
        "days": int(len(daily)),
        "total": float(pnl.sum() * 100),
        "avg_score": float(d[score_col].mean()) if score_col else np.nan,
    }


def add_votes(g: pd.DataFrame, index: float) -> tuple[pd.DataFrame, dict[str, float]]:
    x = g.copy()
    x["up_count"] = 0
    x["down_count"] = 0
    x["up_score"] = 0.0
    x["down_score"] = 0.0
    thresholds = {}
    for name, (col, side, base) in WORTHY.items():
        thr = base + index * (1.0 - base)
        thresholds[name] = float(thr)
        p = x[col].astype(float)
        active = p >= thr
        headroom = ((p - thr) / max(1e-9, 1.0 - thr)).clip(lower=0)
        if side == 1:
            x["up_count"] += active.astype(int)
            x["up_score"] += headroom
        else:
            x["down_count"] += active.astype(int)
            x["down_score"] += headroom
    x["net_score"] = x["up_score"] - x["down_score"]
    x["abs_score"] = np.abs(x["net_score"])
    x["side"] = np.where(x["net_score"] >= 0, 1, -1)
    x["side_count"] = np.where(x["side"] == 1, x["up_count"], x["down_count"])
    x["opp_count"] = np.where(x["side"] == 1, x["down_count"], x["up_count"])
    x["any_signal"] = (x["up_count"] + x["down_count"]) > 0
    x["no_conflict"] = ~((x["up_count"] > 0) & (x["down_count"] > 0))
    return x, thresholds


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(GRID)
    g["anchor_time"] = pd.to_datetime(g["anchor_time"], utc=True)
    rows = []
    threshold_rows = []

    for idx in INDEXES:
        x, thrs = add_votes(g, float(idx))
        threshold_rows.append({"index": idx, **thrs})
        masks = {
            "any_worthy": x["any_signal"],
            "no_conflict_any": x["any_signal"] & x["no_conflict"],
            "side_count>=2": x["side_count"] >= 2,
            "clean_side_count>=2": (x["side_count"] >= 2) & (x["opp_count"] == 0),
            "side_count>=3": x["side_count"] >= 3,
            "clean_side_count>=3": (x["side_count"] >= 3) & (x["opp_count"] == 0),
        }
        for exit_h in EXIT_HORIZONS:
            for rule, mask in masks.items():
                d = x[mask].copy()
                r = stat(d, d["side"].to_numpy(), exit_h, "abs_score")
                r.update({"index": idx, "rule": rule, "exit": exit_h})
                rows.append(r)
            for k in (20, 50, 100, 200):
                d = x[x["any_signal"]].copy()
                if len(d):
                    d["_rank"] = d.groupby("day")["abs_score"].rank(ascending=False, method="first")
                    d = d[d["_rank"] <= k]
                r = stat(d, d["side"].to_numpy(), exit_h, "abs_score")
                r.update({"index": idx, "rule": f"top{k}/day", "exit": exit_h})
                rows.append(r)

    sweep = pd.DataFrame(rows)
    thresholds = pd.DataFrame(threshold_rows)
    sweep.to_csv(OUT / "strictness_index_sweep.csv", index=False)
    thresholds.to_csv(OUT / "strictness_index_thresholds.csv", index=False)

    print("=== GLOBAL STRICTNESS INDEX: CLEAN SIDE_COUNT>=2 ===")
    focus = sweep[(sweep["rule"] == "clean_side_count>=2") & (sweep["exit"].isin(["5m", "8m", "10m"]))]
    focus = focus[focus["index"].isin([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
    print(focus[["index", "exit", "n", "signals_per_24h", "win", "avg_pnl", "touch_green", "green_days", "days", "total"]]
          .to_string(index=False, formatters={
              "index": "{:.2f}".format,
              "signals_per_24h": "{:.1f}".format,
              "win": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
              "total": "{:+.2f}".format,
          }))

    print("\n=== GLOBAL STRICTNESS INDEX: TOP20/DAY ===")
    top = sweep[(sweep["rule"] == "top20/day") & (sweep["exit"].isin(["5m", "8m", "10m"]))]
    top = top[top["index"].isin([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
    print(top[["index", "exit", "n", "signals_per_24h", "win", "avg_pnl", "touch_green", "green_days", "days", "total", "avg_score"]]
          .to_string(index=False, formatters={
              "index": "{:.2f}".format,
              "signals_per_24h": "{:.1f}".format,
              "win": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
              "total": "{:+.2f}".format,
              "avg_score": "{:.3f}".format,
          }))

    print("\n=== BEST INDEX/RULE COMBOS n>=20 days>=3 ===")
    best = sweep[(sweep["n"] >= 20) & (sweep["days"] >= 3)].sort_values(
        ["avg_pnl", "win", "n"], ascending=[False, False, False]
    ).head(40)
    print(best[["index", "rule", "exit", "n", "signals_per_24h", "win", "avg_pnl", "touch_green", "green_days", "days", "total"]]
          .to_string(index=False, formatters={
              "index": "{:.2f}".format,
              "signals_per_24h": "{:.1f}".format,
              "win": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
              "total": "{:+.2f}".format,
          }))
    print(f"\nreports -> {OUT}")


if __name__ == "__main__":
    main()
