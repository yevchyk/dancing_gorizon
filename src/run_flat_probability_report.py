"""Flat probability reports for fast_v2 and standard direction models.

This is deliberately not an engine simulation. It asks the simplest question:
if one model alone fires above a probability level, what is the realized result?

Outputs:
* flat_probability_thresholds.csv  - prob >= threshold sweep, every 0.01
* flat_probability_bins.csv        - calibration-style probability buckets
* flat_probability_model_summary.csv
* flat_probability_report.md       - compact human report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .fast import config as FC
from .fast.candles import load_1m
from .trading.timeutil import NS_PER_MIN, index_to_ns

OUT = FC.FAST_ANALYSIS_DIR / "flat_probability"
GRID = FC.FAST_ANALYSIS_DIR / "combined_signal_math" / "combined_signal_grid.parquet"
THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.01), 2)
REPORT_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.92, 0.95)
BIN_EDGES = np.round(np.arange(0.40, 1.0001, 0.05), 2)

STD_HORIZON_MIN = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
}
FAST_HORIZON_MIN = {
    "2m": 2,
    "5m": 5,
    "8m": 8,
    "10m": 10,
}


@dataclass(frozen=True)
class FlatModel:
    family: str
    model: str
    side_name: str
    side: int
    horizon: str
    horizon_min: int
    prob_col: str
    opp_col: str
    ret_col: str
    mfe_col: str
    mae_col: str


def _empty_targets(n: int) -> dict[str, np.ndarray]:
    return {
        "ret": np.full(n, np.nan, dtype="float64"),
        "mfe": np.full(n, np.nan, dtype="float64"),
        "mae": np.full(n, np.nan, dtype="float64"),
    }


def _targets_for_group(symbol: str, anchors: pd.Series, horizons: dict[str, int]) -> dict[str, dict[str, np.ndarray]]:
    candles = load_1m(symbol)
    if candles is None or candles.empty:
        return {label: _empty_targets(len(anchors)) for label in horizons}
    candles = candles.sort_index()
    ts = index_to_ns(candles.index)
    high = candles["high"].to_numpy("float64")
    low = candles["low"].to_numpy("float64")
    close = candles["close"].to_numpy("float64")
    a_ns = pd.to_datetime(anchors, utc=True).dt.as_unit("ns").astype("int64").to_numpy()

    out: dict[str, dict[str, np.ndarray]] = {}
    for label, minutes in horizons.items():
        ret = np.full(len(a_ns), np.nan, dtype="float64")
        mfe = np.full(len(a_ns), np.nan, dtype="float64")
        mae = np.full(len(a_ns), np.nan, dtype="float64")
        entry_idx = np.searchsorted(ts, a_ns, side="right") - 1
        end_idx = np.searchsorted(ts, a_ns + minutes * NS_PER_MIN, side="right") - 1
        for i, (ei, fi) in enumerate(zip(entry_idx, end_idx)):
            if ei < 0 or fi <= ei or fi >= len(close):
                continue
            entry = close[ei]
            if not np.isfinite(entry) or entry <= 0:
                continue
            ret[i] = close[fi] / entry - 1.0
            mfe[i] = np.nanmax(high[ei + 1:fi + 1]) / entry - 1.0
            mae[i] = np.nanmin(low[ei + 1:fi + 1]) / entry - 1.0
        out[label] = {"ret": ret, "mfe": mfe, "mae": mae}
    return out


def add_standard_targets(grid: pd.DataFrame) -> pd.DataFrame:
    needed = [h for h in STD_HORIZON_MIN if f"standard_real_ret_{h}" not in grid.columns]
    if not needed:
        return grid

    parts = []
    horizons = {h: STD_HORIZON_MIN[h] for h in needed}
    for idx, (symbol, g) in enumerate(grid.groupby("symbol", sort=False), 1):
        part = g.copy()
        targets = _targets_for_group(symbol, part["anchor_time"], horizons)
        for label in needed:
            part[f"standard_real_ret_{label}"] = targets[label]["ret"]
            part[f"standard_real_mfe_{label}"] = targets[label]["mfe"]
            part[f"standard_real_mae_{label}"] = targets[label]["mae"]
        parts.append(part)
        if idx % 25 == 0 or idx == grid["symbol"].nunique():
            print(f"  standard targets {idx}/{grid['symbol'].nunique()}", flush=True)
    return pd.concat(parts, ignore_index=True)


def model_specs() -> list[FlatModel]:
    models: list[FlatModel] = []
    for label, minutes in FAST_HORIZON_MIN.items():
        for side_name, side in (("up", 1), ("down", -1)):
            other = "down" if side_name == "up" else "up"
            models.append(FlatModel(
                "fast_v2",
                f"fast_v2_{side_name}_{label}",
                side_name,
                side,
                label,
                minutes,
                f"fast_v2_p_{side_name}_{label}",
                f"fast_v2_p_{other}_{label}",
                f"real_ret_{label}",
                f"real_mfe_{label}",
                f"real_mae_{label}",
            ))
    for label, minutes in STD_HORIZON_MIN.items():
        for side_name, side in (("up", 1), ("down", -1)):
            other = "down" if side_name == "up" else "up"
            models.append(FlatModel(
                "standard",
                f"standard_{side_name}_{label}",
                side_name,
                side,
                label,
                minutes,
                f"standard_p_{side_name}_{label}",
                f"standard_p_{other}_{label}",
                f"standard_real_ret_{label}",
                f"standard_real_mfe_{label}",
                f"standard_real_mae_{label}",
            ))
    return models


def _stat(d: pd.DataFrame) -> dict:
    if d.empty:
        return {
            "n": 0,
            "signals_per_24h": 0.0,
            "win": np.nan,
            "dir_correct": np.nan,
            "event_hit": np.nan,
            "avg_pnl": np.nan,
            "median_pnl": np.nan,
            "p10_pnl": np.nan,
            "p90_pnl": np.nan,
            "touch_green": np.nan,
            "avg_prob": np.nan,
            "avg_opp": np.nan,
            "avg_spread": np.nan,
            "green_days": 0,
            "days": 0,
            "total_pnl": 0.0,
            "symbols": 0,
        }
    t = pd.to_datetime(d["anchor_time"], utc=True)
    hours = max(1.0, (t.max() - t.min()).total_seconds() / 3600.0)
    daily = d.groupby("day")["pnl"].mean() * 100.0
    pnl_pct = d["pnl"].to_numpy("float64") * 100.0
    return {
        "n": int(len(d)),
        "signals_per_24h": float(len(d) / hours * 24.0),
        "win": float((d["pnl"] > 0).mean()),
        "dir_correct": float(d["dir_correct"].mean()),
        "event_hit": float(d["event_hit"].mean()),
        "avg_pnl": float(np.nanmean(pnl_pct)),
        "median_pnl": float(np.nanmedian(pnl_pct)),
        "p10_pnl": float(np.nanquantile(pnl_pct, 0.10)),
        "p90_pnl": float(np.nanquantile(pnl_pct, 0.90)),
        "touch_green": float(d["touch_green"].mean()),
        "avg_prob": float(d["prob"].mean()),
        "avg_opp": float(d["opp"].mean()),
        "avg_spread": float(d["spread"].mean()),
        "green_days": int((daily > 0).sum()),
        "days": int(len(daily)),
        "total_pnl": float(np.nansum(pnl_pct)),
        "symbols": int(d["symbol"].nunique()),
    }


def flat_candidates(grid: pd.DataFrame, spec: FlatModel) -> pd.DataFrame:
    cols = [
        "symbol", "anchor_time", "day", spec.prob_col, spec.opp_col,
        spec.ret_col, spec.mfe_col, spec.mae_col,
    ]
    d = grid[cols].copy()
    d = d.rename(columns={
        spec.prob_col: "prob",
        spec.opp_col: "opp",
        spec.ret_col: "ret",
        spec.mfe_col: "mfe",
        spec.mae_col: "mae",
    })
    d = d.dropna(subset=["prob", "ret", "mfe", "mae"])
    d["family"] = spec.family
    d["model"] = spec.model
    d["side_name"] = spec.side_name
    d["side"] = spec.side
    d["horizon"] = spec.horizon
    d["horizon_min"] = spec.horizon_min
    d["opp"] = d["opp"].astype(float)
    d["prob"] = d["prob"].astype(float)
    d["spread"] = d["prob"] - d["opp"]
    d["pnl"] = spec.side * d["ret"].astype(float) - FC.EVAL_COST
    d["dir_correct"] = (spec.side * d["ret"].astype(float) > 0).astype(int)
    d["event_hit"] = (spec.side * d["ret"].astype(float) > FC.TARGET_EDGE).astype(int)
    d["touch_green"] = np.where(
        spec.side == 1,
        d["mfe"].astype(float).to_numpy() > FC.EVAL_COST,
        -d["mae"].astype(float).to_numpy() > FC.EVAL_COST,
    ).astype(int)
    return d


def build_reports(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_rows = []
    bin_rows = []
    summary_rows = []

    for spec in model_specs():
        d = flat_candidates(grid, spec)
        if d.empty:
            continue
        base = _stat(d)
        base.update({
            "family": spec.family,
            "model": spec.model,
            "side": spec.side_name,
            "horizon": spec.horizon,
            "horizon_min": spec.horizon_min,
            "prob_p80": float(d["prob"].quantile(0.80)),
            "prob_p90": float(d["prob"].quantile(0.90)),
            "prob_p95": float(d["prob"].quantile(0.95)),
        })
        summary_rows.append(base)

        for thr in THRESHOLDS:
            z = d[d["prob"] >= thr]
            row = _stat(z)
            row.update({
                "family": spec.family,
                "model": spec.model,
                "side": spec.side_name,
                "horizon": spec.horizon,
                "horizon_min": spec.horizon_min,
                "threshold": thr,
                "coverage": float(len(z) / len(d)),
            })
            threshold_rows.append(row)

        for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            if hi < 1.0:
                z = d[(d["prob"] >= lo) & (d["prob"] < hi)]
                label = f"[{lo:.2f},{hi:.2f})"
            else:
                z = d[(d["prob"] >= lo) & (d["prob"] <= hi)]
                label = f"[{lo:.2f},{hi:.2f}]"
            row = _stat(z)
            row.update({
                "family": spec.family,
                "model": spec.model,
                "side": spec.side_name,
                "horizon": spec.horizon,
                "horizon_min": spec.horizon_min,
                "bin": label,
                "bin_lo": lo,
                "bin_hi": hi,
            })
            bin_rows.append(row)

    thresholds = pd.DataFrame(threshold_rows).sort_values(["family", "model", "threshold"])
    bins = pd.DataFrame(bin_rows).sort_values(["family", "model", "bin_lo"])
    summary = pd.DataFrame(summary_rows).sort_values(["family", "horizon_min", "side"])
    return summary, thresholds, bins


def best_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, d in thresholds.groupby("model", sort=True):
        d = d.copy()
        profiles = {
            "active": d[(d["n"] >= 500) & (d["avg_pnl"] > 0) & (d["green_days"] >= 2)],
            "balanced": d[(d["n"] >= 100) & (d["avg_pnl"] > 0.05) & (d["green_days"] >= 2)],
            "precision": d[(d["n"] >= 20) & (d["avg_pnl"] > 0.15) & (d["green_days"] >= 1)],
        }
        for profile, p in profiles.items():
            if p.empty:
                continue
            p = p.assign(score=p["avg_pnl"] * np.sqrt(p["n"].clip(lower=1)))
            row = p.sort_values(["score", "avg_pnl", "n"], ascending=[False, False, False]).iloc[0].to_dict()
            row["profile"] = profile
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["avg_pnl", "n"], ascending=[False, False])


def _fmt_table(df: pd.DataFrame, cols: list[str]) -> str:
    fmt = {
        "threshold": "{:.2f}".format,
        "signals_per_24h": "{:.1f}".format,
        "win": "{:.3f}".format,
        "dir_correct": "{:.3f}".format,
        "event_hit": "{:.3f}".format,
        "avg_pnl": "{:+.4f}".format,
        "median_pnl": "{:+.4f}".format,
        "p10_pnl": "{:+.3f}".format,
        "p90_pnl": "{:+.3f}".format,
        "touch_green": "{:.3f}".format,
        "avg_prob": "{:.3f}".format,
        "avg_opp": "{:.3f}".format,
        "avg_spread": "{:.3f}".format,
        "total_pnl": "{:+.2f}".format,
        "coverage": "{:.4f}".format,
        "prob_p90": "{:.3f}".format,
        "base_avg_pnl": "{:+.4f}".format,
    }
    return df[cols].to_string(index=False, formatters={k: v for k, v in fmt.items() if k in cols})


def write_markdown(summary: pd.DataFrame, thresholds: pd.DataFrame, bins: pd.DataFrame,
                   best: pd.DataFrame, out_dir: Path) -> None:
    key = thresholds[thresholds["threshold"].isin(REPORT_THRESHOLDS)].copy()
    strong = key[(key["n"] >= 20) & (key["avg_pnl"] > 0)].sort_values(
        ["avg_pnl", "win", "n"], ascending=[False, False, False]
    ).head(40)
    core_models = [
        "fast_v2_up_2m",
        "fast_v2_down_2m",
        "fast_v2_up_8m",
        "fast_v2_down_8m",
        "fast_v2_up_10m",
        "fast_v2_down_10m",
        "standard_up_5m",
        "standard_down_5m",
        "standard_up_15m",
        "standard_down_15m",
    ]
    core = key[key["model"].isin(core_models)].sort_values(["model", "threshold"])
    bucket_focus = bins[
        (bins["model"].isin(core_models[:6]))
        & (bins["bin_lo"] >= 0.70)
        & (bins["n"] > 0)
    ].sort_values(["model", "bin_lo"])

    lines = [
        "# Flat Probability Report",
        "",
        f"Window: `{thresholds['anchor_window'].iloc[0] if 'anchor_window' in thresholds else 'untouched fast_v2 holdout'}`",
        f"Cost: `{FC.EVAL_COST * 100:.3f}%` roundtrip fee+slip. No engine filters, no cooldown, no max-open.",
        "",
        "## Best Positive Flat Thresholds",
        "",
        "```text",
        _fmt_table(strong, [
            "model", "threshold", "n", "signals_per_24h", "win", "dir_correct",
            "avg_pnl", "touch_green", "green_days", "days", "symbols", "total_pnl",
        ]),
        "```",
        "",
        "## Profile Picks",
        "",
    ]
    if best.empty:
        lines.extend(["No profile picks.", ""])
    else:
        show = best.head(36)
        lines.extend([
            "```text",
            _fmt_table(show, [
                "model", "profile", "threshold", "n", "signals_per_24h", "win",
                "avg_pnl", "touch_green", "green_days", "days", "symbols",
            ]),
            "```",
            "",
        ])

    lines.extend([
        "## Core Models By Probability Level",
        "",
        "```text",
        _fmt_table(core, [
            "model", "threshold", "n", "signals_per_24h", "win", "dir_correct",
            "avg_pnl", "touch_green", "green_days", "days",
        ]),
        "```",
        "",
        "## Probability Buckets For Fast Core Models",
        "",
        "```text",
        _fmt_table(bucket_focus, [
            "model", "bin", "n", "win", "dir_correct", "avg_pnl",
            "touch_green", "avg_prob", "avg_opp", "green_days", "days",
        ]),
        "```",
        "",
        "## Files",
        "",
        "* `flat_probability_model_summary.csv`",
        "* `flat_probability_thresholds.csv`",
        "* `flat_probability_bins.csv`",
        "* `flat_probability_best_profiles.csv`",
    ])
    (out_dir / "flat_probability_report.md").write_text("\n".join(lines), encoding="utf-8")


def print_report(summary: pd.DataFrame, thresholds: pd.DataFrame, best: pd.DataFrame) -> None:
    tmin = pd.to_datetime(thresholds["window_start"].iloc[0], utc=True)
    tmax = pd.to_datetime(thresholds["window_end"].iloc[0], utc=True)
    print("=== FLAT PROBABILITY REPORT ===")
    print(f"window: {tmin} -> {tmax} ({(tmax - tmin).total_seconds() / 3600:.1f}h)")
    print(f"models={summary['model'].nunique()} cost={FC.EVAL_COST * 100:.3f}%")

    key = thresholds[thresholds["threshold"].isin(REPORT_THRESHOLDS)].copy()
    strong = key[(key["n"] >= 20) & (key["avg_pnl"] > 0)].sort_values(
        ["avg_pnl", "win", "n"], ascending=[False, False, False]
    ).head(30)
    print("\n=== BEST POSITIVE FLAT LEVELS (key thresholds, n>=20) ===")
    print(_fmt_table(strong, [
        "model", "threshold", "n", "signals_per_24h", "win", "dir_correct",
        "avg_pnl", "touch_green", "green_days", "days", "total_pnl",
    ]))

    print("\n=== BEST PROFILE PICKS ===")
    if best.empty:
        print("(empty)")
    else:
        print(_fmt_table(best.head(30), [
            "model", "profile", "threshold", "n", "signals_per_24h", "win",
            "avg_pnl", "touch_green", "green_days", "days", "symbols",
        ]))

    print(f"\nreports -> {OUT.resolve()}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID)
    grid["anchor_time"] = pd.to_datetime(grid["anchor_time"], utc=True)
    grid["day"] = grid["anchor_time"].dt.strftime("%Y-%m-%d")
    grid = add_standard_targets(grid)

    summary, thresholds, bins = build_reports(grid)
    window_start = grid["anchor_time"].min()
    window_end = grid["anchor_time"].max()
    anchor_window = f"{window_start} -> {window_end}"
    for df in (summary, thresholds, bins):
        df["window_start"] = window_start
        df["window_end"] = window_end
        df["anchor_window"] = anchor_window

    best = best_thresholds(thresholds)
    summary.to_csv(OUT / "flat_probability_model_summary.csv", index=False)
    thresholds.to_csv(OUT / "flat_probability_thresholds.csv", index=False)
    bins.to_csv(OUT / "flat_probability_bins.csv", index=False)
    best.to_csv(OUT / "flat_probability_best_profiles.csv", index=False)
    write_markdown(summary, thresholds, bins, best, OUT)
    print_report(summary, thresholds, best)


if __name__ == "__main__":
    main()
