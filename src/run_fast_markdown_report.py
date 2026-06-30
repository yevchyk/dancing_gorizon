"""Build a readable Markdown report from fast_v2 detailed CSV reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .fast import config as FC


OUT = FC.FAST_ANALYSIS_DIR / "detailed_reports"
REPORT = OUT / "fast_v2_untouched_holdout_report.md"


def fmt(v, kind: str = "") -> str:
    if pd.isna(v):
        return "-"
    if kind == "pct":
        return f"{float(v):+.4f}%"
    if kind == "rate":
        return f"{float(v):.3f}"
    if kind == "prob":
        return f"{float(v):.2f}"
    if kind == "money":
        return f"{float(v):+.2f}"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def md_table(df: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    headers = [title for _, title, _ in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = [fmt(row[col], kind) for col, _, kind in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    scores = pd.read_parquet(FC.FAST_ANALYSIS_DIR / "holdout_scores.parquet")
    scores["anchor_time"] = pd.to_datetime(scores["anchor_time"], utc=True)
    summary = pd.read_csv(OUT / "model_summary.csv")
    thresholds = pd.read_csv(OUT / "model_probability_thresholds.csv")
    bins = pd.read_csv(OUT / "model_probability_bins.csv")
    raw = pd.read_csv(OUT / "engine_raw_argmax_thresholds.csv")
    clean = pd.read_csv(OUT / "engine_clean_agreement_sweep.csv")
    scan = pd.read_csv(OUT / "engine_top_per_scan_sweep.csv")
    day = pd.read_csv(OUT / "engine_top_per_day_sweep.csv")
    by_model = pd.read_csv(OUT / "engine_clean_by_model.csv")
    daily = pd.read_csv(OUT / "engine_clean_daily.csv")

    parts: list[str] = []
    parts.append("# fast_v2 Untouched Holdout Report")
    parts.append("")
    parts.append("## Test Window")
    parts.append("")
    parts.append(md_table(pd.DataFrame([{
        "symbols": scores["symbol"].nunique(),
        "rows": len(scores),
        "horizons": scores["horizon"].nunique(),
        "start": str(scores["anchor_time"].min()),
        "end": str(scores["anchor_time"].max()),
        "cost": FC.EVAL_COST * 100,
        "target_edge": FC.TARGET_EDGE * 100,
    }]), [
        ("symbols", "Symbols", ""),
        ("rows", "Rows", ""),
        ("horizons", "Horizons", ""),
        ("start", "Start UTC", ""),
        ("end", "End UTC", ""),
        ("cost", "Cost", "pct"),
        ("target_edge", "Target Edge", "pct"),
    ]))

    parts.append("")
    parts.append("## 1. Model Predictiveness Summary")
    parts.append("")
    model_summary = summary.sort_values(["horizon", "side"])
    parts.append(md_table(model_summary, [
        ("model", "Model", ""),
        ("n", "N", ""),
        ("base_event_rate", "Base Event", "rate"),
        ("base_trade_win", "Base Trade Win", "rate"),
        ("base_avg_pnl", "Base Avg PnL", "pct"),
        ("auc_event", "AUC", "rate"),
        ("spearman_event", "Spearman Event", "rate"),
        ("spearman_pnl", "Spearman PnL", "rate"),
        ("p80", "P80", "rate"),
        ("p90", "P90", "rate"),
        ("p95", "P95", "rate"),
    ]))

    parts.append("")
    parts.append("## 2. Per-Model Threshold Grid")
    parts.append("")
    keep_thr = [0.70, 0.75, 0.80, 0.82, 0.85, 0.90]
    grid = thresholds[thresholds["threshold"].round(2).isin(keep_thr)].copy()
    grid = grid.sort_values(["model", "threshold"])
    parts.append(md_table(grid, [
        ("model", "Model", ""),
        ("threshold", "P >=", "prob"),
        ("n", "N", ""),
        ("coverage", "Coverage", "rate"),
        ("win", "Win", "rate"),
        ("event_hit", "Event Hit", "rate"),
        ("avg_pnl", "Avg PnL", "pct"),
        ("median_pnl", "Median PnL", "pct"),
        ("touch_green", "Touch Green", "rate"),
        ("green_days", "Green Days", ""),
        ("days", "Days", ""),
        ("total", "Total PnL", "money"),
    ]))

    parts.append("")
    parts.append("## 3. Probability Bins By Model")
    parts.append("")
    bin_keep = bins[(bins["bin_lo"] >= 0.70) & (bins["n"] >= 20)].copy()
    bin_keep = bin_keep.sort_values(["model", "bin_lo"])
    parts.append(md_table(bin_keep, [
        ("model", "Model", ""),
        ("bin", "Bin", ""),
        ("n", "N", ""),
        ("avg_prob", "Avg Prob", "rate"),
        ("win", "Win", "rate"),
        ("event_hit", "Event Hit", "rate"),
        ("avg_pnl", "Avg PnL", "pct"),
        ("touch_green", "Touch Green", "rate"),
        ("green_days", "Green Days", ""),
        ("days", "Days", ""),
        ("total", "Total PnL", "money"),
    ]))

    parts.append("")
    parts.append("## 4. Engine Raw Argmax Thresholds")
    parts.append("")
    raw_keep = raw[raw["threshold"].round(2).isin([0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90])].copy()
    parts.append(md_table(raw_keep, [
        ("threshold", "Conf >=", "prob"),
        ("n", "N", ""),
        ("win", "Win", "rate"),
        ("event_hit", "Event Hit", "rate"),
        ("avg_pnl", "Avg PnL", "pct"),
        ("median_pnl", "Median PnL", "pct"),
        ("touch_green", "Touch Green", "rate"),
        ("green_days", "Green Days", ""),
        ("days", "Days", ""),
        ("total", "Total PnL", "money"),
    ]))

    parts.append("")
    parts.append("## 5. Engine Clean Agreement Sweep")
    parts.append("")
    clean_keep = clean[
        (clean["opp_max"].isin([0.30, 0.40, 0.50]))
        & (clean["agree_min"].isin([1, 2, 3]))
        & (clean["floor"].round(2).isin([0.70, 0.75, 0.80, 0.82, 0.85, 0.90]))
    ].copy()
    clean_keep = clean_keep.sort_values(["opp_max", "agree_min", "floor"])
    parts.append(md_table(clean_keep, [
        ("floor", "Floor", "prob"),
        ("opp_max", "Opp <=", "prob"),
        ("agree_min", "Agree >=", ""),
        ("n", "N", ""),
        ("win", "Win", "rate"),
        ("event_hit", "Event Hit", "rate"),
        ("avg_pnl", "Avg PnL", "pct"),
        ("touch_green", "Touch Green", "rate"),
        ("green_days", "Green Days", ""),
        ("days", "Days", ""),
        ("total", "Total PnL", "money"),
    ]))

    parts.append("")
    parts.append("## 6. Engine Top-Per-Scan")
    parts.append("")
    scan_keep = scan[
        (scan["opp_max"] == 0.30)
        & (scan["agree_min"].isin([1, 2]))
        & (scan["top_per_scan"].isin([1, 3, 5, 10]))
        & (scan["floor"].round(2).isin([0.70, 0.75, 0.80, 0.82, 0.85]))
    ].copy()
    scan_keep = scan_keep.sort_values(["agree_min", "floor", "top_per_scan"])
    parts.append(md_table(scan_keep, [
        ("floor", "Floor", "prob"),
        ("opp_max", "Opp <=", "prob"),
        ("agree_min", "Agree >=", ""),
        ("top_per_scan", "Top/Scan", ""),
        ("n", "N", ""),
        ("win", "Win", "rate"),
        ("avg_pnl", "Avg PnL", "pct"),
        ("touch_green", "Touch Green", "rate"),
        ("green_days", "Green Days", ""),
        ("days", "Days", ""),
        ("total", "Total PnL", "money"),
    ]))

    parts.append("")
    parts.append("## 7. Best Engine Top-Per-Day")
    parts.append("")
    day_keep = day[(day["opp_max"] == 0.30) & (day["agree_min"].isin([1, 2]))].copy()
    day_keep = day_keep.sort_values(["avg_pnl", "win", "n"], ascending=[False, False, False]).head(40)
    parts.append(md_table(day_keep, [
        ("floor", "Floor", "prob"),
        ("opp_max", "Opp <=", "prob"),
        ("agree_min", "Agree >=", ""),
        ("top_per_day", "Top/Day", ""),
        ("n", "N", ""),
        ("win", "Win", "rate"),
        ("avg_pnl", "Avg PnL", "pct"),
        ("touch_green", "Touch Green", "rate"),
        ("green_days", "Green Days", ""),
        ("days", "Days", ""),
        ("total", "Total PnL", "money"),
    ]))

    parts.append("")
    parts.append("## 8. Clean Engine By Model")
    parts.append("")
    bm_keep = by_model[
        (by_model["opp_max"] == 0.30)
        & (by_model["agree_min"] == 2)
        & (by_model["floor"].round(2).isin([0.70, 0.75, 0.80, 0.82]))
    ].copy()
    bm_keep = bm_keep.sort_values(["floor", "avg_pnl"], ascending=[True, False])
    parts.append(md_table(bm_keep, [
        ("floor", "Floor", "prob"),
        ("model", "Model", ""),
        ("n", "N", ""),
        ("win", "Win", "rate"),
        ("event_hit", "Event Hit", "rate"),
        ("avg_pnl", "Avg PnL", "pct"),
        ("touch_green", "Touch Green", "rate"),
        ("green_days", "Green Days", ""),
        ("days", "Days", ""),
        ("total", "Total PnL", "money"),
    ]))

    parts.append("")
    parts.append("## 9. Daily Engine Breakdown")
    parts.append("")
    daily_keep = daily[
        (daily["opp_max"] == 0.30)
        & (daily["agree_min"] == 2)
        & (daily["floor"].round(2).isin([0.70, 0.75, 0.80, 0.82]))
    ].copy()
    daily_keep = daily_keep.sort_values(["floor", "day"])
    parts.append(md_table(daily_keep, [
        ("floor", "Floor", "prob"),
        ("day", "Day", ""),
        ("n", "N", ""),
        ("win", "Win", "rate"),
        ("avg_pnl", "Avg PnL", "pct"),
        ("touch_green", "Touch Green", "rate"),
        ("total", "Total PnL", "money"),
    ]))

    parts.append("")
    parts.append("## Files")
    parts.append("")
    for p in sorted(OUT.glob("*.csv")):
        parts.append(f"- `{p.name}`")

    REPORT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
