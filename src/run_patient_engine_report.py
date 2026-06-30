"""Patient engine report: no first-green harvest, wait for the model horizon.

The previous live loop had GREEN_HARVEST: close any open position as soon as it
is net green. That is useful for studying touch rate, but it can destroy the
edge of short-horizon models trained to predict the move at 5/8/10 minutes.

This report tests the new rule:
* open by the candidate engine score;
* respect top_per_scan, max_open, one position per symbol, cooldown;
* do not close merely because the position is green;
* close at the first scan at/after the model horizon deadline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fast import config as FC
from .run_test_engine_harvest_sim import (
    FAVORITES,
    GRID,
    OUT,
    PriceBook,
    daily_summary,
    side_summary,
    simulate_engine,
    summarize_trades,
)
from .run_test_engines_compare import build_engines

PATIENT_MODES = [
    {"harvest": False, "top_per_scan": 1, "max_open": 10, "cooldown_min": 30},
    {"harvest": False, "top_per_scan": 3, "max_open": 10, "cooldown_min": 30},
    {"harvest": False, "top_per_scan": 5, "max_open": 10, "cooldown_min": 30},
    {"harvest": False, "top_per_scan": 3, "max_open": 10, "cooldown_min": 90},
    {"harvest": False, "top_per_scan": 5, "max_open": 10, "cooldown_min": 90},
]

CORE_ENGINES = [
    "PulseClean3_idx0.05_exit10m",
    "PulseClean3_idx0.00_exit10m",
    "DownCrash4_clean_std082_exit10m",
    "UpImpulse3_clean_std090_exit10m",
    "Portfolio_A_down4_up3_exit10m",
    "Portfolio_C_A_plus_PulseClean2_idx030_exit10m",
    "Portfolio_B_A_plus_PulseClean2_idx020_exit10m",
    "Portfolio_D_A_plus_ResearchTop20_std090_exit10m",
    "ResearchTop20Day_combined_std090_exit10m",
    "LiveTop3Scan_combined_std090_exit10m",
]


def _fmt_table(df: pd.DataFrame, cols: list[str]) -> str:
    fmt = {
        "trades_per_24h": "{:.1f}".format,
        "win": "{:.3f}".format,
        "avg_pnl": "{:+.4f}".format,
        "median_pnl": "{:+.4f}".format,
        "p10_pnl": "{:+.3f}".format,
        "p90_pnl": "{:+.3f}".format,
        "total_pnl": "{:+.2f}".format,
        "avg_hold_min": "{:.1f}".format,
        "long_pct": "{:.3f}".format,
        "harvest_rate": "{:.3f}".format,
        "deadline_rate": "{:.3f}".format,
    }
    return df[cols].to_string(index=False, formatters={k: v for k, v in fmt.items() if k in cols})


def print_report(summary: pd.DataFrame, side: pd.DataFrame, daily: pd.DataFrame) -> None:
    print("=== PATIENT ENGINE: BEST n>=20 days>=3 ===")
    show = summary[(summary["n"] >= 20) & (summary["days"] >= 3)].copy()
    show = show.sort_values(["avg_pnl", "win", "n"], ascending=[False, False, False]).head(40)
    print(_fmt_table(show, [
        "engine", "mode", "n", "trades_per_24h", "win", "avg_pnl",
        "median_pnl", "p10_pnl", "p90_pnl", "green_days", "days",
        "symbols", "total_pnl", "max_open_used", "block_max_open", "block_cooldown",
    ]))

    print("\n=== CORE / USER FAVORITES ===")
    fav = summary[summary["engine"].isin(CORE_ENGINES)].copy()
    fav = fav.sort_values(["engine", "mode"])
    print(_fmt_table(fav, [
        "engine", "mode", "n", "trades_per_24h", "win", "avg_pnl",
        "median_pnl", "green_days", "days", "symbols", "total_pnl",
        "max_open_used", "block_max_open", "block_cooldown",
    ]))

    print("\n=== SIDE BREAKDOWN / FIXED_TOP3_CAP10_CD30 ===")
    s = side[
        side["engine"].isin(CORE_ENGINES) &
        (side["mode"] == "fixed_top3_cap10_cd30")
    ].sort_values(["avg_pnl", "n"], ascending=[False, False])
    print(_fmt_table(s, [
        "engine", "side", "n", "win", "avg_pnl", "total_pnl", "avg_hold_min",
    ]))

    print("\n=== DAILY CORE / FIXED_TOP3_CAP10_CD30 ===")
    d = daily[
        daily["engine"].isin(CORE_ENGINES[:6]) &
        (daily["mode"] == "fixed_top3_cap10_cd30")
    ].sort_values(["engine", "day"])
    print(_fmt_table(d, [
        "engine", "day", "n", "win", "avg_pnl", "total_pnl", "symbols",
    ]))


def write_markdown(summary: pd.DataFrame, side: pd.DataFrame, daily: pd.DataFrame, out_dir: Path) -> None:
    best = summary[(summary["n"] >= 20) & (summary["days"] >= 3)].copy()
    best = best.sort_values(["avg_pnl", "win", "n"], ascending=[False, False, False]).head(50)
    core = summary[summary["engine"].isin(CORE_ENGINES)].sort_values(["engine", "mode"])
    s = side[
        side["engine"].isin(CORE_ENGINES) &
        (side["mode"] == "fixed_top3_cap10_cd30")
    ].sort_values(["avg_pnl", "n"], ascending=[False, False])
    d = daily[
        daily["engine"].isin(CORE_ENGINES[:6]) &
        (daily["mode"] == "fixed_top3_cap10_cd30")
    ].sort_values(["engine", "day"])

    lines = [
        "# Patient Engine Report",
        "",
        "Rule: no first-green harvest. Positions wait until the horizon deadline, with live-like top-per-scan, max-open and cooldown guards.",
        "",
        "## Best Patient Runs",
        "",
        "```text",
        _fmt_table(best, [
            "engine", "mode", "n", "trades_per_24h", "win", "avg_pnl",
            "median_pnl", "p10_pnl", "p90_pnl", "green_days", "days",
            "symbols", "total_pnl", "max_open_used", "block_max_open", "block_cooldown",
        ]),
        "```",
        "",
        "## Core / Favorites",
        "",
        "```text",
        _fmt_table(core, [
            "engine", "mode", "n", "trades_per_24h", "win", "avg_pnl",
            "median_pnl", "green_days", "days", "symbols", "total_pnl",
            "max_open_used", "block_max_open", "block_cooldown",
        ]),
        "```",
        "",
        "## Side Breakdown: fixed_top3_cap10_cd30",
        "",
        "```text",
        _fmt_table(s, [
            "engine", "side", "n", "win", "avg_pnl", "total_pnl", "avg_hold_min",
        ]),
        "```",
        "",
        "## Daily Core: fixed_top3_cap10_cd30",
        "",
        "```text",
        _fmt_table(d, [
            "engine", "day", "n", "win", "avg_pnl", "total_pnl", "symbols",
        ]),
        "```",
    ]
    (out_dir / "patient_engine_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID)
    grid["anchor_time"] = pd.to_datetime(grid["anchor_time"], utc=True)
    scan_times = sorted(pd.Timestamp(t) for t in grid["anchor_time"].drop_duplicates())
    window_hours = max(1.0, (scan_times[-1] - scan_times[0]).total_seconds() / 3600.0)

    engines = build_engines(grid)
    book = PriceBook()
    rows = []
    all_trades = []
    block_rows = []

    for engine_name, cand in engines.items():
        for kwargs in PATIENT_MODES:
            trades, blocks = simulate_engine(
                engine_name,
                cand,
                scan_times,
                book,
                **kwargs,
            )
            if len(trades):
                all_trades.append(trades)
            rows.append(summarize_trades(trades, blocks, window_hours))
            block_rows.append(blocks)

    summary = pd.DataFrame(rows)
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    daily = daily_summary(trades_all)
    side = side_summary(trades_all)
    blocks = pd.DataFrame(block_rows)

    summary.to_csv(OUT / "patient_engine_summary.csv", index=False)
    blocks.to_csv(OUT / "patient_engine_blocks.csv", index=False)
    daily.to_csv(OUT / "patient_engine_daily.csv", index=False)
    side.to_csv(OUT / "patient_engine_side.csv", index=False)
    if not trades_all.empty:
        trades_all.to_parquet(OUT / "patient_engine_trades.parquet", index=False)
    write_markdown(summary, side, daily, OUT)
    print_report(summary, side, daily)
    print(f"\nreports -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
