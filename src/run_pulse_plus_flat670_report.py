"""Test PulseClean3 plus high-win flat 2m signals.

User idea:
* PulseClean3_idx0.00_exit10m is the quality motor, size/leverage 8x.
* Add flat single-model signals that reach ~0.670 win-rate, size/leverage 3x,
  to keep turnover flowing.
* No green harvest: wait for each signal's horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fast import config as FC
from .run_test_engine_harvest_sim import GRID, OUT, PriceBook, simulate_engine
from .run_test_engines_compare import build_engines

PULSE = "PulseClean3_idx0.00_exit10m"
FLAT_RULES = [
    {
        "signal_model": "fast_v2_up_2m",
        "prob_col": "fast_v2_p_up_2m",
        "side": 1,
        "exit": "2m",
        "threshold": 0.93,
        "leverage": 3.0,
    },
    {
        "signal_model": "fast_v2_down_2m",
        "prob_col": "fast_v2_p_down_2m",
        "side": -1,
        "exit": "2m",
        "threshold": 0.94,
        "leverage": 3.0,
    },
]
MODES = [
    {"harvest": False, "top_per_scan": 3, "max_open": 10, "cooldown_min": 0},
    {"harvest": False, "top_per_scan": 10, "max_open": 10, "cooldown_min": 0},
    {"harvest": False, "top_per_scan": 3, "max_open": 10, "cooldown_min": 10},
    {"harvest": False, "top_per_scan": 10, "max_open": 10, "cooldown_min": 10},
    {"harvest": False, "top_per_scan": 3, "max_open": 10, "cooldown_min": 30},
    {"harvest": False, "top_per_scan": 5, "max_open": 10, "cooldown_min": 30},
    {"harvest": False, "top_per_scan": 10, "max_open": 10, "cooldown_min": 30},
    {"harvest": False, "top_per_scan": 10, "max_open": 10, "cooldown_min": 90},
]


def flat670_candidates(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in FLAT_RULES:
        p = grid[rule["prob_col"]].astype(float)
        d = grid[p >= rule["threshold"]].copy()
        if d.empty:
            continue
        headroom = ((p[p >= rule["threshold"]] - rule["threshold"]) / (1.0 - rule["threshold"])).clip(0, 1)
        d["engine"] = "Flat670_2m"
        d["family"] = "flat670"
        d["source"] = "flat670"
        d["signal_model"] = rule["signal_model"]
        d["side"] = rule["side"]
        d["exit"] = rule["exit"]
        d["threshold"] = rule["threshold"]
        d["leverage"] = rule["leverage"]
        d["score"] = 10.0 + headroom.to_numpy()
        rows.append(d[[
            "engine", "family", "source", "signal_model", "symbol",
            "anchor_time", "day", "side", "exit", "threshold", "leverage", "score",
        ]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prep_pulse(pulse: pd.DataFrame) -> pd.DataFrame:
    d = pulse.copy()
    d["source"] = "pulse"
    d["signal_model"] = PULSE
    d["leverage"] = 8.0
    d["threshold"] = np.nan
    d["score"] = 100.0 + d["score"].astype(float)
    return d[[
        "engine", "family", "source", "signal_model", "symbol", "anchor_time",
        "day", "side", "exit", "threshold", "leverage", "score",
    ]]


def union_candidates(name: str, parts: list[pd.DataFrame]) -> pd.DataFrame:
    d = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
    if d.empty:
        return d
    d["engine"] = name
    d = d.sort_values("score", ascending=False)
    d = d.drop_duplicates(["symbol", "anchor_time", "side"], keep="first")
    d = d.drop_duplicates(["symbol", "anchor_time"], keep="first")
    return d.sort_values(["anchor_time", "score"], ascending=[True, False])


def overlap_report(pulse: pd.DataFrame, flat: pd.DataFrame) -> pd.DataFrame:
    if pulse.empty or flat.empty:
        return pd.DataFrame()
    p = pulse[["symbol", "anchor_time", "side"]].drop_duplicates()
    f = flat[["symbol", "anchor_time", "side", "signal_model"]].drop_duplicates()
    same = p.merge(f, on=["symbol", "anchor_time", "side"], how="inner")
    opposite = p.merge(f, on=["symbol", "anchor_time"], how="inner", suffixes=("_pulse", "_flat"))
    opposite = opposite[opposite["side_pulse"] != opposite["side_flat"]]
    return pd.DataFrame([
        {"case": "pulse_candidates", "n": len(p)},
        {"case": "flat_candidates", "n": len(f)},
        {"case": "same_symbol_time_side_overlap", "n": len(same)},
        {"case": "same_symbol_time_opposite_conflict", "n": len(opposite)},
    ])


def summarize(trades: pd.DataFrame, blocks: dict, window_hours: float) -> dict:
    row = {
        "engine": blocks["engine"],
        "mode": blocks["mode"],
        "n": 0,
        "trades_per_24h": 0.0,
        "win": np.nan,
        "avg_pnl": np.nan,
        "median_pnl": np.nan,
        "p10_pnl": np.nan,
        "p90_pnl": np.nan,
        "total_pnl": 0.0,
        "avg_levered_pnl": np.nan,
        "total_levered_pnl": 0.0,
        "pulse_n": 0,
        "flat_n": 0,
        "pulse_total_lev": 0.0,
        "flat_total_lev": 0.0,
        "green_days": 0,
        "days": 0,
        "symbols": 0,
        "max_open_used": blocks.get("max_open_used", 0),
        "block_max_open": blocks.get("block_max_open", 0),
        "block_cooldown": blocks.get("block_cooldown", 0),
    }
    if trades.empty:
        return row
    daily = trades.groupby("close_day")["net_pnl_pct"].sum()
    row.update({
        "n": int(len(trades)),
        "trades_per_24h": float(len(trades) / window_hours * 24.0),
        "win": float(trades["won"].mean()),
        "avg_pnl": float(trades["net_pnl_pct"].mean()),
        "median_pnl": float(trades["net_pnl_pct"].median()),
        "p10_pnl": float(trades["net_pnl_pct"].quantile(0.10)),
        "p90_pnl": float(trades["net_pnl_pct"].quantile(0.90)),
        "total_pnl": float(trades["net_pnl_pct"].sum()),
        "avg_levered_pnl": float(trades["levered_pnl_pct"].mean()),
        "total_levered_pnl": float(trades["levered_pnl_pct"].sum()),
        "pulse_n": int((trades["source"] == "pulse").sum()),
        "flat_n": int((trades["source"] == "flat670").sum()),
        "pulse_total_lev": float(trades.loc[trades["source"] == "pulse", "levered_pnl_pct"].sum()),
        "flat_total_lev": float(trades.loc[trades["source"] == "flat670", "levered_pnl_pct"].sum()),
        "green_days": int((daily > 0).sum()),
        "days": int(len(daily)),
        "symbols": int(trades["symbol"].nunique()),
    })
    return row


def source_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (engine, mode, source, signal_model), g in trades.groupby(["engine", "mode", "source", "signal_model"]):
        rows.append({
            "engine": engine,
            "mode": mode,
            "source": source,
            "signal_model": signal_model,
            "n": int(len(g)),
            "win": float(g["won"].mean()),
            "avg_pnl": float(g["net_pnl_pct"].mean()),
            "total_pnl": float(g["net_pnl_pct"].sum()),
            "avg_levered_pnl": float(g["levered_pnl_pct"].mean()),
            "total_levered_pnl": float(g["levered_pnl_pct"].sum()),
            "symbols": int(g["symbol"].nunique()),
        })
    return pd.DataFrame(rows)


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    fmt = {
        "trades_per_24h": "{:.1f}".format,
        "win": "{:.3f}".format,
        "avg_pnl": "{:+.4f}".format,
        "median_pnl": "{:+.4f}".format,
        "p10_pnl": "{:+.3f}".format,
        "p90_pnl": "{:+.3f}".format,
        "total_pnl": "{:+.2f}".format,
        "avg_levered_pnl": "{:+.3f}".format,
        "total_levered_pnl": "{:+.2f}".format,
        "pulse_total_lev": "{:+.2f}".format,
        "flat_total_lev": "{:+.2f}".format,
    }
    return df[cols].to_string(index=False, formatters={k: v for k, v in fmt.items() if k in cols})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID)
    grid["anchor_time"] = pd.to_datetime(grid["anchor_time"], utc=True)
    grid["day"] = grid["anchor_time"].dt.strftime("%m-%d")
    scan_times = sorted(pd.Timestamp(t) for t in grid["anchor_time"].drop_duplicates())
    window_hours = max(1.0, (scan_times[-1] - scan_times[0]).total_seconds() / 3600.0)

    engines = build_engines(grid)
    pulse = prep_pulse(engines[PULSE])
    flat = flat670_candidates(grid)
    combo = union_candidates("Combo_PulseClean3_x8_plus_Flat670_x3", [pulse, flat])

    candidates = {
        "PulseClean3_idx0.00_x8": pulse.assign(engine="PulseClean3_idx0.00_x8"),
        "Flat670_2m_x3": flat.assign(engine="Flat670_2m_x3"),
        "Combo_PulseClean3_x8_plus_Flat670_x3": combo,
    }

    book = PriceBook()
    rows = []
    all_trades = []
    blocks = []
    for name, cand in candidates.items():
        for kwargs in MODES:
            trades, block = simulate_engine(name, cand, scan_times, book, **kwargs)
            rows.append(summarize(trades, block, window_hours))
            blocks.append(block)
            if len(trades):
                all_trades.append(trades)

    summary = pd.DataFrame(rows)
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    by_source = source_summary(trades_all)
    overlap = overlap_report(pulse, flat)

    summary.to_csv(OUT / "pulse_plus_flat670_summary.csv", index=False)
    by_source.to_csv(OUT / "pulse_plus_flat670_by_source.csv", index=False)
    overlap.to_csv(OUT / "pulse_plus_flat670_overlap.csv", index=False)
    if not trades_all.empty:
        trades_all.to_parquet(OUT / "pulse_plus_flat670_trades.parquet", index=False)

    print("=== OVERLAP ===")
    print(overlap.to_string(index=False))
    print("\n=== PULSE + FLAT670 SUMMARY ===")
    print(_fmt(summary.sort_values(["engine", "mode"]), [
        "engine", "mode", "n", "trades_per_24h", "win", "avg_pnl", "median_pnl",
        "p10_pnl", "p90_pnl", "total_pnl", "avg_levered_pnl", "total_levered_pnl",
        "pulse_n", "flat_n", "pulse_total_lev", "flat_total_lev",
        "green_days", "days", "symbols", "max_open_used", "block_max_open", "block_cooldown",
    ]))
    print("\n=== BY SOURCE ===")
    if by_source.empty:
        print("(empty)")
    else:
        print(_fmt(by_source.sort_values(["engine", "mode", "source", "signal_model"]), [
            "engine", "mode", "source", "signal_model", "n", "win", "avg_pnl",
            "total_pnl", "avg_levered_pnl", "total_levered_pnl", "symbols",
        ]))
    print(f"\nreports -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
