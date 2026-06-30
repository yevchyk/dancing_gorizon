"""Live-like config matrix for the proposed patient engine.

Tests the current scored grid as if it were live:
* no harvest;
* hold until each signal horizon;
* max 10 open positions;
* configurable top-per-scan, per-symbol cooldown, and scan cadence;
* full untouched 72h window and rolling last 24h window.

The scored grid has anchors every 2 minutes, so honest scan cadences here are
2m, 4m, 6m, 10m. A true 1m test needs a separate 1m-scored holdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .run_test_engine_harvest_sim import GRID, OUT, PriceBook, simulate_engine
from .run_test_engines_compare import build_engines

SCAN_STEPS = (2, 4, 6, 10)
COOLDOWNS = (0, 2, 10, 30)
TOPS = (3, 10)
MAX_OPEN = 10


def _prep_pulse(pulse: pd.DataFrame, name: str, leverage: float = 8.0) -> pd.DataFrame:
    d = pulse.copy()
    d["engine"] = name
    d["source"] = "pulse"
    d["signal_model"] = name.replace("_x8", "")
    d["leverage"] = leverage
    d["threshold"] = np.nan
    d["score"] = 100.0 + d["score"].astype(float)
    return d[[
        "engine", "family", "source", "signal_model", "symbol", "anchor_time",
        "day", "side", "exit", "threshold", "leverage", "score",
    ]]


def _flat_candidates(grid: pd.DataFrame, *, up_thr: float, down_thr: float,
                     name: str, leverage: float = 3.0) -> pd.DataFrame:
    rules = [
        ("fast_v2_up_2m", "fast_v2_p_up_2m", 1, up_thr),
        ("fast_v2_down_2m", "fast_v2_p_down_2m", -1, down_thr),
    ]
    rows = []
    for signal_model, prob_col, side, thr in rules:
        p = grid[prob_col].astype(float)
        d = grid[p >= thr].copy()
        if d.empty:
            continue
        headroom = ((p[p >= thr] - thr) / (1.0 - thr)).clip(0, 1)
        d["engine"] = name
        d["family"] = "flat2m"
        d["source"] = "flat"
        d["signal_model"] = signal_model
        d["side"] = side
        d["exit"] = "2m"
        d["threshold"] = thr
        d["leverage"] = leverage
        d["score"] = 10.0 + headroom.to_numpy()
        rows.append(d[[
            "engine", "family", "source", "signal_model", "symbol",
            "anchor_time", "day", "side", "exit", "threshold", "leverage", "score",
        ]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _union(name: str, parts: list[pd.DataFrame]) -> pd.DataFrame:
    d = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
    if d.empty:
        return d
    d["engine"] = name
    d = d.sort_values("score", ascending=False)
    d = d.drop_duplicates(["symbol", "anchor_time", "side"], keep="first")
    d = d.drop_duplicates(["symbol", "anchor_time"], keep="first")
    return d.sort_values(["anchor_time", "score"], ascending=[True, False])


def _select_scan_times(times: list[pd.Timestamp], *, window: str, scan_step_min: int) -> list[pd.Timestamp]:
    end = times[-1]
    if window == "last24h":
        start = end - pd.Timedelta(hours=24)
    elif window == "full72h":
        start = times[0]
    else:
        raise ValueError(window)
    selected = [t for t in times if t >= start]
    if scan_step_min == 2:
        return selected
    stride = max(1, scan_step_min // 2)
    return selected[::stride]


def _summarize(trades: pd.DataFrame, blocks: dict, *, window: str, scan_step: int,
               top: int, cooldown: int, window_hours: float) -> dict:
    row = {
        "engine": blocks["engine"],
        "window": window,
        "scan_step_min": scan_step,
        "top_per_scan": top,
        "cooldown_min": cooldown,
        "mode": blocks["mode"],
        "n": 0,
        "trades_per_24h": 0.0,
        "win": np.nan,
        "avg_pnl": np.nan,
        "median_pnl": np.nan,
        "p10_pnl": np.nan,
        "p90_pnl": np.nan,
        "total_pnl": 0.0,
        "avg_lev_pnl": np.nan,
        "total_lev_pnl": 0.0,
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
        "avg_lev_pnl": float(trades["levered_pnl_pct"].mean()),
        "total_lev_pnl": float(trades["levered_pnl_pct"].sum()),
        "pulse_n": int((trades["source"] == "pulse").sum()),
        "flat_n": int((trades["source"] == "flat").sum()),
        "pulse_total_lev": float(trades.loc[trades["source"] == "pulse", "levered_pnl_pct"].sum()),
        "flat_total_lev": float(trades.loc[trades["source"] == "flat", "levered_pnl_pct"].sum()),
        "green_days": int((daily > 0).sum()),
        "days": int(len(daily)),
        "symbols": int(trades["symbol"].nunique()),
    })
    return row


def _source_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in trades.groupby([
        "engine", "window", "scan_step_min", "top_per_scan", "cooldown_min",
        "source", "signal_model",
    ]):
        engine, window, step, top, cd, source, signal_model = keys
        rows.append({
            "engine": engine,
            "window": window,
            "scan_step_min": step,
            "top_per_scan": top,
            "cooldown_min": cd,
            "source": source,
            "signal_model": signal_model,
            "n": int(len(g)),
            "win": float(g["won"].mean()),
            "avg_pnl": float(g["net_pnl_pct"].mean()),
            "total_pnl": float(g["net_pnl_pct"].sum()),
            "avg_lev_pnl": float(g["levered_pnl_pct"].mean()),
            "total_lev_pnl": float(g["levered_pnl_pct"].sum()),
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
        "avg_lev_pnl": "{:+.3f}".format,
        "total_lev_pnl": "{:+.2f}".format,
        "pulse_total_lev": "{:+.2f}".format,
        "flat_total_lev": "{:+.2f}".format,
    }
    return df[cols].to_string(index=False, formatters={k: v for k, v in fmt.items() if k in cols})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID)
    grid["anchor_time"] = pd.to_datetime(grid["anchor_time"], utc=True)
    grid["day"] = grid["anchor_time"].dt.strftime("%m-%d")
    all_times = sorted(pd.Timestamp(t) for t in grid["anchor_time"].drop_duplicates())

    raw_engines = build_engines(grid)
    pulse00 = _prep_pulse(raw_engines["PulseClean3_idx0.00_exit10m"], "Pulse00_x8")
    pulse05 = _prep_pulse(raw_engines["PulseClean3_idx0.05_exit10m"], "Pulse05_x8")
    flat670 = _flat_candidates(grid, up_thr=0.93, down_thr=0.94, name="Flat670_x3")
    flat_strict = _flat_candidates(grid, up_thr=0.94, down_thr=0.94, name="FlatStrict_x3")

    engines = {
        "Pulse00_x8": pulse00,
        "Pulse05_x8": pulse05,
        "Flat670_x3": flat670,
        "FlatStrict_x3": flat_strict,
        "Combo00_Flat670": _union("Combo00_Flat670", [pulse00, flat670]),
        "Combo00_FlatStrict": _union("Combo00_FlatStrict", [pulse00, flat_strict]),
        "Combo05_Flat670": _union("Combo05_Flat670", [pulse05, flat670]),
    }

    book = PriceBook()
    rows = []
    all_trades = []
    for window in ("full72h", "last24h"):
        for scan_step in SCAN_STEPS:
            scan_times = _select_scan_times(all_times, window=window, scan_step_min=scan_step)
            window_hours = max(1.0, (scan_times[-1] - scan_times[0]).total_seconds() / 3600.0)
            for top in TOPS:
                for cooldown in COOLDOWNS:
                    for name, cand in engines.items():
                        trades, blocks = simulate_engine(
                            name,
                            cand,
                            scan_times,
                            book,
                            harvest=False,
                            top_per_scan=top,
                            max_open=MAX_OPEN,
                            cooldown_min=cooldown,
                        )
                        if len(trades):
                            trades = trades.copy()
                            trades["window"] = window
                            trades["scan_step_min"] = scan_step
                            trades["top_per_scan"] = top
                            trades["cooldown_min"] = cooldown
                            all_trades.append(trades)
                        rows.append(_summarize(
                            trades,
                            blocks,
                            window=window,
                            scan_step=scan_step,
                            top=top,
                            cooldown=cooldown,
                            window_hours=window_hours,
                        ))

    summary = pd.DataFrame(rows)
    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    source = _source_summary(trades_all)

    summary.to_csv(OUT / "live_config_matrix_summary.csv", index=False)
    source.to_csv(OUT / "live_config_matrix_by_source.csv", index=False)
    if not trades_all.empty:
        trades_all.to_parquet(OUT / "live_config_matrix_trades.parquet", index=False)

    print("=== BEST FULL72 COMBO CONFIGS ===")
    full = summary[
        (summary["window"] == "full72h") &
        (summary["engine"].str.contains("Combo"))
    ].sort_values(["total_lev_pnl", "avg_lev_pnl"], ascending=[False, False]).head(25)
    print(_fmt(full, [
        "engine", "scan_step_min", "top_per_scan", "cooldown_min", "n",
        "trades_per_24h", "win", "avg_pnl", "total_pnl", "avg_lev_pnl",
        "total_lev_pnl", "pulse_n", "flat_n", "pulse_total_lev",
        "flat_total_lev", "green_days", "days", "max_open_used",
    ]))

    print("\n=== BEST LAST24 COMBO CONFIGS ===")
    last = summary[
        (summary["window"] == "last24h") &
        (summary["engine"].str.contains("Combo"))
    ].sort_values(["total_lev_pnl", "avg_lev_pnl"], ascending=[False, False]).head(25)
    print(_fmt(last, [
        "engine", "scan_step_min", "top_per_scan", "cooldown_min", "n",
        "trades_per_24h", "win", "avg_pnl", "total_pnl", "avg_lev_pnl",
        "total_lev_pnl", "pulse_n", "flat_n", "pulse_total_lev",
        "flat_total_lev", "green_days", "days", "max_open_used",
    ]))

    print("\n=== ENGINE COMPARISON @ 2M TOP3 CD2 ===")
    focus = summary[
        (summary["scan_step_min"] == 2) &
        (summary["top_per_scan"] == 3) &
        (summary["cooldown_min"] == 2)
    ].sort_values(["window", "total_lev_pnl"], ascending=[True, False])
    print(_fmt(focus, [
        "window", "engine", "n", "trades_per_24h", "win", "avg_pnl",
        "total_pnl", "avg_lev_pnl", "total_lev_pnl", "pulse_n", "flat_n",
        "green_days", "days", "max_open_used",
    ]))

    print(f"\nreports -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
