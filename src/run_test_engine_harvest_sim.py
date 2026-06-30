"""Live-like simulation for the experimental fast_v2 test engines.

This bridges the research signal tables with the real live-loop mechanics:

* one scan every 2 minutes on the untouched 72h holdout grid;
* top-N candidates per scan, just like LiveTrader/trust engines;
* one open position per symbol, global max-open cap, per-symbol cooldown;
* optional green-harvest: close at the first scan where gross PnL > fee+slip;
* otherwise close at the first scan at/after the horizon deadline.

The older scripts already test pieces of this separately:
run_sim.py is event-driven with fixed exits, run_green_harvest.py tests the
harvest exit idea, and trading/live_trader.py is the production loop. This file
combines those rules for the new combined-signal engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .fast import config as FC
from .fast.candles import load_1m
from .run_test_engines_compare import EXIT_MIN, GRID, build_engines
from .trading.timeutil import index_to_ns

OUT = FC.FAST_ANALYSIS_DIR / "test_engines"
SCAN_STEP_MIN = FC.HOLDOUT_STEP_MIN
FEE_AND_SLIP = FC.EVAL_COST
HARVEST_GROSS_TRIGGER = C.HARVEST_COST_PCT / 100.0

FAVORITES = {
    "DownCrash4_clean_std082_exit10m",
    "UpImpulse3_clean_std090_exit10m",
    "ResearchTop20Day_combined_std090_exit10m",
    "LiveTop3Scan_combined_std090_exit10m",
    "PulseClean2_idx0.20_exit10m",
    "PulseClean2_idx0.30_exit10m",
    "PulseClean3_idx0.00_exit10m",
    "PulseClean3_idx0.05_exit10m",
    "Portfolio_A_down4_up3_exit10m",
    "Portfolio_B_A_plus_PulseClean2_idx020_exit10m",
    "Portfolio_C_A_plus_PulseClean2_idx030_exit10m",
    "Portfolio_D_A_plus_ResearchTop20_std090_exit10m",
}


@dataclass
class PriceSeries:
    ts_ns: np.ndarray
    close: np.ndarray

    def at(self, t: pd.Timestamp) -> float | None:
        t_ns = int(pd.Timestamp(t).value)
        idx = int(np.searchsorted(self.ts_ns, t_ns, side="right")) - 1
        if idx < 0:
            return None
        px = float(self.close[idx])
        return px if np.isfinite(px) and px > 0 else None


class PriceBook:
    def __init__(self) -> None:
        self._cache: dict[str, PriceSeries | None] = {}

    def at(self, symbol: str, t: pd.Timestamp) -> float | None:
        if symbol not in self._cache:
            candles = load_1m(symbol)
            if candles is None or candles.empty:
                self._cache[symbol] = None
            else:
                self._cache[symbol] = PriceSeries(
                    index_to_ns(candles.index),
                    candles["close"].to_numpy("float64"),
                )
        ps = self._cache[symbol]
        return None if ps is None else ps.at(t)


def _side_name(side: int) -> str:
    return "long" if int(side) > 0 else "short"


def _close_record(pos: dict, now: pd.Timestamp, px: float, reason: str) -> dict:
    side = int(pos["side"])
    gross = side * (px / pos["entry_price"] - 1.0)
    net = gross - FEE_AND_SLIP
    leverage = float(pos.get("leverage", 1.0))
    opened = pd.Timestamp(pos["opened_at"])
    return {
        "engine": pos["engine"],
        "mode": pos["mode"],
        "signal_model": pos.get("signal_model", ""),
        "source": pos.get("source", ""),
        "leverage": leverage,
        "symbol": pos["symbol"],
        "side": _side_name(side),
        "side_int": side,
        "exit": pos["exit"],
        "score": pos["score"],
        "opened_at": opened,
        "closed_at": now,
        "open_day": opened.strftime("%m-%d"),
        "close_day": now.strftime("%m-%d"),
        "entry_price": pos["entry_price"],
        "exit_price": px,
        "gross_pnl": gross,
        "net_pnl": net,
        "net_pnl_pct": net * 100.0,
        "levered_pnl_pct": net * 100.0 * leverage,
        "won": int(net > 0),
        "reason": reason,
        "held_min": (now - opened).total_seconds() / 60.0,
    }


def _close_due_positions(
    *,
    now: pd.Timestamp,
    open_pos: dict[str, dict],
    book: PriceBook,
    harvest: bool,
    trades: list[dict],
) -> None:
    for sym, pos in list(open_pos.items()):
        px = book.at(sym, now)
        if px is None:
            continue
        gross = int(pos["side"]) * (px / pos["entry_price"] - 1.0)
        if harvest and gross > HARVEST_GROSS_TRIGGER:
            trades.append(_close_record(pos, now, px, "harvest"))
            del open_pos[sym]
            continue
        if now >= pos["deadline"]:
            # Close at the exact horizon deadline, not at the (possibly later)
            # scan time. This keeps the exit consistent with the holdout target
            # convention and with the end-of-run flush below, so simulated PnL
            # no longer drifts with scan cadence on horizons that miss the grid
            # (e.g. 5m/8m on a 2m grid). Harvest exits stay scan-bound on purpose.
            px_dl = book.at(sym, pos["deadline"])
            if px_dl is None:
                px_dl = px
            trades.append(_close_record(pos, pos["deadline"], px_dl, "deadline"))
            del open_pos[sym]


def simulate_engine(
    engine_name: str,
    cand: pd.DataFrame,
    scan_times: list[pd.Timestamp],
    book: PriceBook,
    *,
    harvest: bool,
    top_per_scan: int,
    max_open: int,
    cooldown_min: int,
) -> tuple[pd.DataFrame, dict]:
    if cand.empty:
        return pd.DataFrame(), {
            "engine": engine_name, "mode": "", "block_max_open": 0,
            "block_already_open": 0, "block_cooldown": 0, "block_no_price": 0,
            "candidate_seen": 0, "max_open_used": 0,
        }

    mode = (
        f"{'green' if harvest else 'fixed'}"
        f"_top{top_per_scan}_cap{max_open}_cd{cooldown_min}"
    )
    d = cand.copy()
    d["anchor_time"] = pd.to_datetime(d["anchor_time"], utc=True)
    by_time = {
        pd.Timestamp(t): g.sort_values("score", ascending=False).head(top_per_scan)
        for t, g in d.groupby("anchor_time", sort=False)
    }

    open_pos: dict[str, dict] = {}
    last_trade_at: dict[str, pd.Timestamp] = {}
    trades: list[dict] = []
    blocks = {
        "engine": engine_name,
        "mode": mode,
        "block_max_open": 0,
        "block_already_open": 0,
        "block_cooldown": 0,
        "block_no_price": 0,
        "candidate_seen": 0,
        "max_open_used": 0,
    }

    for now in scan_times:
        _close_due_positions(
            now=now,
            open_pos=open_pos,
            book=book,
            harvest=harvest,
            trades=trades,
        )

        g = by_time.get(now)
        if g is None or g.empty:
            blocks["max_open_used"] = max(blocks["max_open_used"], len(open_pos))
            continue

        for row in g.itertuples(index=False):
            blocks["candidate_seen"] += 1
            sym = str(row.symbol)
            if sym in open_pos:
                blocks["block_already_open"] += 1
                continue
            if len(open_pos) >= max_open:
                blocks["block_max_open"] += 1
                continue
            last = last_trade_at.get(sym)
            if last is not None and now < last + pd.Timedelta(minutes=cooldown_min):
                blocks["block_cooldown"] += 1
                continue
            entry = book.at(sym, now)
            if entry is None:
                blocks["block_no_price"] += 1
                continue

            exit_label = str(row.exit)
            open_pos[sym] = {
                "engine": engine_name,
                "mode": mode,
                "signal_model": getattr(row, "signal_model", ""),
                "source": getattr(row, "source", ""),
                "leverage": float(getattr(row, "leverage", 1.0)),
                "symbol": sym,
                "side": int(row.side),
                "exit": exit_label,
                "score": float(row.score),
                "entry_price": entry,
                "opened_at": now,
                "deadline": now + pd.Timedelta(minutes=EXIT_MIN[exit_label]),
            }
            last_trade_at[sym] = now
            blocks["max_open_used"] = max(blocks["max_open_used"], len(open_pos))

    # Flush positions still open after the last grid scan at their exact deadline.
    for sym, pos in list(open_pos.items()):
        px = book.at(sym, pos["deadline"])
        if px is None:
            blocks["block_no_price"] += 1
            continue
        trades.append(_close_record(pos, pos["deadline"], px, "deadline_flush"))
        del open_pos[sym]

    return pd.DataFrame(trades), blocks


def _empty_summary(engine: str, mode: str, blocks: dict, window_hours: float) -> dict:
    return {
        "engine": engine,
        "mode": mode,
        "n": 0,
        "trades_per_24h": 0.0,
        "win": np.nan,
        "avg_pnl": np.nan,
        "median_pnl": np.nan,
        "total_pnl": 0.0,
        "harvest_rate": np.nan,
        "deadline_rate": np.nan,
        "avg_hold_min": np.nan,
        "p10_pnl": np.nan,
        "p90_pnl": np.nan,
        "green_days": 0,
        "days": 0,
        "symbols": 0,
        "long_pct": np.nan,
        "window_hours": window_hours,
        **blocks,
    }


def summarize_trades(trades: pd.DataFrame, blocks: dict, window_hours: float) -> dict:
    engine = blocks["engine"]
    mode = blocks["mode"]
    if trades.empty:
        return _empty_summary(engine, mode, blocks, window_hours)
    daily = trades.groupby("close_day")["net_pnl_pct"].sum()
    reasons = trades["reason"].astype(str)
    out = {
        "engine": engine,
        "mode": mode,
        "n": int(len(trades)),
        "trades_per_24h": float(len(trades) / window_hours * 24.0),
        "win": float(trades["won"].mean()),
        "avg_pnl": float(trades["net_pnl_pct"].mean()),
        "median_pnl": float(trades["net_pnl_pct"].median()),
        "total_pnl": float(trades["net_pnl_pct"].sum()),
        "harvest_rate": float((reasons == "harvest").mean()),
        "deadline_rate": float(reasons.str.startswith("deadline").mean()),
        "avg_hold_min": float(trades["held_min"].mean()),
        "p10_pnl": float(trades["net_pnl_pct"].quantile(0.10)),
        "p90_pnl": float(trades["net_pnl_pct"].quantile(0.90)),
        "green_days": int((daily > 0).sum()),
        "days": int(len(daily)),
        "symbols": int(trades["symbol"].nunique()),
        "long_pct": float((trades["side"] == "long").mean()),
        "window_hours": window_hours,
    }
    out.update(blocks)
    return out


def daily_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (engine, mode, day), g in trades.groupby(["engine", "mode", "close_day"]):
        rows.append({
            "engine": engine,
            "mode": mode,
            "day": day,
            "n": int(len(g)),
            "win": float(g["won"].mean()),
            "avg_pnl": float(g["net_pnl_pct"].mean()),
            "total_pnl": float(g["net_pnl_pct"].sum()),
            "harvest_rate": float((g["reason"] == "harvest").mean()),
            "symbols": int(g["symbol"].nunique()),
        })
    return pd.DataFrame(rows)


def side_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (engine, mode, side), g in trades.groupby(["engine", "mode", "side"]):
        rows.append({
            "engine": engine,
            "mode": mode,
            "side": side,
            "n": int(len(g)),
            "win": float(g["won"].mean()),
            "avg_pnl": float(g["net_pnl_pct"].mean()),
            "total_pnl": float(g["net_pnl_pct"].sum()),
            "harvest_rate": float((g["reason"] == "harvest").mean()),
            "avg_hold_min": float(g["held_min"].mean()),
        })
    return pd.DataFrame(rows)


def print_tables(summary: pd.DataFrame, side: pd.DataFrame, daily: pd.DataFrame) -> None:
    fmt = {
        "trades_per_24h": "{:.1f}".format,
        "win": "{:.3f}".format,
        "avg_pnl": "{:+.4f}".format,
        "median_pnl": "{:+.4f}".format,
        "total_pnl": "{:+.2f}".format,
        "harvest_rate": "{:.3f}".format,
        "deadline_rate": "{:.3f}".format,
        "avg_hold_min": "{:.1f}".format,
        "p10_pnl": "{:+.3f}".format,
        "p90_pnl": "{:+.3f}".format,
        "long_pct": "{:.3f}".format,
    }

    print("=== LIVE-LIKE GREEN HARVEST: BEST n>=20 days>=3 ===")
    show = summary[
        (summary["mode"].str.startswith("green_top3")) &
        (summary["n"] >= 20) &
        (summary["days"] >= 3)
    ].sort_values(["avg_pnl", "n"], ascending=[False, False]).head(35)
    print(show[[
        "engine", "mode", "n", "trades_per_24h", "win", "avg_pnl",
        "median_pnl", "harvest_rate", "avg_hold_min", "green_days",
        "days", "symbols", "total_pnl", "max_open_used",
        "block_max_open", "block_cooldown",
    ]].to_string(index=False, formatters=fmt))

    print("\n=== FAVORITES: GREEN vs FIXED / TOP3+TOP5 ===")
    fav = summary[summary["engine"].isin(FAVORITES)].copy()
    fav = fav.sort_values(["engine", "mode"])
    print(fav[[
        "engine", "mode", "n", "trades_per_24h", "win", "avg_pnl",
        "harvest_rate", "deadline_rate", "avg_hold_min", "green_days",
        "days", "symbols", "total_pnl", "max_open_used",
    ]].to_string(index=False, formatters=fmt))

    print("\n=== SIDE BREAKDOWN FOR FAVORITES / GREEN_TOP3_CD30 ===")
    s = side[
        side["engine"].isin(FAVORITES) &
        (side["mode"] == "green_top3_cap10_cd30")
    ].sort_values(["avg_pnl", "n"], ascending=[False, False])
    if s.empty:
        print("(empty)")
    else:
        print(s[[
            "engine", "side", "n", "win", "avg_pnl", "total_pnl",
            "harvest_rate", "avg_hold_min",
        ]].to_string(index=False, formatters=fmt))

    print("\n=== DAILY FOR CORE CANDIDATES / GREEN_TOP3_CD30 ===")
    core = [
        "PulseClean3_idx0.05_exit10m",
        "DownCrash4_clean_std082_exit10m",
        "Portfolio_A_down4_up3_exit10m",
        "Portfolio_C_A_plus_PulseClean2_idx030_exit10m",
    ]
    d = daily[
        daily["engine"].isin(core) &
        (daily["mode"] == "green_top3_cap10_cd30")
    ].sort_values(["engine", "day"])
    if d.empty:
        print("(empty)")
    else:
        print(d[[
            "engine", "day", "n", "win", "avg_pnl", "total_pnl",
            "harvest_rate", "symbols",
        ]].to_string(index=False, formatters=fmt))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID)
    grid["anchor_time"] = pd.to_datetime(grid["anchor_time"], utc=True)
    scan_times = sorted(pd.Timestamp(t) for t in grid["anchor_time"].drop_duplicates())
    window_hours = max(
        SCAN_STEP_MIN / 60.0,
        (scan_times[-1] - scan_times[0]).total_seconds() / 3600.0,
    )

    engines = build_engines(grid)
    book = PriceBook()
    all_trades: list[pd.DataFrame] = []
    rows: list[dict] = []
    block_rows: list[dict] = []

    modes = [
        {"harvest": True, "top_per_scan": 3, "max_open": 10, "cooldown_min": 30},
        {"harvest": True, "top_per_scan": 3, "max_open": 10, "cooldown_min": 90},
        {"harvest": True, "top_per_scan": 5, "max_open": 10, "cooldown_min": 30},
        {"harvest": False, "top_per_scan": 3, "max_open": 10, "cooldown_min": 30},
    ]

    for engine_name, cand in engines.items():
        for kwargs in modes:
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
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    daily = daily_summary(trades)
    side = side_summary(trades)
    blocks = pd.DataFrame(block_rows)

    summary.to_csv(OUT / "live_green_engine_summary.csv", index=False)
    blocks.to_csv(OUT / "live_green_engine_blocks.csv", index=False)
    daily.to_csv(OUT / "live_green_engine_daily.csv", index=False)
    side.to_csv(OUT / "live_green_engine_side.csv", index=False)
    if not trades.empty:
        trades.to_parquet(OUT / "live_green_engine_trades.parquet", index=False)

    print_tables(summary, side, daily)
    print(f"\nreports -> {Path(OUT).resolve()}")


if __name__ == "__main__":
    main()
