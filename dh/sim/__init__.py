"""Sim helpers: executable outcomes (entry+5m -> exit at horizon) and windows.

Outcomes are model-independent (pure prices) -> compute once, reuse for any model.
Immature rows (exit price not yet available) are dropped automatically.
"""
from __future__ import annotations

import pandas as pd

from src.hc import config as HC
from src.run_hc_horizon_threshold_optimizer import attach_exact_outcomes as _attach


def outcomes(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, base_time, horizon_minutes): gross_move% of a LONG (short = -that).

    gross_move = (exit/entry - 1) * 100, entry = base_time + 5m, exit = entry + horizon.
    """
    tmp = rows[["symbol", "base_time", "horizon_minutes"]].drop_duplicates().copy()
    tmp["side"] = 1
    mv = _attach(tmp, max_stale_min=2.0)
    mv["gross_move"] = mv["net_pnl_pct"] + HC.ROUND_TRIP_FEE_PCT
    return mv[["symbol", "base_time", "horizon_minutes", "gross_move"]]


def date_grid(date: str, days: float, stride_min: int = 5) -> pd.DatetimeIndex:
    """5-minute entry grid covering [date 00:00, date+days), capped at 'now'."""
    start = pd.Timestamp(date, tz="UTC")
    end = start + pd.Timedelta(days=float(days))
    now = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
    end = min(end, now)
    return pd.date_range(start, end, freq=f"{int(stride_min)}min", tz="UTC", inclusive="left")
