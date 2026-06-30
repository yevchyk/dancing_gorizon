"""1m candle download/cache helpers for fast_v1."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..database import CandleStore, OKXClient
from . import config as FC

RAW_COLUMNS = [
    "timestamp_ms", "open", "high", "low", "close",
    "volume", "volume_ccy", "volume_quote", "confirm",
]


def okx_inst(symbol: str) -> str:
    return symbol.replace("_", "-")


def top_liquid_symbols(n: int) -> list[str]:
    """Top local symbols by recent quote volume, minus the production blacklist."""
    from .. import config as C
    store = CandleStore(C.CANDLES_DIR)
    blacklist = set(C.BLACKLIST_SYMBOLS)
    vols: list[tuple[str, float]] = []
    for sym in store.symbols():
        if sym in blacklist:
            continue
        candles = store.load(sym)
        if candles is None or candles.empty:
            continue
        tail = candles.iloc[-1440:]
        vols.append((sym, float((tail["close"] * tail["volume"]).sum())))
    vols.sort(key=lambda item: item[1], reverse=True)
    return [sym for sym, _ in vols[:n]]


def _payload_to_df(rows: list[list[Any]], start_ms: int, end_ms: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)]
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def candle_path(symbol: str) -> Path:
    return FC.FAST_CANDLES_DIR / f"{symbol}.parquet"


def load_1m(symbol: str) -> pd.DataFrame | None:
    path = candle_path(symbol)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None   # corrupt/truncated file -> treat as missing, never crash
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def save_1m(symbol: str, df: pd.DataFrame) -> None:
    FC.FAST_CANDLES_DIR.mkdir(parents=True, exist_ok=True)
    out = df.reset_index()
    if "index" in out.columns:
        out = out.rename(columns={"index": "timestamp"})
    out.to_parquet(candle_path(symbol), index=False)


def fetch_1m_range(client: OKXClient, symbol: str, start: pd.Timestamp,
                   end: pd.Timestamp, sleep_seconds: float = 0.07) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    after = str(end_ms)
    rows: list[list[Any]] = []
    last_oldest: str | None = None
    while True:
        data = None
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                data = client.history_candles(okx_inst(symbol), "1m", after=after, limit=300)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))
        if data is None:
            raise RuntimeError(f"{symbol} page after={after} failed after retries: {last_error}")
        if not data:
            break
        rows.extend(data)
        oldest = str(min(int(item[0]) for item in data))
        if int(oldest) <= start_ms or oldest == last_oldest:
            break
        last_oldest = after = oldest
        time.sleep(sleep_seconds)
    df = _payload_to_df(rows, start_ms, end_ms)
    if df.empty:
        return df
    return df.set_index("timestamp").sort_index()


def ensure_1m(symbol: str, start: pd.Timestamp, end: pd.Timestamp,
              client: OKXClient | None = None) -> dict:
    """Ensure the symbol has 1m candles covering [start, end]."""
    client = client or OKXClient()
    existing = load_1m(symbol)
    fetch_start = start
    need_fetch = True
    if existing is not None and not existing.empty:
        min_t, max_t = existing.index.min(), existing.index.max()
        has_history = min_t <= start + pd.Timedelta(minutes=2)
        has_recent = max_t >= end - pd.Timedelta(minutes=2)
        need_fetch = not (has_history and has_recent)
        if has_history and not has_recent:
            fetch_start = max(start, max_t - pd.Timedelta(minutes=2))
    if need_fetch:
        fresh = fetch_1m_range(client, symbol, fetch_start, end)
        if existing is not None and not existing.empty:
            fresh = pd.concat([existing, fresh])
            fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
        if fresh.empty:
            return {"symbol": symbol, "status": "empty", "rows": 0}
        fresh = fresh[(fresh.index >= start) & (fresh.index <= end)]
        save_1m(symbol, fresh)
        existing = fresh
    return {
        "symbol": symbol,
        "status": "ok",
        "rows": int(len(existing)),
        "min": str(existing.index.min()),
        "max": str(existing.index.max()),
    }
