"""Resumable bulk candle downloader into the local CandleStore.

Migrated from old src/fetch_candles.py (_fetch_range, fetch_one) and
build_pump_dataset.candles_payload_to_df. Like the old code, all resolutions
are fetched and merged into ONE deduplicated timeline per symbol.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .okx_client import OKXClient
from .candle_store import CandleStore, prepare_candles

# Depth per resolution (days back). Kept from old fetch_candles RESOLUTION_DAYS.
# NOTE: 1m only covers ~7 days on OKX, so historic anchors bottom out at 5m
# granularity (see curve-design discussion).
RESOLUTION_DAYS: tuple[tuple[str, int], ...] = (
    ("1m", 7),
    ("5m", 240),
    ("1H", 730),
    ("1D", 1460),
)

_RAW_COLUMNS = [
    "timestamp_ms", "open", "high", "low", "close",
    "volume", "volume_ccy", "volume_quote", "confirm",
]


def _payload_to_df(rows: list[list[Any]], start_ms: int, end_ms: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=_RAW_COLUMNS)
    if "confirm" in df.columns:
        df = df[df["confirm"].astype(str) == "1"]
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)]
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


class CandleFetcher:
    def __init__(self, client: OKXClient, store: CandleStore, sleep_seconds: float = 0.12):
        self.client = client
        self.store = store
        self.sleep_seconds = sleep_seconds

    @staticmethod
    def _okx_inst(symbol: str) -> str:
        """Store id 'BTC_USDT_SWAP' -> OKX instId 'BTC-USDT-SWAP'."""
        return symbol.replace("_", "-")

    def _fetch_range(self, symbol: str, bar: str,
                     start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        inst = self._okx_inst(symbol)
        rows: list[list[Any]] = []
        after = str(end_ms)
        last_oldest: str | None = None

        while True:
            data = self.client.history_candles(inst, bar, after=after, limit=300)
            if not data:
                break
            rows.extend(data)
            oldest = str(min(int(item[0]) for item in data))
            if int(oldest) <= start_ms or oldest == last_oldest:
                break
            last_oldest = after = oldest
            time.sleep(self.sleep_seconds)

        return _payload_to_df(rows, start_ms, end_ms)

    def fetch_symbol(self, symbol: str, resolutions=RESOLUTION_DAYS, update: bool = False) -> dict:
        """Download all resolutions, merge into one timeline, store. Resumable
        via `update` (delta from latest cached timestamp)."""
        now = pd.Timestamp.now(tz="UTC")
        existing = self.store.load(symbol)
        delta_start = None
        if update and existing is not None and not existing.empty:
            delta_start = existing.index.max() - pd.Timedelta(minutes=5)

        frames: list[pd.DataFrame] = []
        for bar, days_back in resolutions:
            start = delta_start if delta_start is not None else now - pd.Timedelta(days=days_back)
            df = self._fetch_range(symbol, bar, start, now)
            if not df.empty:
                frames.append(prepare_candles(df))

        if not frames:
            return {"symbol": symbol, "status": "empty"}

        new = pd.concat(frames)
        new = new[~new.index.duplicated(keep="last")].sort_index()
        confirmed_max = new.index.max()
        if existing is not None and not existing.empty:
            new = pd.concat([existing, new])
            new = new[~new.index.duplicated(keep="last")].sort_index()
            new = new[new.index <= confirmed_max]
        self.store.save(symbol, new)
        return {"symbol": symbol, "status": "ok", "candles": int(len(new))}

    def fetch_all(self, symbols: list[str], resolutions=RESOLUTION_DAYS, update: bool = False) -> None:
        for sym in symbols:
            self.fetch_symbol(sym, resolutions, update)

    def update_recent(self, symbol: str, lookback_min: int = 180) -> dict:
        """Light refresh for the live loop: pull only the last `lookback_min`
        minutes of 1m candles and merge into the cached timeline. Much faster
        than fetch_symbol (no deep history, single resolution)."""
        now = pd.Timestamp.now(tz="UTC")
        start = now - pd.Timedelta(minutes=lookback_min)
        df = self._fetch_range(symbol, "1m", start, now)
        if df.empty:
            return {"symbol": symbol, "status": "empty"}
        new = prepare_candles(df)
        confirmed_max = new.index.max()
        existing = self.store.load(symbol)
        if existing is not None and not existing.empty:
            new = pd.concat([existing, new])
            new = new[~new.index.duplicated(keep="last")].sort_index()
            new = new[new.index <= confirmed_max]
        self.store.save(symbol, new)
        return {"symbol": symbol, "status": "ok", "last": new.index.max()}
