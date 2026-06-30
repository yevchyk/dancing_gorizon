"""Local candle cache. One file per symbol — a single merged OHLCV timeline
whose resolution degrades going back in time (1m last ~7d, 5m to ~8mo, 1H to
~2y, 1D beyond). New format: parquet (plan change from the old CSV layout).

Candle schema: timestamp (UTC, index), open, high, low, close, volume.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def safe_symbol_filename(symbol: str) -> str:
    return symbol.replace("\\", "_").replace("-", "_").replace(":", "_")


def prepare_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Clean + sort + index by timestamp. Migrated from market_features.prepare_candles."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Candle data is missing columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Candle data contains invalid timestamps.")
    for c in OHLCV_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", *OHLCV_COLUMNS])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.set_index("timestamp")


class CandleStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path_for(self, symbol: str) -> Path:
        return self.root / f"{safe_symbol_filename(symbol)}.parquet"

    def has(self, symbol: str) -> bool:
        return self.path_for(symbol).exists()

    def load(self, symbol: str) -> pd.DataFrame | None:
        path = self.path_for(symbol)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
        except Exception:
            # A truncated/corrupt parquet (e.g. a fetch interrupted mid-write)
            # must never crash the live loop — treat it as missing so it gets
            # re-fetched, instead of killing the whole engine on one bad file.
            return None
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        return df

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        path = self.path_for(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_parquet(path, index=False)

    def symbols(self) -> list[str]:
        return [p.stem for p in self.root.glob("*.parquet")]
