"""One-off migration: old CSV candle files (one merged timeline per symbol)
-> new parquet store.

Usage:
  python -m src.database.migrate_csv <old_csv_dir> <new_parquet_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .candle_store import CandleStore, prepare_candles


def migrate(old_dir: Path, new_dir: Path) -> None:
    store = CandleStore(new_dir)
    csvs = sorted(Path(old_dir).glob("*.csv"))
    print(f"Migrating {len(csvs)} files: {old_dir} -> {new_dir}")
    ok = err = 0
    for i, csv in enumerate(csvs, 1):
        symbol = csv.stem  # already in safe form (e.g. ADA_USDT_SWAP)
        try:
            df = prepare_candles(pd.read_csv(csv))
            store.save(symbol, df)
            ok += 1
        except Exception as exc:
            err += 1
            print(f"  [ERR] {csv.name}: {exc}")
        if i % 25 == 0 or i == len(csvs):
            print(f"  {i}/{len(csvs)}  ok={ok} err={err}", flush=True)
    print(f"Done: {ok} migrated, {err} errors -> {new_dir}")


if __name__ == "__main__":
    old = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../ml_predictor/data/candles")
    new = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/candles")
    migrate(old, new)
