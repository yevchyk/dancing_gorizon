"""Orchestrates the single full-curve dataset build: features + 15 targets.

Resumable: one parquet chunk per symbol under chunks_dir (skipped if present);
final output merges all chunks into one dataset parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..database import CandleStore
from ..features import CurveBuilder
from .anchor_sampler import AnchorSampler
from .target_builder import TargetBuilder


class DatasetCollector:
    def __init__(self, store: CandleStore, curve: CurveBuilder,
                 sampler: AnchorSampler, targets: TargetBuilder, chunks_dir: Path):
        self.store = store
        self.curve = curve
        self.sampler = sampler
        self.targets = targets
        self.chunks_dir = Path(chunks_dir)

    def _chunk_path(self, symbol: str) -> Path:
        return self.chunks_dir / f"{symbol}.parquet"

    def collect_symbol(self, symbol: str, now: pd.Timestamp | None = None) -> int:
        """Build rows for one symbol and write a chunk. Returns row count.
        Skips work if the chunk already exists (resume)."""
        chunk = self._chunk_path(symbol)
        if chunk.exists():
            return len(pd.read_parquet(chunk))

        candles = self.store.load(symbol)
        if candles is None or candles.empty:
            return 0

        rows: list[dict] = []
        for anchor in self.sampler.sample(symbol, candles, now=now):
            curve = self.curve.build(candles, anchor)
            if curve is None:
                continue
            tgt = self.targets.build(candles, anchor)
            if tgt is None:
                continue
            rows.append({"symbol": symbol, "anchor_time": anchor, **curve, **tgt})

        if not rows:
            return 0
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(chunk, index=False)
        return len(rows)

    def collect(self, symbols: list[str], out_path: Path,
                now: pd.Timestamp | None = None) -> Path:
        """Run all symbols (resumable), merge chunks into one dataset parquet."""
        total = 0
        for i, sym in enumerate(symbols, 1):
            n = self.collect_symbol(sym, now=now)
            total += n
            if i % 25 == 0 or i == len(symbols):
                print(f"  {i}/{len(symbols)} symbols  rows so far ~{total}", flush=True)

        frames = [pd.read_parquet(p) for p in sorted(self.chunks_dir.glob("*.parquet"))]
        dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(out_path, index=False)
        print(f"Dataset: {len(dataset)} rows, {dataset.shape[1]} cols -> {out_path}")
        return Path(out_path)
