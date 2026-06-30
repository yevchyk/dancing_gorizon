"""Pick random anchor timestamps per symbol within the training window.

Window: [now - start_offset_days, now - end_offset_days]  (default -4mo .. -10d).
An anchor is valid only if it has enough lookback (for the curve) and enough
lookahead (for the longest target horizon).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AnchorSampler:
    def __init__(self, anchors_per_symbol: int, start_offset_days: int,
                 end_offset_days: int, min_history_days: int = 7,
                 lookahead_minutes: int = 120, seed: int = 42):
        self.anchors_per_symbol = anchors_per_symbol
        self.start_offset_days = start_offset_days
        self.end_offset_days = end_offset_days
        self.min_history_days = min_history_days
        self.lookahead_minutes = lookahead_minutes
        self.seed = seed

    def sample(self, symbol: str, candles: pd.DataFrame, now: pd.Timestamp | None = None) -> list[pd.Timestamp]:
        if candles.empty:
            return []
        now = now or pd.Timestamp.now(tz="UTC")
        win_start = now - pd.Timedelta(days=self.start_offset_days)
        win_end = now - pd.Timedelta(days=self.end_offset_days)

        # need history before the anchor and future after it
        earliest_ok = candles.index.min() + pd.Timedelta(days=self.min_history_days)
        latest_ok = candles.index.max() - pd.Timedelta(minutes=self.lookahead_minutes)

        lo = max(win_start, earliest_ok)
        hi = min(win_end, latest_ok)
        candidates = candles.index[(candles.index >= lo) & (candles.index <= hi)]
        if len(candidates) == 0:
            return []

        n = min(self.anchors_per_symbol, len(candidates))
        rng = np.random.default_rng(abs(hash((symbol, self.seed))) % (2**32))
        picks = rng.choice(len(candidates), size=n, replace=False)
        return sorted(candidates[i] for i in picks)
