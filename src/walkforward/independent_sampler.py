"""Independent anchor sampler: at most one anchor per coin per UTC day inside an
absolute [start, end] window.

The dense 300-anchors-over-4-days sampling made trades highly correlated (a coin
trending for two days produced hundreds of near-identical 'wins'), which inflated
small-n win rates. One anchor per coin per day makes each trade an approximately
independent observation, so the statistics mean what they say.

Conforms to the DatasetCollector sampler interface: .sample(symbol, candles, now).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class IndependentAnchorSampler:
    def __init__(self, start: pd.Timestamp, end: pd.Timestamp, per_day: int = 1,
                 min_history_days: int = 7, lookahead_minutes: int = 120, seed: int = 42):
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.per_day = per_day
        self.min_history_days = min_history_days
        self.lookahead_minutes = lookahead_minutes
        self.seed = seed

    def sample(self, symbol: str, candles: pd.DataFrame,
               now: pd.Timestamp | None = None) -> list[pd.Timestamp]:
        if candles.empty:
            return []
        earliest_ok = candles.index.min() + pd.Timedelta(days=self.min_history_days)
        latest_ok = candles.index.max() - pd.Timedelta(minutes=self.lookahead_minutes)
        lo = max(self.start, earliest_ok)
        hi = min(self.end, latest_ok)
        idx = candles.index[(candles.index >= lo) & (candles.index <= hi)]
        if len(idx) == 0:
            return []

        rng = np.random.default_rng(abs(hash((symbol, self.seed))) % (2 ** 32))
        picks: list[pd.Timestamp] = []
        # group candidate timestamps by UTC day, pick up to per_day from each
        days = pd.Series(idx, index=idx).groupby(idx.normalize())
        for _, day_idx in days:
            arr = day_idx.to_numpy()
            k = min(self.per_day, len(arr))
            chosen = rng.choice(len(arr), size=k, replace=False)
            picks.extend(pd.Timestamp(arr[i]) for i in chosen)
        return sorted(picks)
