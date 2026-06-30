"""Builds all 15 target columns for one anchor from the future price path.

For each horizon, look at candles in (anchor, anchor + horizon]:
  up_{h}     = 1 if high reaches entry*(1+move_pct)   (touched +move%)
  down_{h}   = 1 if low  reaches entry*(1-move_pct)   (touched -move%)
  stable_{h} = 1 if (max_high - min_low)/entry < stability_range  (clean path)

`up` and `down` are touch-based (both can be 1 = whipsaw), which is exactly why
the separate stability model exists: it flags whippy / deceptive paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HORIZONS, DIRECTIONS, HorizonSpec


class TargetBuilder:
    def __init__(self, horizons=HORIZONS, directions=DIRECTIONS):
        self.horizons = horizons
        self.directions = directions

    def target_columns(self) -> list[str]:
        cols = []
        for h in self.horizons:
            for d in self.directions:
                cols.append(f"{d}_{h.label}")
            cols.append(f"stable_{h.label}")
        return cols

    def build(self, candles: pd.DataFrame, anchor_time: pd.Timestamp) -> dict[str, int] | None:
        past = candles.loc[candles.index <= anchor_time]
        if past.empty:
            return None
        entry = float(past["close"].iloc[-1])
        if entry <= 0 or not np.isfinite(entry):
            return None

        out: dict[str, int] = {}
        for h in self.horizons:
            end = anchor_time + pd.Timedelta(minutes=h.minutes)
            future = candles.loc[(candles.index > anchor_time) & (candles.index <= end)]
            if future.empty:
                return None  # not enough forward data -> skip this anchor entirely

            hi = float(future["high"].max())
            lo = float(future["low"].min())
            up_touch = (hi / entry - 1.0) >= h.move_pct
            down_touch = (1.0 - lo / entry) >= h.move_pct
            stable = ((hi - lo) / entry) < h.stability_range

            out[f"up_{h.label}"] = int(up_touch)
            out[f"down_{h.label}"] = int(down_touch)
            out[f"stable_{h.label}"] = int(stable)
        return out
