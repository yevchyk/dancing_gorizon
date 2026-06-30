"""Regression targets for the next-version models: instead of a binary 'did it
touch +X%', describe the whole path over each horizon with three real numbers.

For each horizon H, over future candles in (anchor, anchor + H]:
  ret_H = close(t+H) / entry - 1        where it ENDS  (fixed-time close return)
  mfe_H = max(high) / entry - 1         highest it goes UP on the way   (>= 0)
  mae_H = min(low)  / entry - 1         lowest it dips DOWN on the way   (<= 0)

ret drives expected profit, mae drives risk/stop, mfe drives a smarter take.
All are fractions (0.02 = +2%). Returns None if any horizon lacks forward data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HORIZONS


class ExcursionTargetBuilder:
    def __init__(self, horizons=HORIZONS):
        self.horizons = horizons

    def target_columns(self) -> list[str]:
        cols = []
        for h in self.horizons:
            cols += [f"ret_{h.label}", f"mfe_{h.label}", f"mae_{h.label}"]
        return cols

    def build(self, candles: pd.DataFrame, anchor_time: pd.Timestamp) -> dict[str, float] | None:
        past = candles.loc[candles.index <= anchor_time]
        if past.empty:
            return None
        entry = float(past["close"].iloc[-1])
        if entry <= 0 or not np.isfinite(entry):
            return None

        out: dict[str, float] = {}
        for h in self.horizons:
            end = anchor_time + pd.Timedelta(minutes=h.minutes)
            fut = candles.loc[(candles.index > anchor_time) & (candles.index <= end)]
            if fut.empty:
                return None
            out[f"ret_{h.label}"] = float(fut["close"].iloc[-1]) / entry - 1.0
            out[f"mfe_{h.label}"] = float(fut["high"].max()) / entry - 1.0
            out[f"mae_{h.label}"] = float(fut["low"].min()) / entry - 1.0
        return out
