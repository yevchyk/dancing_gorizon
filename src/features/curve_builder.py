"""Builds the log-spaced feature curve looking BACK from an anchor.

300 geometric sample points, min step 5min -> depth ~2 months. One metric per
point (300 columns):
  price_ratio = close(t) / close(anchor)   (relative price, ~1.0 near anchor)

No velocity / volume / date columns: trees infer trend from the many price lags.
Price semantics migrated from old market_features.build_log_features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _finite(x: float, default: float) -> float:
    return float(x) if x is not None and np.isfinite(x) else default


class CurveBuilder:
    def __init__(self, points: int, min_step_min: float, max_depth_min: float):
        self.points = points
        self.min_step_min = min_step_min
        self.max_depth_min = max_depth_min
        self._offsets = self._build_offsets()

    def _build_offsets(self) -> np.ndarray:
        """Geometric offsets (minutes back from anchor): offset_0 = min_step,
        offset_{N-1} = max_depth, spaced so spacing grows non-linearly."""
        ratio = self.max_depth_min / self.min_step_min
        exponents = np.arange(self.points) / (self.points - 1)
        return self.min_step_min * (ratio ** exponents)

    def boundaries(self) -> list[float]:
        return self._offsets.tolist()

    def columns(self) -> list[str]:
        return [f"p_{i:03d}" for i in range(self.points)]

    def build(self, candles: pd.DataFrame, anchor_time: pd.Timestamp) -> dict[str, float] | None:
        """Curve columns for one anchor, or None if there isn't enough history."""
        past = candles.loc[candles.index <= anchor_time]
        if len(past) < 10:
            return None
        entry = float(past["close"].iloc[-1])
        if entry <= 0 or not np.isfinite(entry):
            return None

        closes = candles["close"]
        offset_secs = np.round(self._offsets * 60).astype("int64")
        times = pd.DatetimeIndex(anchor_time - pd.to_timedelta(offset_secs, unit="s"))
        times = times.as_unit(closes.index.unit)
        sampled = closes.asof(times).to_numpy(dtype=float)   # nearest close at/before each time

        price_ratio = sampled / entry

        return {f"p_{i:03d}": _finite(price_ratio[i], 1.0) for i in range(self.points)}
