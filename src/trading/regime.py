"""Simple per-coin trend regime detector.

regime = sign of (entry / price[lookback ago] - 1):
  > +band  -> "up"    (only longs allowed)
  < -band  -> "down"  (only shorts allowed)
  else     -> "neutral" (no directional trade)

This is the single highest-value strategy gate: the benchmarks showed model PnL
is dominated by which way the market drifted, so trading *with* the local trend
removes most of that regime risk.
"""

from __future__ import annotations

import numpy as np

from .. import config as C

_MIN_NS = 60_000_000_000


class RegimeDetector:
    def __init__(self, lookback_min: int = C.REGIME_LOOKBACK_MIN,
                 band: float = C.REGIME_BAND):
        self.lookback_min = lookback_min
        self.band = band

    def detect(self, ts: np.ndarray, close: np.ndarray,
               anchor_ns: int, entry: float) -> str:
        ref_ns = anchor_ns - self.lookback_min * _MIN_NS
        idx = int(np.searchsorted(ts, ref_ns, side="right")) - 1
        if idx < 0 or entry <= 0:
            return "neutral"
        ref = close[idx]
        if ref <= 0:
            return "neutral"
        trend = entry / ref - 1.0
        if trend > self.band:
            return "up"
        if trend < -self.band:
            return "down"
        return "neutral"
