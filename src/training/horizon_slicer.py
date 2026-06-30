"""Selects the curve-column subset a given horizon's model should see.

Short horizons (5m) only look back ~1 day; long horizons (2h) use the full
curve. The full curve is collected once; this picks the slice by each
HorizonSpec.lookback_min, mapping column index -> geometric offset (minutes).
"""

from __future__ import annotations

from ..config import HorizonSpec
from ..features import CurveBuilder


class HorizonSlicer:
    def __init__(self, curve: CurveBuilder):
        self.curve = curve
        self._offsets = curve.boundaries()           # minutes back, per column index
        self._columns = curve.columns()              # p_000 .. p_299

    def columns_for(self, horizon: HorizonSpec) -> list[str]:
        cap = horizon.lookback_min
        return [self._columns[i] for i, off in enumerate(self._offsets) if off <= cap]
