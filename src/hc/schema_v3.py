"""V3 schema = V2 + a 1-minute timeframe block (for the fast band-A scalper).

c1m_rel/vol (30 pts) prepended to the v2 blocks -> microstructure resolution.
No BTC; keeps time-of-day features. 305 columns total.
Raw candles are already 1-minute on disk, so no extra fetch is needed.
"""

from __future__ import annotations

from . import config as HC
from .schema_v2 import TAIL_COLUMNS_V2  # horizon_minutes, horizon_log, hour_sin, hour_cos, weekday


def feature_names_v3(n_points: int = HC.N_POINTS) -> list[str]:
    cols: list[str] = []
    for i in range(n_points):
        cols.extend((f"c1m_rel_{i}", f"c1m_vol_{i}"))
    for i in range(n_points):
        cols.extend((f"c5m_rel_{i}", f"c5m_vol_{i}"))
    for i in range(n_points):
        cols.extend((f"c15m_rel_{i}", f"c15m_vol_{i}"))
    for i in range(n_points):
        cols.extend((f"c1h_rel_{i}", f"c1h_vol_{i}"))
    for i in range(n_points):
        cols.extend((f"c4h_rel_{i}", f"c4h_vol_{i}"))
    cols.extend(TAIL_COLUMNS_V2)
    return cols


FEATURE_COLUMNS_V3 = feature_names_v3()
EXPECTED_FEATURE_COUNT_V3 = len(FEATURE_COLUMNS_V3)            # 305
CURVE_COLUMNS_V3 = FEATURE_COLUMNS_V3[: -len(TAIL_COLUMNS_V2)]  # 300
