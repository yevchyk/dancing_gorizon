"""V2 feature schema for the band-specialist models.

Differences vs the legacy 302-column schema (hc.config):
  - DROP BTC-reference columns (c1h_btc_*, c4h_btc_*)  -> 60 fewer
  - ADD time-of-day features: hour_sin, hour_cos (cyclical, Kyiv), weekday 1-7
  - band-specific horizon grids: A 5-30/1min, B 30-90/5min, C 90-360/20min

Old schema and the live portfolio stay on hc.config untouched; this module is a
SEPARATE schema used only by data_v2 / the band models. Each trained model also
writes its column list to feature_names.json so scoring uses the exact schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as HC

LOCAL_TZ = "Europe/Kiev"

# scalar (non-curve) tail of the feature vector
TAIL_COLUMNS_V2 = ["horizon_minutes", "horizon_log", "hour_sin", "hour_cos", "weekday"]


def feature_names_v2(n_points: int = HC.N_POINTS) -> list[str]:
    cols: list[str] = []
    for i in range(n_points):
        cols.extend((f"c5m_rel_{i}", f"c5m_vol_{i}"))
    for i in range(n_points):
        cols.extend((f"c15m_rel_{i}", f"c15m_vol_{i}"))
    for i in range(n_points):
        cols.extend((f"c1h_rel_{i}", f"c1h_vol_{i}"))   # no btc
    for i in range(n_points):
        cols.extend((f"c4h_rel_{i}", f"c4h_vol_{i}"))   # no btc
    cols.extend(TAIL_COLUMNS_V2)
    return cols


FEATURE_COLUMNS_V2 = feature_names_v2()
EXPECTED_FEATURE_COUNT_V2 = len(FEATURE_COLUMNS_V2)            # 245
CURVE_COLUMNS_V2 = FEATURE_COLUMNS_V2[: -len(TAIL_COLUMNS_V2)]  # 240 curve cols

# band -> (lo, hi, step) in minutes
BANDS = {
    "A": (5, 30, 1),
    "B": (30, 90, 5),
    "C": (90, 360, 20),
}


def band_horizons(band: str) -> list[int]:
    lo, hi, step = BANDS[band]
    hs = list(range(lo, hi + 1, step))
    if hs[-1] != hi:
        hs.append(hi)
    return hs


def union_horizons() -> list[int]:
    s: set[int] = set()
    for b in BANDS:
        s.update(band_horizons(b))
    return sorted(s)


def time_features(base_time_utc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """hour_sin, hour_cos (cyclical Kyiv hour), weekday 1=Mon..7=Sun (Kyiv)."""
    idx = pd.DatetimeIndex(pd.to_datetime(base_time_utc, utc=True)).tz_convert(LOCAL_TZ)
    hour = idx.hour.to_numpy().astype("float32")
    hsin = np.sin(2.0 * np.pi * hour / 24.0).astype("float32")
    hcos = np.cos(2.0 * np.pi * hour / 24.0).astype("float32")
    wd = (idx.weekday.to_numpy() + 1).astype("float32")
    return hsin, hcos, wd
