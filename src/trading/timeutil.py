"""Time-unit helpers. Candle indexes are datetime64[us] but our offset math uses
nanoseconds; mixing the two silently scaled horizons by 1000x. Always normalise
to int64 nanoseconds through these before doing offset arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NS_PER_MIN = 60_000_000_000


def index_to_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """DatetimeIndex (any unit/tz) -> int64 nanoseconds since epoch."""
    return index.as_unit("ns").asi8


def anchors_to_ns(series) -> np.ndarray:
    """Anchor column (any unit/tz) -> int64 nanoseconds array."""
    return pd.to_datetime(series, utc=True).dt.as_unit("ns").astype("int64").to_numpy()


def ts_to_ns(ts) -> int:
    """A single Timestamp -> int nanoseconds (Timestamp.value is always ns)."""
    return int(pd.Timestamp(ts).value)
