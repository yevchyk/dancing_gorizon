"""Vectorised reconstruction of the old ml_predictor feature curve.

Old layout (per anchor):
  w_{i:03d}_price = close(asof anchor - LOG_BOUNDS[i]) / entry      (i = 0..339)
  w_{i:03d}_vol   = log1p(sum volume in (anchor-LOG_BOUNDS[i+1], anchor-LOG_BOUNDS[i]])
  btc_w_{i:02d}_price = close_BTC(asof anchor - BTC_BOUNDS[i]) / btc_entry  (i = 0..29)

The original implementation looped pandas .loc per window (fine for ~50 live
symbols, far too slow for ~20k holdout anchors), so this version does it with
numpy searchsorted + a volume cumsum. Output matches the old builder window-wise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..trading.timeutil import index_to_ns

_BOUNDS = json.loads((C.DATA_DIR / "_old_bounds.json").read_text())
LOG_BOUNDS = np.asarray(_BOUNDS["LOG_BOUNDS"], dtype="int64")   # 341 -> 340 windows
BTC_BOUNDS = np.asarray(_BOUNDS["BTC_BOUNDS"], dtype="int64")   # 31  -> 30 windows
_MIN_NS = 60_000_000_000


class OldFeatureBuilder:
    """Builds the legacy 710-col feature dict for anchors of a single symbol."""

    def __init__(self, btc_candles: pd.DataFrame):
        self._btc = self._arrays(btc_candles)

    @staticmethod
    def _arrays(candles: pd.DataFrame):
        ts = index_to_ns(candles.index)
        close = candles["close"].to_numpy(float)
        vol = candles["volume"].to_numpy(float)
        cumvol = np.concatenate([[0.0], np.cumsum(vol)])   # cumvol[k] = sum of first k
        return ts, close, cumvol

    @staticmethod
    def _asof_idx(ts: np.ndarray, t_ns: int) -> int:
        return int(np.searchsorted(ts, t_ns, side="right")) - 1

    def _price_curve(self, ts, close, anchor_ns: int, entry: float,
                     bounds: np.ndarray, n: int) -> np.ndarray:
        out = np.ones(n, dtype=float)
        for i in range(n):
            idx = self._asof_idx(ts, anchor_ns - int(bounds[i]) * _MIN_NS)
            if idx >= 0 and entry > 0:
                v = close[idx] / entry
                out[i] = v if math.isfinite(v) else 1.0
        return out

    def _vol_curve(self, ts, cumvol, anchor_ns: int, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=float)
        for i in range(n):
            end_idx = self._asof_idx(ts, anchor_ns - int(LOG_BOUNDS[i]) * _MIN_NS)
            start_idx = self._asof_idx(ts, anchor_ns - int(LOG_BOUNDS[i + 1]) * _MIN_NS)
            s = cumvol[end_idx + 1] - cumvol[start_idx + 1]
            out[i] = math.log1p(s) if s > 0 and math.isfinite(s) else 0.0
        return out

    def build_rows(self, candles: pd.DataFrame, anchors_ns: np.ndarray) -> list[dict]:
        ts, close, cumvol = self._arrays(candles)
        bts, bclose, _ = self._btc
        n_log = len(LOG_BOUNDS) - 1
        n_btc = len(BTC_BOUNDS) - 1
        rows: list[dict] = []
        for a_ns in anchors_ns:
            a_ns = int(a_ns)
            ei = self._asof_idx(ts, a_ns)
            entry = close[ei] if ei >= 0 else 0.0
            price = self._price_curve(ts, close, a_ns, entry, LOG_BOUNDS, n_log)
            vol = self._vol_curve(ts, cumvol, a_ns, n_log)
            bidx = self._asof_idx(bts, a_ns)
            bentry = bclose[bidx] if bidx >= 0 and bclose[bidx] > 0 else 1.0
            btc = self._price_curve(bts, bclose, a_ns, bentry, BTC_BOUNDS, n_btc)
            row = {"_entry": entry}
            for i in range(n_log):
                row[f"w_{i:03d}_price"] = price[i]
                row[f"w_{i:03d}_vol"] = vol[i]
            for i in range(n_btc):
                row[f"btc_w_{i:02d}_price"] = btc[i]
            rows.append(row)
        return rows
