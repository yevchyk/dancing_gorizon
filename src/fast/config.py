"""Configuration for the short-horizon research track.

The production config remains untouched so live trading can keep running while
we experiment with denser target data and short horizons.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .. import config as C


EXPERIMENT = os.environ.get("ML_FAST_EXPERIMENT", "fast_v2")
CANDLE_CACHE_EXPERIMENT = os.environ.get("ML_FAST_CANDLE_CACHE", "fast_v1")

ROOT = C.ROOT
FAST_DIR = C.DATA_DIR / EXPERIMENT
FAST_CANDLES_DIR = C.DATA_DIR / CANDLE_CACHE_EXPERIMENT / "candles_1m"
FAST_DATASETS_DIR = FAST_DIR / "datasets"
FAST_CHUNKS_DIR = FAST_DIR / "chunks"
FAST_MODELS_DIR = C.MODELS_DIR / EXPERIMENT / "base"
FAST_ANALYSIS_DIR = C.OUTPUTS_DIR / "analysis" / EXPERIMENT

TOP_SYMBOLS = 160
TRAIN_DAYS = 30
HOLDOUT_DAYS = 3
DOWNLOAD_CUSHION_DAYS = 1
HOLDOUT_STEP_MIN = 2
TRAIN_ANCHORS_PER_SYMBOL = 1800

CURVE_POINTS = 320
CURVE_MIN_STEP_MIN = 2.0
CURVE_MAX_DEPTH_MIN = 60 * 24 * 60
# Hybrid curve: keep dense near-anchor detail, then spend extra columns on
# broader 5m/1h production-cache context instead of asking 1m data to cover it.
CURVE_SEGMENTS = (
    (2.0, 12 * 60.0, 160),          # 2m -> 12h: dense short-term shape
    (12 * 60.0, 7 * 24 * 60.0, 80), # 12h -> 7d: regime / impulse memory
    (7 * 24 * 60.0, 60 * 24 * 60.0, 80), # 7d -> 60d: broad context
)

SLIPPAGE_PCT = 0.05
ROUNDTRIP_FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
EVAL_COST = ROUNDTRIP_FEE + SLIPPAGE_PCT / 100.0
TARGET_EDGE = ROUNDTRIP_FEE

# --- Bitcoin market-context features (fast_bitcoin_plus) ---------------------
# When ML_FAST_BTC=1, every sample gets extra columns = BTC's own price curve
# relative to the anchor: btc_i = BTC_close(anchor - offset_i) / BTC_close(anchor).
# Same form as the per-symbol curve, but it tells each model where BTC (the
# market) is. Default OFF so fast_v2 is byte-for-byte unchanged.
BTC_CONTEXT = os.environ.get("ML_FAST_BTC", "0") == "1"
BTC_SYMBOL = os.environ.get("ML_FAST_BTC_SYMBOL", "BTC_USDT_SWAP")
BTC_OFFSETS_MIN = (2, 5, 10, 15, 30, 60, 120, 240, 480, 720,
                   1440, 2880, 5760, 10080, 20160, 43200)   # 2m .. 30d (16 values)


def btc_columns() -> list[str]:
    return [f"btc_{i:03d}" for i in range(len(BTC_OFFSETS_MIN))]


@dataclass(frozen=True)
class FastHorizon:
    minutes: int
    label: str
    lookback_min: float


HORIZONS = (
    FastHorizon(2, "2m", 24 * 60),
    FastHorizon(5, "5m", 7 * 24 * 60),
    FastHorizon(8, "8m", 30 * 24 * 60),
    FastHorizon(10, "10m", 60 * 24 * 60),
)

# latest_crisis: trained only on the freshest regime, horizons 2/6/8m (6m added).
if EXPERIMENT == "latest_crisis":
    HORIZONS = (
        FastHorizon(2, "2m", 24 * 60),
        FastHorizon(6, "6m", 14 * 24 * 60),
        FastHorizon(8, "8m", 30 * 24 * 60),
    )


def ensure_dirs() -> None:
    for path in (
        FAST_CANDLES_DIR,
        FAST_DATASETS_DIR,
        FAST_CHUNKS_DIR,
        FAST_MODELS_DIR,
        FAST_ANALYSIS_DIR,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
