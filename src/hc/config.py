"""Constants for the horizon-conditioned model track."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C

STORE_KEY = "crypto_feature"
BTC_SYMBOL = "BTC_USDT_SWAP"

HC_DIR = C.DATA_DIR / "hc"
DATASET_DIR = HC_DIR / "dataset"
SMOKE_DATASET_DIR = HC_DIR / "smoke" / "dataset"
EXEC_HC_DIR = C.DATA_DIR / "hc_exec"
EXEC_DATASET_DIR = EXEC_HC_DIR / "dataset"
MODELS_DIR = C.MODELS_DIR / "hc"
SMOKE_MODELS_DIR = MODELS_DIR / "smoke"
EXEC_MODELS_DIR = C.MODELS_DIR / "hc_exec"
ANALYSIS_DIR = C.OUTPUTS_DIR / "analysis" / "hc"
SMOKE_ANALYSIS_DIR = ANALYSIS_DIR / "smoke"
EXEC_ANALYSIS_DIR = C.OUTPUTS_DIR / "analysis" / "hc_exec"
UNIVERSE_PATH = C.CONFIGS_DIR / "hc_universe.json"
RESULTS_MD = C.ROOT / "docs" / "HC_MODEL_RESULTS.md"
RESULTS_CSV = C.ROOT / "docs" / "HC_MODEL_RESULTS.csv"
SMOKE_RESULTS_MD = C.ROOT / "docs" / "HC_MODEL_RESULTS_SMOKE.md"
SMOKE_RESULTS_CSV = C.ROOT / "docs" / "HC_MODEL_RESULTS_SMOKE.csv"
TRUTH_PATH = C.ROOT / "TRADING_MODEL_TRUTH.md"

N_POINTS = 30
EXPECTED_FEATURE_COUNT = 302
SAMPLE_STRIDE_MIN = 120
EXEC_ENTRY_DELAY_MIN = 5
EMBARGO_MIN = 180
TEST_DAYS = 7
VALIDATION_FRACTION = 0.10
HC_ERA_START = pd.Timestamp("2025-09-27T00:00:00Z")

HORIZON_ANCHORS = (5, 15, 30, 60, 120, 180)
RANDOM_HORIZONS_PER_SNAPSHOT = 2
# The target is close_5m[t + h]. Non-5m random horizons cannot exist on a 5m
# target grid, so the default random draw is 5m-aligned while still filling the
# gaps between anchor horizons.
RANDOM_HORIZON_STEP_MIN = 5

THRESHOLD_GRID_PCT = {
    5: 0.4,
    15: 0.6,
    30: 0.8,
    60: 1.1,
    120: 1.5,
    180: 1.8,
}

ROUND_TRIP_FEE_PCT = 0.15
DECISION_PROB_HIGH = 0.70
DECISION_PROB_LOW = 0.30
MIN_SYMBOL_SIGNALS = 10

NS_PER_MIN = 60_000_000_000


@dataclass(frozen=True)
class TimeframeSpec:
    key: str
    freq: str
    features: tuple[str, ...]
    expected_5m_bars: int | None


TIMEFRAMES = (
    TimeframeSpec("5m", "5min", ("rel", "vol"), None),
    TimeframeSpec("15m", "15min", ("rel", "vol"), 3),
    TimeframeSpec("1h", "1h", ("rel", "btc", "vol"), 12),
    TimeframeSpec("4h", "4h", ("rel", "btc", "vol"), 48),
)

MODEL_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "task_type": "GPU",
    "devices": "0",
    "iterations": 4000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "od_type": "Iter",
    "od_wait": 200,
    "allow_writing_files": False,
    # GPU memory accommodation (RTX 4070, ~8GB free with desktop apps running):
    # CatBoost GPU OOMs at the default border_count=128 because split histograms
    # scale with borders (~10GB). border_count=32 cuts that to ~2.5GB; quality
    # impact on these noisy ratio features is negligible. gpu_ram_part guards
    # against fluctuating desktop GPU usage. Both are ignored on CPU task_type.
    "border_count": 32,
    "gpu_ram_part": 0.50,
}


def feature_names(n_points: int = N_POINTS) -> list[str]:
    cols: list[str] = []
    for i in range(n_points):
        cols.extend((f"c5m_rel_{i}", f"c5m_vol_{i}"))
    for i in range(n_points):
        cols.extend((f"c15m_rel_{i}", f"c15m_vol_{i}"))
    for i in range(n_points):
        cols.extend((f"c1h_rel_{i}", f"c1h_btc_{i}", f"c1h_vol_{i}"))
    for i in range(n_points):
        cols.extend((f"c4h_rel_{i}", f"c4h_btc_{i}", f"c4h_vol_{i}"))
    cols.extend(("horizon_minutes", "horizon_log"))
    return cols


FEATURE_COLUMNS = feature_names()
META_COLUMNS = ["symbol", "base_time"]
TARGET_COLUMNS = ["up_label", "down_label", "weight", "ret", "ret_pct", "thr_pct"]


def threshold_pct(horizon_min: int | float) -> float:
    xs = np.array(sorted(THRESHOLD_GRID_PCT), dtype="float64")
    ys = np.array([THRESHOLD_GRID_PCT[int(x)] for x in xs], dtype="float64")
    return float(np.interp(float(horizon_min), xs, ys))


def config_snapshot(extra: dict | None = None) -> dict:
    data = {
        "store_key": STORE_KEY,
        "btc_symbol": BTC_SYMBOL,
        "hc_era_start": HC_ERA_START.isoformat(),
        "n_points": N_POINTS,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
        "sample_stride_min": SAMPLE_STRIDE_MIN,
        "exec_entry_delay_min": EXEC_ENTRY_DELAY_MIN,
        "embargo_min": EMBARGO_MIN,
        "test_days": TEST_DAYS,
        "validation_fraction": VALIDATION_FRACTION,
        "horizon_anchors": list(HORIZON_ANCHORS),
        "random_horizons_per_snapshot": RANDOM_HORIZONS_PER_SNAPSHOT,
        "random_horizon_step_min": RANDOM_HORIZON_STEP_MIN,
        "threshold_grid_pct": THRESHOLD_GRID_PCT,
        "round_trip_fee_pct": ROUND_TRIP_FEE_PCT,
        "decision_prob_high": DECISION_PROB_HIGH,
        "decision_prob_low": DECISION_PROB_LOW,
        "timeframes": [asdict(tf) for tf in TIMEFRAMES],
        "model_params": MODEL_PARAMS,
    }
    if extra:
        data.update(extra)
    return data


def ensure_hc_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
