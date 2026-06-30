"""Dancing Horizon - central config.

One place for paths, the model registry, horizons, costs and the working thresholds.
The proven HC engine lives in the vendored ``src/`` package; ``dh/`` is the clean,
reusable layer on top. Paths auto-resolve to THIS project root, so the whole folder
is portable between machines (copy the directory, create a venv, go).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))            # make the vendored `src` package importable

from src.hc import config as HC               # noqa: E402  (vendored HC constants)

# --- data / models / configs (all inside this project) ---
CANDLES_DIR = ROOT / "data" / "candles"
REPORTS_DIR = ROOT / "reports"
UNIVERSE = ROOT / "configs" / "hc_universe.json"
THRESHOLDS = ROOT / "configs" / "thresholds" / "thr_300pd_fit_jun1-4.csv"

# Two models we keep (both decent). Code references dirs by these exact names.
MODELS = {
    "old": ROOT / "models" / "hc_exec_stride120_nonoverlap",   # trained to 2026-05-26
    "new": ROOT / "models" / "hc_exec_to20260604_prod",        # trained to 2026-06-04 20:00 UTC
}
MODEL_CUTOFF = {"old": "2026-05-26", "new": "2026-06-04 20:00 UTC"}

# --- signal / cost knobs ---
HORIZONS_DEFAULT = (30, 45, 60, 90)
ENTRY_DELAY_MIN = int(HC.EXEC_ENTRY_DELAY_MIN)     # features at base; enter at base+5m
FEE_PCT = float(HC.ROUND_TRIP_FEE_PCT)             # 0.15% round-trip
SLIP_ALL = 0.6                                     # slippage assumption: all coins
SLIP_LIQUID = 0.3                                  # slippage assumption: liquid only
EDGE_PROB = 0.90        # the validated edge lives in the high-conviction tail (>= ~0.90)
OPP_CAP = 0.20


def cost(slip: float = SLIP_ALL) -> float:
    return FEE_PCT + float(slip)
