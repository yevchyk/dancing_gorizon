"""ТТ feature schema — MAXIMAL inputs for the curve model (user: «максимальну хочу»).

Union of every curve block we have, at a WIDER window (N_POINTS=45 = 30×1.5,
user: «навіть на половину більше»), PLUS the full BTC reference curve, PLUS the
18 v5 regime scalars. Horizon is the OUTPUT axis (the predicted curve), so
`horizon_minutes`/`horizon_log` are NOT features here — the tail is time-of-day
only. All symbol-blind (no identity features, §6).

Per curve point:
  c1m: rel, vol            (1-min microstructure — back in as input)
  c5m: rel, vol
  c15m: rel, vol
  c1h: rel, btc, vol       (full BTC reference curve)
  c4h: rel, btc, vol
=> 12 cols/point × 45 = 540, + 18 regime + 3 time = 561 features.

Target (the "graph"): vol-normalized cumulative log-return on a 1-min grid
1..240. Horizon h node = cumret(entry->entry+h) / sigma_1m(base). See data_tt.
"""

from __future__ import annotations

from ..hc.schema_v5 import REGIME_COLUMNS_V5

TT_N_POINTS = 45          # legacy 30 × 1.5 — wider input window
TIME_TAIL = ["hour_sin", "hour_cos", "weekday"]   # NO horizon_* (horizon is the output axis)


def curve_columns_tt(n_points: int = TT_N_POINTS) -> list[str]:
    cols: list[str] = []
    for i in range(n_points):
        cols += [f"c1m_rel_{i}", f"c1m_vol_{i}"]
    for i in range(n_points):
        cols += [f"c5m_rel_{i}", f"c5m_vol_{i}"]
    for i in range(n_points):
        cols += [f"c15m_rel_{i}", f"c15m_vol_{i}"]
    for i in range(n_points):
        cols += [f"c1h_rel_{i}", f"c1h_btc_{i}", f"c1h_vol_{i}"]
    for i in range(n_points):
        cols += [f"c4h_rel_{i}", f"c4h_btc_{i}", f"c4h_vol_{i}"]
    return cols


def feature_names_tt(n_points: int = TT_N_POINTS, include_regime: bool = True) -> list[str]:
    cols = curve_columns_tt(n_points)
    if include_regime:
        cols = cols + list(REGIME_COLUMNS_V5)
    return cols + TIME_TAIL


CURVE_COLUMNS_TT = curve_columns_tt()
FEATURE_COLUMNS_TT = feature_names_tt()
EXPECTED_FEATURE_COUNT_TT = len(FEATURE_COLUMNS_TT)        # 561


# ---- target (the curve / "graph") ----
TT_HORIZON_MAX = 240      # minutes — width of the predicted graph along the time axis
TT_HORIZON_STEP = 1       # minimal 1-min grid (continuous-h query = interpolate the curve)


def target_horizons_tt(h_max: int = TT_HORIZON_MAX, step: int = TT_HORIZON_STEP) -> list[int]:
    return list(range(step, h_max + 1, step))


def target_columns_tt(h_max: int = TT_HORIZON_MAX, step: int = TT_HORIZON_STEP) -> list[str]:
    return [f"y_{h}" for h in target_horizons_tt(h_max, step)]


TARGET_COLUMNS_TT = target_columns_tt()
TT_META_COLUMNS = ["symbol", "base_time", "entry_time", "sigma"]
