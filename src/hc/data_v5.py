"""V5 builder: v4 (1-min targets, v3 curves, per-symbol cost+funding labels)
PLUS the 18-col market-regime/volatility block (BINANCE_V5_PLAN §1).

Parity by construction: every regime feature is computed by the functions in
THIS module from prepared 5m frames; the live engine must call the same
functions on the same frozen universe file. All features are causal (rolling
windows ending at the bar stamped <= base_time; funding = last SETTLED rate)
and NaN-free after warm-up (anchors with any NaN regime value are dropped).

Blocks:
  A market (BTC): btc_ret_15m/1h/4h/24h, btc_vol_1h/24h, btc_range_pos_24h
  C market (breadth over the FROZEN trade universe): breadth_above_4h,
    breadth_red_1h, panic_cascade, univ_vol_1h
  B/D symbol: rs_1h, rs_24h, sym_vol_1h/24h, sym_vol_ratio, sym_range_pos_24h
  E funding_level: last settled 8h rate / own EXPANDING median |rate| (signed
    crowding signal; expanding => causal, no full-year lookahead).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as HC
from . import schema_v2 as S2
from . import schema_v5 as S5
from .data import (
    SymbolBuildStats,
    _exact_positions,
    _load_raw,
    prepare_5m,
    prepare_btc_frames,
    prepare_timeframes,
    to_ns,
)
from .data_v2 import _anchor_grid_v2
from .data_v3 import _build_feature_matrix_v3, prepare_1m

FUNDING_SERIES_DIR = Path("data/binance/funding")

# rolling windows in 5m bars
W_15M, W_1H, W_4H, W_24H = 3, 12, 48, 288
PANIC_RET_1H = -0.02
EPS = 1e-12


# ---------- shared rolling helpers (used for BTC and per-symbol frames) ----------
def _logret(close: pd.Series, k: int) -> pd.Series:
    return np.log(close) - np.log(close.shift(k))


def _roll_vol(close: pd.Series, k: int) -> pd.Series:
    lr = np.log(close) - np.log(close.shift(1))
    return lr.rolling(k, min_periods=k).std(ddof=0)


def _range_pos(close: pd.Series, k: int) -> pd.Series:
    lo = close.rolling(k, min_periods=k).min()
    hi = close.rolling(k, min_periods=k).max()
    return ((close - lo) / (hi - lo + EPS)).clip(0.0, 1.0)


# ---------- A + C: the market frame (one pre-pass, cached) ----------
def build_market_frame(universe_symbols: Iterable[str], cache: Path | None = None,
                       fresh: bool = False) -> pd.DataFrame:
    """5m-indexed frame with MARKET_COLUMNS_V5; cached to parquet for reuse by
    dataset workers and the live engine warm start."""
    if cache and cache.exists() and not fresh:
        return pd.read_parquet(cache)

    btc_raw = _load_raw(HC.BTC_SYMBOL)
    btc = prepare_5m(btc_raw)["close"]
    out = pd.DataFrame(index=btc.index)
    out["btc_ret_15m"] = _logret(btc, W_15M)
    out["btc_ret_1h"] = _logret(btc, W_1H)
    out["btc_ret_4h"] = _logret(btc, W_4H)
    out["btc_ret_24h"] = _logret(btc, W_24H)
    out["btc_vol_1h"] = _roll_vol(btc, W_1H)
    out["btc_vol_24h"] = _roll_vol(btc, W_24H)
    out["btc_range_pos_24h"] = _range_pos(btc, W_24H)

    # breadth panel over the frozen universe
    closes: dict[str, pd.Series] = {}
    for s in universe_symbols:
        raw = _load_raw(s)
        if raw is None or raw.empty:
            continue
        c = prepare_5m(raw)["close"]
        if len(c):
            closes[s] = c
    panel = pd.DataFrame(closes).reindex(out.index)
    above = panel.gt(panel.rolling(W_4H, min_periods=W_4H).mean())
    ret1h = np.log(panel) - np.log(panel.shift(W_1H))
    lr5 = np.log(panel) - np.log(panel.shift(1))
    vol1h = lr5.rolling(W_1H, min_periods=W_1H).std(ddof=0)
    alive = panel.notna()
    n_alive = alive.sum(axis=1).replace(0, np.nan)
    out["breadth_above_4h"] = above.where(alive).sum(axis=1) / n_alive
    out["breadth_red_1h"] = ret1h.lt(0).where(ret1h.notna()).sum(axis=1) / ret1h.notna().sum(axis=1).replace(0, np.nan)
    out["panic_cascade"] = ret1h.lt(PANIC_RET_1H).where(ret1h.notna()).sum(axis=1) / ret1h.notna().sum(axis=1).replace(0, np.nan)
    out["univ_vol_1h"] = vol1h.median(axis=1)

    out = out.astype("float32")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache)
    return out


# ---------- E: funding level series per symbol ----------
def funding_level_series(symbol: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(times_ns, level) where level = settled rate / expanding median |rate|.
    Causal: at time t the last SETTLED rate and the median of rates settled
    so far. None if no stored series."""
    p = FUNDING_SERIES_DIR / f"{symbol}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.empty:
        return None
    ts = pd.to_datetime(df["timestamp"], utc=True)
    r = df["rate"].to_numpy("float64")
    med = pd.Series(np.abs(r)).expanding(min_periods=10).median().to_numpy()
    level = np.where(np.isfinite(med) & (med > 1e-6), r / np.maximum(med, 1e-6), 0.0)
    return to_ns(pd.DatetimeIndex(ts)), level.astype("float64")


def _asof_values(src_ns: np.ndarray, values: np.ndarray, at_ns: np.ndarray,
                 fill: float = np.nan) -> np.ndarray:
    pos = np.searchsorted(src_ns, at_ns, side="right") - 1
    out = np.full(len(at_ns), fill, dtype="float64")
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return out


# ---------- B + D: symbol regime arrays on its own 5m frame ----------
def symbol_regime_frame(base_close: pd.Series, btc_close_on_idx: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=base_close.index)
    out["rs_1h"] = _logret(base_close, W_1H) - _logret(btc_close_on_idx, W_1H)
    out["rs_24h"] = _logret(base_close, W_24H) - _logret(btc_close_on_idx, W_24H)
    v1 = _roll_vol(base_close, W_1H)
    v24 = _roll_vol(base_close, W_24H)
    out["sym_vol_1h"] = v1
    out["sym_vol_24h"] = v24
    out["sym_vol_ratio"] = v1 / (v24 + EPS)
    out["sym_range_pos_24h"] = _range_pos(base_close, W_24H)
    return out


# ---------- the v5 symbol frame builder (v4 + regime block) ----------
def build_symbol_frame_v5(symbol, *, btc_frames, market: pd.DataFrame,
                          anchors, candidates, random_count=3,
                          stride_min=60, days=None,
                          entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN, seed=42,
                          threshold_fn=None, grid_offset_min: int = 0):
    from .data_v4 import _rand_horizons  # local import to avoid cycle

    raw = _load_raw(symbol)
    if raw is None or raw.empty:
        return pd.DataFrame(), SymbolBuildStats(symbol, "missing")
    prepared = prepare_timeframes(raw, btc_frames)
    p1m = prepare_1m(raw)
    if not prepared or p1m is None:
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_5m_or_1m")
    base = prepared["5m"]
    max_h = int(max(candidates))
    anchors_idx = _anchor_grid_v2(base, stride_min, days, max_h, offset_min=grid_offset_min)
    if len(anchors_idx) == 0:
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_anchors")

    features, fvalid = _build_feature_matrix_v3(anchors_idx, prepared, p1m, HC.N_POINTS)
    anchor_pos, anchor_exact = _exact_positions(base.index_ns, to_ns(anchors_idx))
    fvalid &= anchor_exact & np.isfinite(base.close[anchor_pos])

    # ---- regime block at anchors ----
    base_close = pd.Series(base.close, index=base.index)
    btc5 = btc_frames["5m"]["close"].reindex(base.index)
    sym_reg = symbol_regime_frame(base_close, btc5)
    market_ns = to_ns(market.index)
    anchors_ns = to_ns(anchors_idx)
    reg_cols = {}
    for c in S5.MARKET_COLUMNS_V5:
        reg_cols[c] = _asof_values(market_ns, market[c].to_numpy("float64"), anchors_ns)
    sym_pos, sym_exact = _exact_positions(to_ns(sym_reg.index), anchors_ns)
    for c in ["rs_1h", "rs_24h", "sym_vol_1h", "sym_vol_24h", "sym_vol_ratio", "sym_range_pos_24h"]:
        v = np.full(len(anchors_idx), np.nan)
        v[sym_exact] = sym_reg[c].to_numpy("float64")[sym_pos[sym_exact]]
        reg_cols[c] = v
    fl = funding_level_series(symbol)
    reg_cols["funding_level"] = (_asof_values(fl[0], fl[1], anchors_ns, fill=0.0)
                                 if fl is not None else np.zeros(len(anchors_idx)))
    reg_mat = np.column_stack([reg_cols[c] for c in S5.REGIME_COLUMNS_V5]).astype("float32")
    fvalid &= np.isfinite(reg_mat).all(axis=1)
    if not fvalid.any():
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_valid_features", anchors=len(anchors_idx))

    anchors_valid = anchors_idx[fvalid]
    features_valid = features[fvalid]
    reg_valid = reg_mat[fvalid]
    hmat = _rand_horizons(len(anchors_valid), anchors, candidates, random_count, symbol, seed)
    h_count = hmat.shape[1]
    flat_h = hmat.reshape(-1).astype("int64")
    base_ns = np.repeat(to_ns(anchors_valid), h_count)
    entry_ns = base_ns + int(entry_delay_min) * HC.NS_PER_MIN
    target_ns = base_ns + (flat_h + int(entry_delay_min)) * HC.NS_PER_MIN

    entry_pos, entry_exact = _exact_positions(p1m.index_ns, entry_ns)
    target_pos, target_exact = _exact_positions(p1m.index_ns, target_ns)
    entry_close = np.full(len(flat_h), np.nan, dtype="float64")
    entry_close[entry_exact] = p1m.close[entry_pos[entry_exact]]
    target_close = np.full(len(flat_h), np.nan, dtype="float64")
    target_close[target_exact] = p1m.close[target_pos[target_exact]]
    target_valid = (entry_exact & target_exact & np.isfinite(entry_close)
                    & np.isfinite(target_close) & (entry_close > 0) & (target_close > 0))
    if not target_valid.any():
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_valid_targets", anchors=len(anchors_idx),
                                                valid_anchors=len(anchors_valid))

    feature_rows = np.repeat(features_valid, h_count, axis=0)[target_valid]
    reg_rows = np.repeat(reg_valid, h_count, axis=0)[target_valid]
    h = flat_h[target_valid].astype("int16")
    base_ns_v = base_ns[target_valid]
    ret = (target_close[target_valid] / entry_close[target_valid]) - 1.0
    ret_pct = ret * 100.0
    if threshold_fn is not None:
        thr = np.asarray(threshold_fn(symbol, h), dtype="float32")
    else:
        thr = np.array([HC.threshold_pct(int(x)) for x in h], dtype="float32")
    up_label = (ret_pct >= thr).astype("int8")
    down_label = (ret_pct <= -thr).astype("int8")
    weight = (1.0 + np.minimum(np.abs(ret_pct) / 3.0, 1.0) * 4.0).astype("float32")
    base_time = pd.to_datetime(base_ns_v, unit="ns", utc=True)
    hsin, hcos, wd = S2.time_features(base_time)

    data: dict[str, object] = {
        "symbol": np.full(len(h), symbol, dtype=object),
        "base_time": base_time,
        "entry_time": pd.to_datetime(entry_ns[target_valid], unit="ns", utc=True),
        "exit_time": pd.to_datetime(target_ns[target_valid], unit="ns", utc=True),
    }
    curve_cols = S5.FEATURE_COLUMNS_V5[: feature_rows.shape[1]]
    for i, col in enumerate(curve_cols):
        data[col] = feature_rows[:, i].astype("float32", copy=False)
    for i, col in enumerate(S5.REGIME_COLUMNS_V5):
        data[col] = reg_rows[:, i].astype("float32", copy=False)
    data["horizon_minutes"] = h
    data["horizon_log"] = np.log1p(h.astype("float32")).astype("float32")
    data["hour_sin"] = hsin
    data["hour_cos"] = hcos
    data["weekday"] = wd
    data["up_label"] = up_label
    data["down_label"] = down_label
    data["weight"] = weight
    data["ret"] = ret.astype("float32")
    data["ret_pct"] = ret_pct.astype("float32")
    data["thr_pct"] = thr
    data["entry_delay_min"] = np.full(len(h), int(entry_delay_min), dtype="int16")
    df = pd.DataFrame(data)
    df = df[HC.META_COLUMNS + ["entry_time", "exit_time"] + S5.FEATURE_COLUMNS_V5
            + HC.TARGET_COLUMNS + ["entry_delay_min"]]
    if df[S5.FEATURE_COLUMNS_V5].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in v5 feature columns")
    stats = SymbolBuildStats(symbol=symbol, status="ok", rows=len(df), anchors=len(anchors_idx),
                             valid_anchors=len(anchors_valid),
                             dropped_targets=int((~target_valid).sum()),
                             first_time=anchors_valid.min().isoformat(),
                             last_time=anchors_valid.max().isoformat())
    return df, stats
