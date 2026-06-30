"""ТТ dataset builder — ONE row per (symbol, scan), target = forward price CURVE.

Reuses the proven hc candle-prep (prepare_timeframes / prepare_1m / regime block)
unchanged; the new parts are:
  * MAXIMAL feature matrix (c1m + 5m/15m/1h(+btc)/4h(+btc)) at N_POINTS=45;
  * a MULTI-OUTPUT target: vol-normalized cumulative log-return on a 1-min grid
    1..h_max, computed on the 1-MINUTE close series (entry at base+entry_delay);
  * a holdout guard: no target node may read past `cutoff` (last 4 days reserved
    for the user's own test — never built into the dataset).

Horizon is the OUTPUT axis (no horizon feature). The legacy hc/v2..v5 pipeline is
untouched — this module only imports from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..hc import config as HC
from ..hc import schema_v2 as S2
from ..hc import schema_v5 as S5
from ..hc.data import (
    SymbolBuildStats,
    _exact_positions,
    _load_raw,
    prepare_timeframes,
    to_ns,
)
from ..hc.data_v2 import _anchor_grid_v2
from ..hc.data_v3 import prepare_1m
from ..hc.data_v5 import _asof_values, funding_level_series, symbol_regime_frame
from . import schema_tt as STT


def _empty(symbol: str, status: str, **kw) -> tuple[pd.DataFrame, SymbolBuildStats]:
    return pd.DataFrame(), SymbolBuildStats(symbol, status, **kw)


def _sigma_1m(p1m, window: int = 1440, min_periods: int | None = None) -> np.ndarray:
    """Causal trailing std of 1-min log-returns (per-symbol vol scale for the
    target normalization). NaN until min_periods non-NaN returns are available."""
    close = np.asarray(p1m.close, dtype="float64")
    prev = np.roll(close, 1)
    lr = np.full(len(close), np.nan)
    m = np.isfinite(close) & np.isfinite(prev) & (close > 0) & (prev > 0)
    lr[m] = np.log(close[m]) - np.log(prev[m])
    if len(lr):
        lr[0] = np.nan
    mp = min_periods if min_periods is not None else max(60, window // 6)
    return pd.Series(lr).rolling(window, min_periods=mp).std(ddof=0).to_numpy()


def _build_feature_matrix_tt(anchors, prepared, p1m, n_points: int):
    """MAXIMAL curve matrix: c1m(rel,vol) then 5m/15m(rel,vol) then 1h/4h(rel,btc,vol).
    Column order matches schema_tt.curve_columns_tt(n_points)."""
    n_cols = len(STT.curve_columns_tt(n_points))
    matrix = np.full((len(anchors), n_cols), np.nan, dtype="float32")
    valid = np.ones(len(anchors), dtype=bool)
    offsets = np.arange(n_points, dtype=np.int64)
    anchor_ns = to_ns(anchors)
    for row, t_ns in enumerate(anchor_ns):
        col = 0
        # 1m microstructure block first
        k = int(np.searchsorted(p1m.index_ns, t_ns, side="right") - 1)
        if k < n_points:
            valid[row] = False
            continue
        idx = k - offsets
        rel = p1m.rel[idx]
        vol = p1m.vol[idx]
        if not (np.isfinite(rel).all() and np.isfinite(vol).all()):
            valid[row] = False
            continue
        for i in range(n_points):
            matrix[row, col] = rel[i]; col += 1
            matrix[row, col] = vol[i]; col += 1
        # then 5m/15m/1h/4h (btc curve where the timeframe carries it)
        ok = True
        for tf in HC.TIMEFRAMES:
            d = prepared[tf.key]
            kk = int(np.searchsorted(d.index_ns, t_ns, side="right") - 1)
            if kk < n_points:
                ok = False; break
            ii = kk - offsets
            r = d.rel[ii]; v = d.vol[ii]
            btc = d.btc_rel[ii] if "btc" in tf.features else None
            if not (np.isfinite(r).all() and np.isfinite(v).all()):
                ok = False; break
            if btc is not None and not np.isfinite(btc).all():
                ok = False; break
            for i in range(n_points):
                matrix[row, col] = r[i]; col += 1
                if btc is not None:
                    matrix[row, col] = btc[i]; col += 1
                matrix[row, col] = v[i]; col += 1
        if not ok:
            valid[row] = False
    return matrix, valid


def _regime_matrix(symbol, base, anchors_idx, anchors_ns, btc_frames, market):
    """The 18 v5 regime scalars at the anchors (asof market frame + symbol regime
    + funding). Same functions the v5 live engine must call -> parity by construction."""
    base_close = pd.Series(base.close, index=base.index)
    btc5 = btc_frames["5m"]["close"].reindex(base.index)
    sym_reg = symbol_regime_frame(base_close, btc5)
    market_ns = to_ns(market.index)
    cols: dict[str, np.ndarray] = {}
    for c in S5.MARKET_COLUMNS_V5:
        cols[c] = _asof_values(market_ns, market[c].to_numpy("float64"), anchors_ns)
    sym_pos, sym_exact = _exact_positions(to_ns(sym_reg.index), anchors_ns)
    for c in ["rs_1h", "rs_24h", "sym_vol_1h", "sym_vol_24h", "sym_vol_ratio", "sym_range_pos_24h"]:
        v = np.full(len(anchors_idx), np.nan)
        v[sym_exact] = sym_reg[c].to_numpy("float64")[sym_pos[sym_exact]]
        cols[c] = v
    fl = funding_level_series(symbol)
    cols["funding_level"] = (_asof_values(fl[0], fl[1], anchors_ns, fill=0.0)
                             if fl is not None else np.zeros(len(anchors_idx)))
    mat = np.column_stack([cols[c] for c in S5.REGIME_COLUMNS_V5]).astype("float32")
    return mat, np.isfinite(mat).all(axis=1)


def build_symbol_curve_tt(symbol, *, btc_frames, market, cutoff: pd.Timestamp | None,
                          h_max: int = STT.TT_HORIZON_MAX, step: int = STT.TT_HORIZON_STEP,
                          n_points: int = STT.TT_N_POINTS, stride_min: int = 60,
                          days: int | None = None, entry_delay_min: int = HC.EXEC_ENTRY_DELAY_MIN,
                          seed: int = 42, grid_offset_min: int = 0, vol_window: int = 1440,
                          with_regime: bool = True, target_mode: str = "volnorm",
                          ) -> tuple[pd.DataFrame, SymbolBuildStats]:
    raw = _load_raw(symbol)
    if raw is None or raw.empty:
        return _empty(symbol, "missing")
    prepared = prepare_timeframes(raw, btc_frames)
    p1m = prepare_1m(raw)
    if not prepared or p1m is None:
        return _empty(symbol, "no_5m_or_1m")
    base = prepared["5m"]

    # decision grid on the 5m base; leave room for entry_delay + the whole curve.
    anchors_idx = _anchor_grid_v2(base, stride_min, days, entry_delay_min + h_max,
                                  offset_min=grid_offset_min)
    if len(anchors_idx) == 0:
        return _empty(symbol, "no_anchors")
    # HOLDOUT GUARD: drop anchors whose curve would read past the cutoff.
    if cutoff is not None:
        keep = (anchors_idx + pd.Timedelta(minutes=entry_delay_min + h_max)) <= cutoff
        anchors_idx = anchors_idx[keep]
        if len(anchors_idx) == 0:
            return _empty(symbol, "past_cutoff")
    n_anchor = len(anchors_idx)
    anchors_ns = to_ns(anchors_idx)

    features, valid = _build_feature_matrix_tt(anchors_idx, prepared, p1m, n_points)
    base_pos, base_exact = _exact_positions(base.index_ns, anchors_ns)
    valid &= base_exact & np.isfinite(base.close[base_pos])

    if with_regime:
        reg_mat, reg_ok = _regime_matrix(symbol, base, anchors_idx, anchors_ns, btc_frames, market)
        valid &= reg_ok
    else:
        reg_mat = np.zeros((n_anchor, len(S5.REGIME_COLUMNS_V5)), dtype="float32")

    # ---- target curve on the 1-MINUTE close series ----
    sigma_full = _sigma_1m(p1m, vol_window)
    entry_ns = anchors_ns + int(entry_delay_min) * HC.NS_PER_MIN
    entry_pos, entry_exact = _exact_positions(p1m.index_ns, entry_ns)
    hs = np.arange(step, h_max + 1, step, dtype=np.int64)
    n1 = len(p1m.close)
    # p1m index is a contiguous 1-min grid, so exit(h) is positional: entry_pos + h.
    valid &= entry_exact & (entry_pos + int(hs[-1]) < n1)
    if not valid.any():
        return _empty(symbol, "no_valid_targets", anchors=n_anchor)

    vi = np.flatnonzero(valid)
    ep = entry_pos[vi]
    entry_close = p1m.close[ep]
    exit_close = p1m.close[ep[:, None] + hs[None, :]]            # [nv, H]
    sig = sigma_full[ep]
    row_ok = (np.isfinite(entry_close) & (entry_close > 0)
              & np.isfinite(exit_close).all(axis=1) & (exit_close > 0).all(axis=1)
              & np.isfinite(sig) & (sig > 0))
    vi = vi[row_ok]
    if len(vi) == 0:
        return _empty(symbol, "no_valid_targets", anchors=n_anchor)
    entry_close = entry_close[row_ok]
    exit_close = exit_close[row_ok]
    sig = sig[row_ok]
    cumret = np.log(exit_close) - np.log(entry_close)[:, None]   # raw cumulative log-return
    if target_mode == "ratioA":                                  # price ratio vs ENTRY (cumulative): exit/entry
        target_norm = (exit_close / entry_close[:, None]).astype("float32")
    elif target_mode == "ratioB":                                # price ratio vs PREVIOUS minute (per-step)
        prev = np.concatenate([entry_close[:, None], exit_close[:, :-1]], axis=1)
        target_norm = (exit_close / prev).astype("float32")
    else:                                                        # "volnorm": vol-normalized cumret (legacy default)
        target_norm = (cumret / sig[:, None]).astype("float32")

    feats = features[vi]
    regs = reg_mat[vi]
    anchors_keep = anchors_idx[vi]
    entry_keep_ns = entry_ns[vi]
    hsin, hcos, wd = S2.time_features(anchors_keep)

    feat_cols = STT.feature_names_tt(n_points, include_regime=with_regime)
    curve_cols = STT.curve_columns_tt(n_points)
    tgt_cols = STT.target_columns_tt(h_max, step)

    data: dict[str, object] = {
        "symbol": np.full(len(vi), symbol, dtype=object),
        "base_time": anchors_keep,
        "entry_time": pd.to_datetime(entry_keep_ns, unit="ns", utc=True),
        "sigma": sig.astype("float32"),
    }
    for i, col in enumerate(curve_cols):
        data[col] = feats[:, i].astype("float32", copy=False)
    if with_regime:
        for i, col in enumerate(S5.REGIME_COLUMNS_V5):
            data[col] = regs[:, i].astype("float32", copy=False)
    data["hour_sin"] = hsin
    data["hour_cos"] = hcos
    data["weekday"] = wd
    for j, col in enumerate(tgt_cols):
        data[col] = target_norm[:, j]

    df = pd.DataFrame(data)[STT.TT_META_COLUMNS + feat_cols + tgt_cols]
    if df[feat_cols].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in TT feature columns")
    if df[tgt_cols].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in TT target columns")
    stats = SymbolBuildStats(symbol=symbol, status="ok", rows=len(df), anchors=n_anchor,
                             valid_anchors=len(vi), dropped_targets=int(n_anchor - len(vi)),
                             first_time=anchors_keep.min().isoformat(),
                             last_time=anchors_keep.max().isoformat())
    return df, stats
