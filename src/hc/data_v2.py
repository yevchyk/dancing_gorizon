"""V2 dataset builder for band-specialist models (schema_v2).

Reuses the legacy candle prep (prepare_timeframes/prepare_btc_frames/_rel/_vol)
unchanged, but:
  - emits the feature matrix WITHOUT btc columns,
  - appends time-of-day features (hour_sin/hour_cos/weekday) per row,
  - uses an explicit horizon grid (the union of all bands by default),
  - clamps threshold flat beyond 180 (np.interp already does this).

The legacy hc.data path is untouched, so existing models and the live portfolio
keep working on the 302-column schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config as C
from . import config as HC
from . import schema_v2 as S2
from .data import (
    SymbolBuildStats,
    _exact_positions,
    _load_raw,
    prepare_btc_frames,
    prepare_timeframes,
    read_json_symbols,
    to_ns,
)


def _anchor_grid_v2(base, stride_min: int, days: int | None, max_horizon: int,
                    offset_min: int = 0) -> pd.DatetimeIndex:
    finite = np.isfinite(base.close)
    if not finite.any():
        return pd.DatetimeIndex([], tz="UTC")
    finite_index = base.index[finite]
    latest_entry = finite_index.max() - pd.Timedelta(minutes=int(max_horizon))
    start = finite_index.min()
    if days is not None:
        start = max(start, latest_entry - pd.Timedelta(days=days))
    if latest_entry <= start:
        return pd.DatetimeIndex([], tz="UTC")
    grid = pd.date_range(start.ceil(f"{stride_min}min"), latest_entry.floor(f"{stride_min}min"),
                         freq=f"{stride_min}min", tz="UTC")
    if offset_min:
        # phase-shift the whole grid (e.g. per-symbol jitter so a coarse stride
        # doesn't pin every snapshot to the same minute-of-hour universe-wide);
        # must stay 5m-aligned for the exact-position check below to succeed.
        grid = grid + pd.Timedelta(minutes=int(offset_min))
        grid = grid[grid <= latest_entry]
    pos, exact = _exact_positions(base.index_ns, to_ns(grid))
    ok = exact & np.isfinite(base.close[pos])
    return grid[ok]


def _build_feature_matrix_v2(anchors: pd.DatetimeIndex, prepared, n_points: int):
    """Curve matrix WITHOUT btc: per tf per point -> (rel, vol). 240 cols."""
    matrix = np.full((len(anchors), len(S2.CURVE_COLUMNS_V2)), np.nan, dtype="float32")
    valid = np.ones(len(anchors), dtype=bool)
    offsets = np.arange(n_points, dtype=np.int64)
    anchor_ns = to_ns(anchors)
    for row, t_ns in enumerate(anchor_ns):
        col = 0
        for tf in HC.TIMEFRAMES:
            data = prepared[tf.key]
            k = int(np.searchsorted(data.index_ns, t_ns, side="right") - 1)
            if k < n_points:
                valid[row] = False
                break
            idx = k - offsets
            rel = data.rel[idx]
            vol = data.vol[idx]
            if not (np.isfinite(rel).all() and np.isfinite(vol).all()):
                valid[row] = False
                break
            for i in range(n_points):
                matrix[row, col] = rel[i]; col += 1
                matrix[row, col] = vol[i]; col += 1
    return matrix, valid


def build_symbol_frame_v2(
    symbol: str,
    *,
    btc_frames: dict,
    horizons: list[int],
    stride_min: int = HC.SAMPLE_STRIDE_MIN,
    days: int | None = None,
    entry_delay_min: int = HC.EXEC_ENTRY_DELAY_MIN,
) -> tuple[pd.DataFrame, SymbolBuildStats]:
    raw = _load_raw(symbol)
    if raw is None or raw.empty:
        return pd.DataFrame(), SymbolBuildStats(symbol, "missing")
    prepared = prepare_timeframes(raw, btc_frames)
    if not prepared:
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_5m")
    base = prepared["5m"]
    max_h = int(max(horizons))
    anchors = _anchor_grid_v2(base, stride_min, days, max_h)
    if len(anchors) == 0:
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_anchors")

    features, fvalid = _build_feature_matrix_v2(anchors, prepared, HC.N_POINTS)
    anchor_pos, anchor_exact = _exact_positions(base.index_ns, to_ns(anchors))
    fvalid &= anchor_exact & np.isfinite(base.close[anchor_pos])
    if not fvalid.any():
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_valid_features", anchors=len(anchors))

    anchors_valid = anchors[fvalid]
    features_valid = features[fvalid]
    hgrid = np.array(sorted(set(int(h) for h in horizons)), dtype="int64")
    h_count = len(hgrid)
    flat_h = np.tile(hgrid, len(anchors_valid))
    base_ns = np.repeat(to_ns(anchors_valid), h_count)
    entry_ns = base_ns + int(entry_delay_min) * HC.NS_PER_MIN
    target_ns = base_ns + (flat_h + int(entry_delay_min)) * HC.NS_PER_MIN
    entry_pos, entry_exact = _exact_positions(base.index_ns, entry_ns)
    target_pos, target_exact = _exact_positions(base.index_ns, target_ns)

    entry_close = np.full(len(flat_h), np.nan, dtype="float64")
    entry_close[entry_exact] = base.close[entry_pos[entry_exact]]
    target_close = np.full(len(flat_h), np.nan, dtype="float64")
    target_close[target_exact] = base.close[target_pos[target_exact]]
    target_valid = (entry_exact & target_exact & np.isfinite(entry_close)
                    & np.isfinite(target_close) & (entry_close > 0) & (target_close > 0))
    if not target_valid.any():
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_valid_targets", anchors=len(anchors),
                                                valid_anchors=len(anchors_valid))

    feature_rows = np.repeat(features_valid, h_count, axis=0)[target_valid]
    h = flat_h[target_valid].astype("int16")
    base_ns_v = base_ns[target_valid]
    ret = (target_close[target_valid] / entry_close[target_valid]) - 1.0
    ret_pct = ret * 100.0
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
    for i, col in enumerate(S2.CURVE_COLUMNS_V2):
        data[col] = feature_rows[:, i].astype("float32", copy=False)
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
    df = df[HC.META_COLUMNS + ["entry_time", "exit_time"] + S2.FEATURE_COLUMNS_V2
            + HC.TARGET_COLUMNS + ["entry_delay_min"]]

    if df[S2.FEATURE_COLUMNS_V2].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in v2 feature columns after validity filtering")
    if len(S2.FEATURE_COLUMNS_V2) != S2.EXPECTED_FEATURE_COUNT_V2:
        raise RuntimeError("v2 feature count mismatch")

    stats = SymbolBuildStats(symbol=symbol, status="ok", rows=len(df), anchors=len(anchors),
                             valid_anchors=len(anchors_valid),
                             dropped_targets=int((~target_valid).sum()),
                             first_time=anchors_valid.min().isoformat(),
                             last_time=anchors_valid.max().isoformat())
    return df, stats


def build_dataset_shards_v2(
    *,
    out_dir: Path,
    universe_path: Path,
    horizons: list[int],
    symbols: Iterable[str] | None = None,
    stride_min: int = HC.SAMPLE_STRIDE_MIN,
    days: int | None = None,
    entry_delay_min: int = HC.EXEC_ENTRY_DELAY_MIN,
    fresh: bool = False,
) -> dict:
    selected = list(symbols) if symbols else read_json_symbols(universe_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        for p in out_dir.glob("*.parquet"):
            p.unlink()
    btc_frames = prepare_btc_frames()
    stats: list[dict] = []
    total = 0
    for i, sym in enumerate(selected, 1):
        shard = out_dir / f"{sym}.parquet"
        if shard.exists() and not fresh:
            rows = len(pd.read_parquet(shard, columns=["symbol"]))
            total += rows
            stats.append(SymbolBuildStats(sym, "cached", rows=rows).__dict__)
        else:
            df, stat = build_symbol_frame_v2(sym, btc_frames=btc_frames, horizons=horizons,
                                             stride_min=stride_min, days=days,
                                             entry_delay_min=entry_delay_min)
            if len(df):
                df.to_parquet(shard, index=False)
            total += len(df)
            stats.append(stat.__dict__)
        if i % 10 == 0 or i == len(selected):
            ok = sum(1 for s in stats if s["status"] in {"ok", "cached"} and s["rows"] > 0)
            print(f"  v2 dataset {i}/{len(selected)} shards_ok={ok} rows={total}", flush=True)

    feat_path_names = json.dumps(S2.FEATURE_COLUMNS_V2, indent=2)
    (out_dir / "feature_names.json").write_text(feat_path_names, encoding="utf-8")
    (out_dir.parent / "feature_names.json").write_text(feat_path_names, encoding="utf-8")
    ok_stats = [s for s in stats if s.get("first_time")]
    summary = {
        "out_dir": str(out_dir),
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "schema": "v2",
        "feature_columns": len(S2.FEATURE_COLUMNS_V2),
        "horizons": list(horizons),
        "symbols_requested": len(selected),
        "shards": len(sorted(out_dir.glob("*.parquet"))),
        "rows": int(total),
        "stride_min": stride_min,
        "days": days,
        "entry_delay_min": int(entry_delay_min),
        "valid_base_time_min": min((s["first_time"] for s in ok_stats), default=None),
        "valid_base_time_max": max((s["last_time"] for s in ok_stats), default=None),
        "stats": stats,
    }
    (out_dir.parent / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
