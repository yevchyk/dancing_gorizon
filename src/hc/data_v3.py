"""V3 dataset builder = V2 + a 1-minute timeframe block (fast band-A scalper).

Reuses legacy candle prep; adds a 1m frame straight from the raw (already 1m on
disk). Emits c1m_rel/vol + the v2 curve blocks (no btc) + time features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as HC
from . import schema_v2 as S2
from . import schema_v3 as S3
from .data import (
    PreparedTf,
    SymbolBuildStats,
    _exact_positions,
    _fine_start,
    _load_raw,
    _normalise_ohlcv,
    _rel,
    _vol_ratio,
    prepare_btc_frames,
    prepare_timeframes,
    read_json_symbols,
    to_ns,
)
from .data_v2 import _anchor_grid_v2


def prepare_1m(raw: pd.DataFrame) -> PreparedTf | None:
    df = _normalise_ohlcv(raw)
    start = _fine_start(df.index)
    if start is None:
        return None
    df = df[df.index >= start]
    if df.empty:
        return None
    o = df.resample("1min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    o = o[o.index <= df.index.max().floor("1min")]
    close = o["close"].to_numpy("float64")
    vol = o["volume"].to_numpy("float64")
    return PreparedTf(index=o.index, index_ns=to_ns(o.index), close=close,
                      rel=_rel(close), vol=_vol_ratio(vol))


def _build_feature_matrix_v3(anchors, prepared, p1m: PreparedTf, n_points: int):
    matrix = np.full((len(anchors), len(S3.CURVE_COLUMNS_V3)), np.nan, dtype="float32")
    valid = np.ones(len(anchors), dtype=bool)
    offsets = np.arange(n_points, dtype=np.int64)
    anchor_ns = to_ns(anchors)
    for row, t_ns in enumerate(anchor_ns):
        col = 0
        # 1m block first
        k = int(np.searchsorted(p1m.index_ns, t_ns, side="right") - 1)
        if k < n_points:
            valid[row] = False
            continue
        idx = k - offsets
        rel = p1m.rel[idx]; vol = p1m.vol[idx]
        if not (np.isfinite(rel).all() and np.isfinite(vol).all()):
            valid[row] = False
            continue
        for i in range(n_points):
            matrix[row, col] = rel[i]; col += 1
            matrix[row, col] = vol[i]; col += 1
        # then the v2 blocks (5m/15m/1h/4h, rel+vol)
        ok = True
        for tf in HC.TIMEFRAMES:
            d = prepared[tf.key]
            kk = int(np.searchsorted(d.index_ns, t_ns, side="right") - 1)
            if kk < n_points:
                ok = False; break
            ii = kk - offsets
            r = d.rel[ii]; v = d.vol[ii]
            if not (np.isfinite(r).all() and np.isfinite(v).all()):
                ok = False; break
            for i in range(n_points):
                matrix[row, col] = r[i]; col += 1
                matrix[row, col] = v[i]; col += 1
        if not ok:
            valid[row] = False
    return matrix, valid


def build_symbol_frame_v3(symbol, *, btc_frames, horizons, stride_min=HC.SAMPLE_STRIDE_MIN,
                          days=None, entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN):
    raw = _load_raw(symbol)
    if raw is None or raw.empty:
        return pd.DataFrame(), SymbolBuildStats(symbol, "missing")
    prepared = prepare_timeframes(raw, btc_frames)
    p1m = prepare_1m(raw)
    if not prepared or p1m is None:
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_5m_or_1m")
    base = prepared["5m"]
    max_h = int(max(horizons))
    anchors = _anchor_grid_v2(base, stride_min, days, max_h)
    if len(anchors) == 0:
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_anchors")

    features, fvalid = _build_feature_matrix_v3(anchors, prepared, p1m, HC.N_POINTS)
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
    for i, col in enumerate(S3.CURVE_COLUMNS_V3):
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
    df = df[HC.META_COLUMNS + ["entry_time", "exit_time"] + S3.FEATURE_COLUMNS_V3
            + HC.TARGET_COLUMNS + ["entry_delay_min"]]
    if df[S3.FEATURE_COLUMNS_V3].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in v3 feature columns")
    stats = SymbolBuildStats(symbol=symbol, status="ok", rows=len(df), anchors=len(anchors),
                             valid_anchors=len(anchors_valid),
                             dropped_targets=int((~target_valid).sum()),
                             first_time=anchors_valid.min().isoformat(),
                             last_time=anchors_valid.max().isoformat())
    return df, stats


def build_dataset_shards_v3(*, out_dir: Path, universe_path: Path, horizons: list[int],
                            symbols: Iterable[str] | None = None, stride_min=HC.SAMPLE_STRIDE_MIN,
                            days=None, entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN, fresh=False) -> dict:
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
            df, stat = build_symbol_frame_v3(sym, btc_frames=btc_frames, horizons=horizons,
                                             stride_min=stride_min, days=days, entry_delay_min=entry_delay_min)
            if len(df):
                df.to_parquet(shard, index=False)
            total += len(df)
            stats.append(stat.__dict__)
        if i % 10 == 0 or i == len(selected):
            ok = sum(1 for s in stats if s["status"] in {"ok", "cached"} and s["rows"] > 0)
            print(f"  v3 dataset {i}/{len(selected)} shards_ok={ok} rows={total}", flush=True)

    names = json.dumps(S3.FEATURE_COLUMNS_V3, indent=2)
    (out_dir / "feature_names.json").write_text(names, encoding="utf-8")
    (out_dir.parent / "feature_names.json").write_text(names, encoding="utf-8")
    ok_stats = [s for s in stats if s.get("first_time")]
    summary = {
        "out_dir": str(out_dir), "schema": "v3", "feature_columns": len(S3.FEATURE_COLUMNS_V3),
        "horizons": list(horizons), "symbols_requested": len(selected),
        "shards": len(sorted(out_dir.glob("*.parquet"))), "rows": int(total),
        "stride_min": stride_min, "days": days, "entry_delay_min": int(entry_delay_min),
        "valid_base_time_min": min((s["first_time"] for s in ok_stats), default=None),
        "valid_base_time_max": max((s["last_time"] for s in ok_stats), default=None),
        "stats": stats,
    }
    (out_dir.parent / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
