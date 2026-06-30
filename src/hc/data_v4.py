"""V4 builder: ONE horizon-conditioned model over 1-MINUTE horizons (2..120).

Key difference vs v2/v3: entry/exit prices are looked up on the 1-MINUTE close
series (prepare_1m), so any per-minute horizon is a valid target (5m-base targets
can't represent a 2-min horizon). Features = v3 schema (c1m + 5m/15m/1h/4h curves,
no BTC, + time). Horizons are sampled per snapshot (anchors + random) like d8 so
the row count stays bounded while the model still sees all of 2..120.
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
    SymbolBuildStats,
    _exact_positions,
    _load_raw,
    prepare_btc_frames,
    prepare_timeframes,
    read_json_symbols,
    stable_seed,
    to_ns,
)
from .data_v2 import _anchor_grid_v2
from .data_v3 import _build_feature_matrix_v3, prepare_1m

DEFAULT_ANCHORS = (2, 5, 10, 15, 30, 45, 60, 90, 120)
DEFAULT_CANDIDATES = tuple(range(2, 121))  # 2..120 by 1 minute


def _rand_horizons(n_rows, anchors, candidates, random_count, symbol, seed):
    anchors = np.array(sorted(set(int(a) for a in anchors)), dtype="int16")
    aset = set(int(a) for a in anchors)
    cand = np.array([int(c) for c in candidates if int(c) not in aset], dtype="int16")
    if random_count <= 0 or len(cand) == 0:
        return np.tile(anchors, (n_rows, 1))
    rc = min(int(random_count), len(cand))
    rng = np.random.default_rng(stable_seed(symbol, seed))
    out = np.empty((n_rows, len(anchors) + rc), dtype="int16")
    out[:, : len(anchors)] = anchors
    for i in range(n_rows):
        out[i, len(anchors):] = np.sort(rng.choice(cand, size=rc, replace=False))
    return out


def build_symbol_frame_v4(symbol, *, btc_frames, anchors=DEFAULT_ANCHORS,
                          candidates=DEFAULT_CANDIDATES, random_count=25,
                          stride_min=HC.SAMPLE_STRIDE_MIN, days=None,
                          entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN, seed=42,
                          threshold_fn=None, grid_offset_min: int = 0):
    """threshold_fn(symbol, h_int16_array) -> float32 array of per-row win/loss
    thresholds in %. Default None keeps the legacy flat HC.threshold_pct curve.
    grid_offset_min phase-shifts the snapshot grid (per-symbol jitter)."""
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
    if not fvalid.any():
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_valid_features", anchors=len(anchors_idx))

    anchors_valid = anchors_idx[fvalid]
    features_valid = features[fvalid]
    hmat = _rand_horizons(len(anchors_valid), anchors, candidates, random_count, symbol, seed)
    h_count = hmat.shape[1]
    flat_h = hmat.reshape(-1).astype("int64")
    base_ns = np.repeat(to_ns(anchors_valid), h_count)
    entry_ns = base_ns + int(entry_delay_min) * HC.NS_PER_MIN
    target_ns = base_ns + (flat_h + int(entry_delay_min)) * HC.NS_PER_MIN

    # 1-MINUTE close lookups (the whole point of v4)
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
    h = flat_h[target_valid].astype("int16")
    base_ns_v = base_ns[target_valid]
    ret = (target_close[target_valid] / entry_close[target_valid]) - 1.0
    ret_pct = ret * 100.0
    if threshold_fn is not None:
        thr = np.asarray(threshold_fn(symbol, h), dtype="float32")
        if thr.shape != h.shape:
            raise RuntimeError(f"{symbol}: threshold_fn returned shape {thr.shape}, want {h.shape}")
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
        raise RuntimeError(f"{symbol}: NaN in v4 feature columns")
    stats = SymbolBuildStats(symbol=symbol, status="ok", rows=len(df), anchors=len(anchors_idx),
                             valid_anchors=len(anchors_valid),
                             dropped_targets=int((~target_valid).sum()),
                             first_time=anchors_valid.min().isoformat(),
                             last_time=anchors_valid.max().isoformat())
    return df, stats


def build_dataset_shards_v4(*, out_dir: Path, universe_path: Path, symbols: Iterable[str] | None = None,
                            anchors=DEFAULT_ANCHORS, candidates=DEFAULT_CANDIDATES, random_count=25,
                            stride_min=HC.SAMPLE_STRIDE_MIN, days=None,
                            entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN, seed=42, fresh=False) -> dict:
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
            df, stat = build_symbol_frame_v4(sym, btc_frames=btc_frames, anchors=anchors,
                                             candidates=candidates, random_count=random_count,
                                             stride_min=stride_min, days=days,
                                             entry_delay_min=entry_delay_min, seed=seed)
            if len(df):
                df.to_parquet(shard, index=False)
            total += len(df)
            stats.append(stat.__dict__)
        if i % 10 == 0 or i == len(selected):
            ok = sum(1 for s in stats if s["status"] in {"ok", "cached"} and s["rows"] > 0)
            print(f"  v4 dataset {i}/{len(selected)} shards_ok={ok} rows={total}", flush=True)

    names = json.dumps(S3.FEATURE_COLUMNS_V3, indent=2)
    (out_dir / "feature_names.json").write_text(names, encoding="utf-8")
    (out_dir.parent / "feature_names.json").write_text(names, encoding="utf-8")
    ok_stats = [s for s in stats if s.get("first_time")]
    summary = {
        "out_dir": str(out_dir), "schema": "v4_1min_targets", "feature_columns": len(S3.FEATURE_COLUMNS_V3),
        "anchors": list(anchors), "candidates_min": min(candidates), "candidates_max": max(candidates),
        "random_count": random_count, "symbols_requested": len(selected),
        "shards": len(sorted(out_dir.glob("*.parquet"))), "rows": int(total),
        "stride_min": stride_min, "days": days, "entry_delay_min": int(entry_delay_min),
        "valid_base_time_min": min((s["first_time"] for s in ok_stats), default=None),
        "valid_base_time_max": max((s["last_time"] for s in ok_stats), default=None),
        "stats": stats,
    }
    (out_dir.parent / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
