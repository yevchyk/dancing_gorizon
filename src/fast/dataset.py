"""Dataset builder for the short-horizon research track."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..database import CandleStore
from . import config as FC
from .candles import load_1m
from .curve import FastCurve

NS_PER_MIN = 60_000_000_000


def _to_ns(index: pd.DatetimeIndex) -> np.ndarray:
    return index.as_unit("ns").asi8


def _targets(ts_ns: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
             anchors_ns: np.ndarray) -> dict[str, np.ndarray]:
    n = len(anchors_ns)
    out: dict[str, np.ndarray] = {}
    for h in FC.HORIZONS:
        for k in ("ret", "mfe", "mae"):
            out[f"{k}_{h.label}"] = np.full(n, np.nan, dtype="float32")

    entry_idx = np.searchsorted(ts_ns, anchors_ns, side="right") - 1
    for i, a_ns in enumerate(anchors_ns):
        ei = int(entry_idx[i])
        if ei < 0:
            continue
        entry = close[ei]
        if not np.isfinite(entry) or entry <= 0:
            continue
        for h in FC.HORIZONS:
            end = a_ns + h.minutes * NS_PER_MIN
            fj = int(np.searchsorted(ts_ns, end, side="right"))
            if fj <= ei + 1:
                continue
            hh = high[ei + 1:fj]
            ll = low[ei + 1:fj]
            out[f"ret_{h.label}"][i] = close[fj - 1] / entry - 1.0
            out[f"mfe_{h.label}"][i] = hh.max() / entry - 1.0
            out[f"mae_{h.label}"][i] = ll.min() / entry - 1.0
    return out


def _train_anchors(candles: pd.DataFrame, train_start: pd.Timestamp,
                   holdout_start: pd.Timestamp, per_symbol: int,
                   seed: int, symbol: str) -> pd.DatetimeIndex:
    idx = candles.index[(candles.index >= train_start) & (candles.index < holdout_start)]
    if len(idx) <= per_symbol:
        return idx
    rng = np.random.default_rng(abs(hash((symbol, seed))) % (2**32))
    picks = np.sort(rng.choice(len(idx), size=per_symbol, replace=False))
    return idx[picks]


def _holdout_anchors(candles: pd.DataFrame, holdout_start: pd.Timestamp,
                     holdout_end: pd.Timestamp, step_min: int) -> pd.DatetimeIndex:
    grid = pd.date_range(holdout_start.ceil(f"{step_min}min"),
                         holdout_end.floor(f"{step_min}min"),
                         freq=f"{step_min}min")
    if grid.empty:
        return grid
    # Keep anchors that have a candle at or before the timestamp.
    return grid[grid <= candles.index.max()]


def build_symbol(symbol: str, curve: FastCurve, as_of: pd.Timestamp,
                 train_days: int, holdout_days: int, train_anchors: int,
                 holdout_step_min: int, seed: int = 42, btc=None) -> int:
    chunk = FC.FAST_CHUNKS_DIR / f"{symbol}.parquet"
    if chunk.exists():
        return len(pd.read_parquet(chunk))

    target_candles = load_1m(symbol)
    if target_candles is None or target_candles.empty:
        return 0
    target_candles = target_candles.sort_index()

    feature_candles = CandleStore(C.CANDLES_DIR).load(symbol)
    if feature_candles is None or feature_candles.empty:
        feature_candles = target_candles
    feature_candles = feature_candles.sort_index()

    max_horizon = max(h.minutes for h in FC.HORIZONS)
    holdout_end = as_of - pd.Timedelta(minutes=max_horizon + 2)
    holdout_start = holdout_end - pd.Timedelta(days=holdout_days)
    train_start = holdout_start - pd.Timedelta(days=train_days)
    required_start = train_start - pd.Timedelta(minutes=FC.CURVE_MAX_DEPTH_MIN)
    if target_candles.index.min() > train_start + pd.Timedelta(minutes=2):
        return 0
    if target_candles.index.max() < as_of - pd.Timedelta(minutes=2):
        return 0
    if feature_candles.index.min() > required_start + pd.Timedelta(minutes=5):
        return 0
    if feature_candles.index.max() < holdout_end:
        return 0

    train_idx = _train_anchors(target_candles, train_start, holdout_start, train_anchors, seed, symbol)
    hold_idx = _holdout_anchors(target_candles, holdout_start, holdout_end, holdout_step_min)
    if len(train_idx) == 0 or len(hold_idx) == 0:
        return 0

    anchors = train_idx.append(hold_idx)
    split = np.array(["train"] * len(train_idx) + ["holdout"] * len(hold_idx), dtype=object)
    anchors_ns = anchors.as_unit("ns").asi8

    feat_ts_ns = _to_ns(feature_candles.index)
    feat_close = feature_candles["close"].to_numpy("float64")
    tgt_ts_ns = _to_ns(target_candles.index)
    high = target_candles["high"].to_numpy("float64")
    low = target_candles["low"].to_numpy("float64")
    close = target_candles["close"].to_numpy("float64")

    feats, valid = curve.build_matrix(feat_ts_ns, feat_close, anchors_ns)
    tgt = _targets(tgt_ts_ns, high, low, close, anchors_ns)
    for values in tgt.values():
        valid &= np.isfinite(values)

    btc_feats = None
    if FC.BTC_CONTEXT and btc is not None:
        btc_ts_ns, btc_close, btc_curve = btc
        btc_feats, btc_valid = btc_curve.build_matrix(btc_ts_ns, btc_close, anchors_ns)
        valid &= btc_valid

    if valid.sum() == 0:
        return 0
    data = {
        "symbol": np.array([symbol] * len(anchors), dtype=object)[valid],
        "anchor_time": anchors[valid],
        "split": split[valid],
    }
    for i, col in enumerate(curve.columns()):
        data[col] = feats[valid, i]
    for col, values in tgt.items():
        data[col] = values[valid]
    if btc_feats is not None:
        for i, col in enumerate(FC.btc_columns()):
            data[col] = btc_feats[valid, i]

    FC.FAST_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(chunk, index=False)
    return int(valid.sum())


def build_dataset(symbols: list[str], as_of: pd.Timestamp, *,
                  train_days: int = FC.TRAIN_DAYS,
                  holdout_days: int = FC.HOLDOUT_DAYS,
                  train_anchors: int = FC.TRAIN_ANCHORS_PER_SYMBOL,
                  holdout_step_min: int = FC.HOLDOUT_STEP_MIN,
                  fresh: bool = False) -> Path:
    if fresh and FC.FAST_CHUNKS_DIR.exists():
        for path in FC.FAST_CHUNKS_DIR.glob("*.parquet"):
            path.unlink()
    curve = FastCurve(
        FC.CURVE_POINTS,
        FC.CURVE_MIN_STEP_MIN,
        FC.CURVE_MAX_DEPTH_MIN,
        FC.CURVE_SEGMENTS,
    )
    btc = None
    if FC.BTC_CONTEXT:
        btc_candles = CandleStore(C.CANDLES_DIR).load(FC.BTC_SYMBOL)
        if btc_candles is None or btc_candles.empty:
            raise RuntimeError(f"BTC context on but {FC.BTC_SYMBOL} candles missing")
        btc_candles = btc_candles.sort_index()
        btc_curve = FastCurve(0, FC.CURVE_MIN_STEP_MIN, FC.CURVE_MAX_DEPTH_MIN,
                              offsets_min=FC.BTC_OFFSETS_MIN)
        btc = (_to_ns(btc_candles.index), btc_candles["close"].to_numpy("float64"), btc_curve)
        print(f"  BTC context ON: {len(FC.BTC_OFFSETS_MIN)} cols from {FC.BTC_SYMBOL} "
              f"({btc_candles.index.min().date()}..{btc_candles.index.max().date()})", flush=True)
    total = 0
    kept: list[str] = []
    for i, sym in enumerate(symbols, 1):
        n = build_symbol(sym, curve, as_of, train_days, holdout_days,
                         train_anchors, holdout_step_min, btc=btc)
        if n:
            kept.append(sym)
        total += n
        if i % 10 == 0 or i == len(symbols):
            print(f"  dataset {i}/{len(symbols)} kept={len(kept)} rows~{total}", flush=True)

    frames = [pd.read_parquet(p) for p in sorted(FC.FAST_CHUNKS_DIR.glob("*.parquet"))]
    ds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = FC.FAST_DATASETS_DIR / "master.parquet"
    FC.FAST_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out, index=False)
    print(f"fast dataset: {len(ds)} rows, {ds.shape[1]} cols -> {out}")
    if len(ds):
        print(ds.groupby("split")["anchor_time"].agg(["min", "max", "count"]).to_string())
    return out
