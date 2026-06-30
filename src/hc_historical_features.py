"""Historical feature builders keyed by HC model schema."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from .hc import config as HC
from .hc import schema_v2 as S2
from .hc import schema_v3 as S3
from .hc.data import prepare_btc_frames, prepare_timeframes
from .hc.data_v3 import _build_feature_matrix_v3, prepare_1m
from .markets import get
from .run_hc_offgrid_sim import build_feature_rows as build_legacy_feature_rows


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build_v4_symbol_feature_rows(
    symbol: str,
    *,
    entries: pd.DatetimeIndex,
    horizons: tuple[int, ...],
    entry_delay_min: int,
    btc_frames,
) -> pd.DataFrame:
    raw = get(HC.STORE_KEY).load(symbol)
    if raw is None or raw.empty:
        return pd.DataFrame()
    prepared = prepare_timeframes(raw, btc_frames)
    p1m = prepare_1m(raw)
    if not prepared or p1m is None:
        return pd.DataFrame()

    base_times = entries - pd.Timedelta(minutes=int(entry_delay_min))
    features, valid = _build_feature_matrix_v3(base_times, prepared, p1m, HC.N_POINTS)
    if not bool(valid.any()):
        return pd.DataFrame()

    horizon_arr = np.array(sorted(set(int(h) for h in horizons)), dtype="int16")
    h_count = len(horizon_arr)
    base_valid = base_times[valid]
    entry_valid = entries[valid]
    feature_rows = np.repeat(features[valid], h_count, axis=0)
    h = np.tile(horizon_arr, int(valid.sum()))
    base_rep = np.repeat(base_valid.to_numpy(), h_count)
    entry_rep = np.repeat(entry_valid.to_numpy(), h_count)
    base_dt = pd.to_datetime(base_rep, utc=True)
    hsin, hcos, wd = S2.time_features(base_dt)

    data: dict[str, object] = {
        "symbol": np.repeat(symbol, len(h)),
        "base_time": base_rep,
        "entry_time": entry_rep,
    }
    for i, col in enumerate(S3.CURVE_COLUMNS_V3):
        data[col] = feature_rows[:, i].astype("float32", copy=False)
    hf = h.astype("float32")
    data["horizon_minutes"] = h
    data["horizon_log"] = np.log1p(hf).astype("float32")
    data["hour_sin"] = hsin
    data["hour_cos"] = hcos
    data["weekday"] = wd

    out = pd.DataFrame(data)
    out["base_time"] = pd.to_datetime(out["base_time"], utc=True)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    return out[["symbol", "base_time", "entry_time", *S3.FEATURE_COLUMNS_V3]]


def build_feature_rows_for_schema(
    schema: str,
    *,
    symbols: list[str],
    entries: pd.DatetimeIndex,
    horizons: tuple[int, ...],
    entry_delay_min: int,
) -> pd.DataFrame:
    if schema == "legacy":
        return build_legacy_feature_rows(
            symbols=symbols,
            entries=entries,
            horizons=horizons,
            entry_delay_min=entry_delay_min,
        )
    if schema != "v4":
        raise ValueError(f"historical scoring for schema {schema!r} is not wired")

    btc_frames = prepare_btc_frames()
    frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols, start=1):
        if idx == 1 or idx % 25 == 0 or idx == len(symbols):
            print(f"  v4 features {idx}/{len(symbols)} {symbol}", flush=True)
        try:
            df = build_v4_symbol_feature_rows(
                symbol,
                entries=entries,
                horizons=horizons,
                entry_delay_min=entry_delay_min,
                btc_frames=btc_frames,
            )
        except Exception as exc:
            print(f"  skip {symbol}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def iter_feature_row_chunks_for_schema(
    schema: str,
    *,
    symbols: list[str],
    entries: pd.DatetimeIndex,
    horizons: tuple[int, ...],
    entry_delay_min: int,
    batch_size: int = 12,
) -> Iterator[pd.DataFrame]:
    for batch in _chunks(symbols, batch_size):
        df = build_feature_rows_for_schema(
            schema,
            symbols=batch,
            entries=entries,
            horizons=horizons,
            entry_delay_min=entry_delay_min,
        )
        if not df.empty:
            yield df
