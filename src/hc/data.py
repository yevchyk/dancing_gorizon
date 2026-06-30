"""Dataset construction for the horizon-conditioned model track."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config as C
from ..markets import get
from . import config as HC


@dataclass
class PreparedTf:
    index: pd.DatetimeIndex
    index_ns: np.ndarray
    close: np.ndarray
    rel: np.ndarray
    vol: np.ndarray
    btc_rel: np.ndarray | None = None


@dataclass
class SymbolBuildStats:
    symbol: str
    status: str
    rows: int = 0
    anchors: int = 0
    valid_anchors: int = 0
    dropped_targets: int = 0
    first_time: str | None = None
    last_time: str | None = None
    message: str = ""


def to_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Return UTC nanoseconds even when pandas stores datetimes at us resolution."""
    return pd.DatetimeIndex(index).to_numpy(dtype="datetime64[ns]").astype("int64")


def stable_seed(symbol: str, seed: int = 42) -> int:
    raw = f"{symbol}:{seed}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), "little")


def read_json_symbols(path: Path = HC.UNIVERSE_PATH) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    symbols = data.get("symbols", data) if isinstance(data, dict) else data
    return [str(s) for s in symbols]


def generate_universe(path: Path = HC.UNIVERSE_PATH, expected_count: int = 218) -> dict:
    store = get(HC.STORE_KEY)
    blacklist = set(C.BLACKLIST_SYMBOLS)
    keep: list[str] = []
    rejected: list[dict] = []

    for parquet_path in store.files():
        try:
            ts = pd.to_datetime(
                pd.read_parquet(parquet_path, columns=["timestamp"])["timestamp"],
                utc=True,
            ).sort_values()
            if len(ts) < 10:
                rejected.append({"symbol": parquet_path.stem, "reason": "too_short"})
                continue
            fine = ts[ts.diff().dt.total_seconds() <= 300]
            fine_days = (
                (fine.max() - fine.min()).total_seconds() / 86400 if len(fine) else 0.0
            )
            if len(fine) and fine_days >= 200 and parquet_path.stem not in blacklist:
                keep.append(parquet_path.stem)
            else:
                rejected.append(
                    {
                        "symbol": parquet_path.stem,
                        "reason": "blacklist" if parquet_path.stem in blacklist else "fine_days_lt_200",
                        "fine_days": round(float(fine_days), 3),
                    }
                )
        except Exception as exc:  # pragma: no cover - defensive inventory path
            rejected.append({"symbol": parquet_path.stem, "reason": type(exc).__name__, "error": str(exc)})

    keep = sorted(keep)
    if len(keep) != expected_count:
        raise RuntimeError(
            f"HC universe count mismatch: expected {expected_count}, got {len(keep)}. "
            "Per brief, stop and inspect the data instead of silently changing it."
        )

    payload = {
        "store": HC.STORE_KEY,
        "store_dir": str(store.store_dir),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "expected_count": expected_count,
        "actual_count": len(keep),
        "blacklist_count": len(blacklist),
        "symbols": keep,
        "rejected_count": len(rejected),
        "rejected_sample": rejected[:50],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_raw(symbol: str) -> pd.DataFrame | None:
    return get(HC.STORE_KEY).load(symbol)


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.set_index("timestamp")
    else:
        out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    cols = ["open", "high", "low", "close", "volume"]
    return out[cols].astype("float64")


def _fine_start(index: pd.DatetimeIndex) -> pd.Timestamp | None:
    if len(index) < 2:
        return None
    diffs = pd.Series(index).diff().dt.total_seconds().to_numpy()
    positions = np.flatnonzero(diffs <= 300)
    if len(positions) == 0:
        return None
    return max(index[int(positions[0])], HC.HC_ERA_START)


def _resample_ohlcv(
    df: pd.DataFrame,
    freq: str,
    *,
    max_complete: pd.Timestamp | None = None,
    expected_count: int | None = None,
) -> pd.DataFrame:
    out = df.resample(freq, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    if expected_count is not None:
        counts = df["close"].resample(freq, label="right", closed="right").count()
        out.loc[counts < expected_count, ["open", "high", "low", "close", "volume"]] = np.nan
    if max_complete is None and len(df.index):
        max_complete = df.index.max().floor(freq)
    if max_complete is not None:
        out = out[out.index <= max_complete]
    return out


def prepare_5m(raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalise_ohlcv(raw)
    start = _fine_start(df.index)
    if start is None:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df[df.index >= start]
    if df.empty:
        return df
    return _resample_ohlcv(df, "5min", max_complete=df.index.max().floor("5min"))


def _rel(values: np.ndarray) -> np.ndarray:
    prev = np.roll(values, 1)
    out = np.full(len(values), np.nan, dtype="float32")
    mask = np.isfinite(values) & np.isfinite(prev) & (values > 0) & (prev > 0)
    out[mask] = (values[mask] / prev[mask]).astype("float32")
    if len(out):
        out[0] = np.nan
    return out


def _vol_ratio(values: np.ndarray) -> np.ndarray:
    prev = np.roll(values, 1)
    out = np.full(len(values), np.nan, dtype="float32")
    mask = np.isfinite(values) & np.isfinite(prev)
    zero = mask & (prev <= 0)
    out[zero] = 1.0
    div = mask & (prev > 0)
    out[div] = (values[div] / prev[div]).astype("float32")
    if len(out):
        out[0] = np.nan
    return out


def prepare_timeframes(raw: pd.DataFrame, btc_frames: dict[str, pd.DataFrame] | None = None) -> dict[str, PreparedTf]:
    base = prepare_5m(raw)
    if base.empty:
        return {}

    frames: dict[str, pd.DataFrame] = {"5m": base}
    for tf in HC.TIMEFRAMES:
        if tf.key == "5m":
            continue
        frames[tf.key] = _resample_ohlcv(
            base,
            tf.freq,
            max_complete=base.index.max().floor(tf.freq),
            expected_count=tf.expected_5m_bars,
        )

    prepared: dict[str, PreparedTf] = {}
    for tf in HC.TIMEFRAMES:
        frame = frames[tf.key]
        close = frame["close"].to_numpy("float64")
        volume = frame["volume"].to_numpy("float64")
        btc_rel = None
        if "btc" in tf.features:
            if btc_frames is None or tf.key not in btc_frames:
                raise RuntimeError(f"BTC frame missing for {tf.key}")
            btc_close = btc_frames[tf.key]["close"]
            btc_rel = pd.Series(_rel(btc_close.to_numpy("float64")), index=btc_close.index)
            btc_rel = btc_rel.reindex(frame.index).to_numpy("float32")
        prepared[tf.key] = PreparedTf(
            index=frame.index,
            index_ns=to_ns(frame.index),
            close=close,
            rel=_rel(close),
            vol=_vol_ratio(volume),
            btc_rel=btc_rel,
        )
    return prepared


def prepare_btc_frames() -> dict[str, pd.DataFrame]:
    raw = _load_raw(HC.BTC_SYMBOL)
    if raw is None or raw.empty:
        raise RuntimeError(f"BTC reference missing: {HC.BTC_SYMBOL}")
    base = prepare_5m(raw)
    if base.empty:
        raise RuntimeError(f"BTC reference has no 5m fine-era data: {HC.BTC_SYMBOL}")
    frames: dict[str, pd.DataFrame] = {"5m": base}
    for tf in HC.TIMEFRAMES:
        if tf.key == "5m":
            continue
        frames[tf.key] = _resample_ohlcv(
            base,
            tf.freq,
            max_complete=base.index.max().floor(tf.freq),
            expected_count=tf.expected_5m_bars,
        )
    return frames


def _exact_positions(index_ns: np.ndarray, query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(index_ns) == 0:
        return np.zeros(len(query_ns), dtype=np.int64), np.zeros(len(query_ns), dtype=bool)
    pos = np.searchsorted(index_ns, query_ns)
    valid = (pos < len(index_ns)) & (index_ns[np.minimum(pos, len(index_ns) - 1)] == query_ns)
    return pos, valid


def _anchor_grid(base: PreparedTf, stride_min: int, days: int | None) -> pd.DatetimeIndex:
    finite = np.isfinite(base.close)
    if not finite.any():
        return pd.DatetimeIndex([], tz="UTC")
    finite_index = base.index[finite]
    latest_entry = finite_index.max() - pd.Timedelta(minutes=max(HC.HORIZON_ANCHORS))
    start = finite_index.min()
    if days is not None:
        start = max(start, latest_entry - pd.Timedelta(days=days))
    if latest_entry <= start:
        return pd.DatetimeIndex([], tz="UTC")
    grid = pd.date_range(
        start.ceil(f"{stride_min}min"),
        latest_entry.floor(f"{stride_min}min"),
        freq=f"{stride_min}min",
        tz="UTC",
    )
    pos, exact = _exact_positions(base.index_ns, to_ns(grid))
    ok = exact & np.isfinite(base.close[pos])
    return grid[ok]


def _build_feature_matrix(
    anchors: pd.DatetimeIndex,
    prepared: dict[str, PreparedTf],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    no_horizon_cols = HC.FEATURE_COLUMNS[:-2]
    matrix = np.full((len(anchors), len(no_horizon_cols)), np.nan, dtype="float32")
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
            btc = data.btc_rel[idx] if "btc" in tf.features else None
            if not (np.isfinite(rel).all() and np.isfinite(vol).all()):
                valid[row] = False
                break
            if btc is not None and not np.isfinite(btc).all():
                valid[row] = False
                break
            for i in range(n_points):
                matrix[row, col] = rel[i]
                col += 1
                if btc is not None:
                    matrix[row, col] = btc[i]
                    col += 1
                matrix[row, col] = vol[i]
                col += 1
    return matrix, valid


def _horizon_matrix(
    n_rows: int,
    *,
    symbol: str,
    anchors_only: bool,
    random_count: int,
    random_step_min: int,
    seed: int,
) -> np.ndarray:
    anchor_values = np.array(HC.HORIZON_ANCHORS, dtype="int16")
    if anchors_only or random_count <= 0:
        return np.tile(anchor_values, (n_rows, 1))

    candidates = np.arange(
        min(HC.HORIZON_ANCHORS),
        max(HC.HORIZON_ANCHORS) + 1,
        random_step_min,
        dtype="int16",
    )
    candidates = candidates[~np.isin(candidates, anchor_values)]
    if len(candidates) < random_count:
        raise ValueError("Not enough random horizon candidates after excluding anchors")

    rng = np.random.default_rng(stable_seed(symbol, seed))
    out = np.empty((n_rows, len(anchor_values) + random_count), dtype="int16")
    out[:, : len(anchor_values)] = anchor_values
    for i in range(n_rows):
        picks = np.sort(rng.choice(candidates, size=random_count, replace=False)).astype("int16")
        out[i, len(anchor_values) :] = picks
    return out


def build_symbol_frame(
    symbol: str,
    *,
    btc_frames: dict[str, pd.DataFrame],
    stride_min: int = HC.SAMPLE_STRIDE_MIN,
    days: int | None = None,
    anchors_only: bool = False,
    random_count: int = HC.RANDOM_HORIZONS_PER_SNAPSHOT,
    random_step_min: int = HC.RANDOM_HORIZON_STEP_MIN,
    entry_delay_min: int = 0,
    seed: int = 42,
) -> tuple[pd.DataFrame, SymbolBuildStats]:
    raw = _load_raw(symbol)
    if raw is None or raw.empty:
        return pd.DataFrame(), SymbolBuildStats(symbol, "missing")

    prepared = prepare_timeframes(raw, btc_frames)
    if not prepared:
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_5m")

    base = prepared["5m"]
    anchors = _anchor_grid(base, stride_min, days)
    if len(anchors) == 0:
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_anchors")

    features, feature_valid = _build_feature_matrix(anchors, prepared, HC.N_POINTS)
    anchor_pos, anchor_exact = _exact_positions(base.index_ns, to_ns(anchors))
    feature_valid &= anchor_exact & np.isfinite(base.close[anchor_pos])
    if not feature_valid.any():
        return pd.DataFrame(), SymbolBuildStats(symbol, "no_valid_features", anchors=len(anchors))

    anchors_valid = anchors[feature_valid]
    anchor_pos_valid = anchor_pos[feature_valid]
    features_valid = features[feature_valid]
    horizons = _horizon_matrix(
        len(anchors_valid),
        symbol=symbol,
        anchors_only=anchors_only,
        random_count=random_count,
        random_step_min=random_step_min,
        seed=seed,
    )

    h_count = horizons.shape[1]
    flat_h = horizons.reshape(-1).astype("int64")
    base_ns = np.repeat(to_ns(anchors_valid), h_count)
    entry_ns = base_ns + int(entry_delay_min) * HC.NS_PER_MIN
    target_ns = base_ns + (flat_h + int(entry_delay_min)) * HC.NS_PER_MIN
    entry_pos, entry_exact = _exact_positions(base.index_ns, entry_ns)
    target_pos, target_exact = _exact_positions(base.index_ns, target_ns)

    entry_close = np.full(len(flat_h), np.nan, dtype="float64")
    entry_close[entry_exact] = base.close[entry_pos[entry_exact]]
    target_close = np.full(len(flat_h), np.nan, dtype="float64")
    target_close[target_exact] = base.close[target_pos[target_exact]]
    target_valid = (
        entry_exact
        & target_exact
        & np.isfinite(entry_close)
        & np.isfinite(target_close)
        & (entry_close > 0)
        & (target_close > 0)
    )
    if not target_valid.any():
        return pd.DataFrame(), SymbolBuildStats(
            symbol,
            "no_valid_targets",
            anchors=len(anchors),
            valid_anchors=len(anchors_valid),
            dropped_targets=int(len(target_valid)),
        )

    feature_rows = np.repeat(features_valid, h_count, axis=0)[target_valid]
    h = flat_h[target_valid].astype("int16")
    ret = (target_close[target_valid] / entry_close[target_valid]) - 1.0
    ret_pct = ret * 100.0
    thr = np.array([HC.threshold_pct(int(x)) for x in h], dtype="float32")
    up_label = (ret_pct >= thr).astype("int8")
    down_label = (ret_pct <= -thr).astype("int8")
    weight = (1.0 + np.minimum(np.abs(ret_pct) / 3.0, 1.0) * 4.0).astype("float32")

    data: dict[str, object] = {
        "symbol": np.full(len(h), symbol, dtype=object),
        "base_time": pd.to_datetime(base_ns[target_valid], unit="ns", utc=True),
        "entry_time": pd.to_datetime(entry_ns[target_valid], unit="ns", utc=True),
        "exit_time": pd.to_datetime(target_ns[target_valid], unit="ns", utc=True),
    }
    no_horizon_cols = HC.FEATURE_COLUMNS[:-2]
    for i, col in enumerate(no_horizon_cols):
        data[col] = feature_rows[:, i].astype("float32", copy=False)
    data["horizon_minutes"] = h
    data["horizon_log"] = np.log1p(h.astype("float32")).astype("float32")
    data["up_label"] = up_label
    data["down_label"] = down_label
    data["weight"] = weight
    data["ret"] = ret.astype("float32")
    data["ret_pct"] = ret_pct.astype("float32")
    data["thr_pct"] = thr
    data["entry_delay_min"] = np.full(len(h), int(entry_delay_min), dtype="int16")
    df = pd.DataFrame(data)
    df = df[
        HC.META_COLUMNS
        + ["entry_time", "exit_time"]
        + HC.FEATURE_COLUMNS
        + HC.TARGET_COLUMNS
        + ["entry_delay_min"]
    ]

    if df[HC.FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError(f"{symbol}: NaN in HC feature columns after validity filtering")
    if len(HC.FEATURE_COLUMNS) != HC.EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"feature count mismatch: {len(HC.FEATURE_COLUMNS)} != {HC.EXPECTED_FEATURE_COUNT}"
        )

    stats = SymbolBuildStats(
        symbol=symbol,
        status="ok",
        rows=len(df),
        anchors=len(anchors),
        valid_anchors=len(anchors_valid),
        dropped_targets=int((~target_valid).sum()),
        first_time=anchors_valid.min().isoformat(),
        last_time=anchors_valid.max().isoformat(),
    )
    return df, stats


def _iter_symbols(symbols: Iterable[str] | None, universe_path: Path, max_symbols: int | None) -> list[str]:
    selected = list(symbols) if symbols else read_json_symbols(universe_path)
    if max_symbols is not None:
        selected = selected[:max_symbols]
    return selected


def build_dataset_shards(
    *,
    out_dir: Path,
    universe_path: Path = HC.UNIVERSE_PATH,
    symbols: Iterable[str] | None = None,
    max_symbols: int | None = None,
    stride_min: int = HC.SAMPLE_STRIDE_MIN,
    days: int | None = None,
    anchors_only: bool = False,
    random_count: int = HC.RANDOM_HORIZONS_PER_SNAPSHOT,
    random_step_min: int = HC.RANDOM_HORIZON_STEP_MIN,
    entry_delay_min: int = 0,
    seed: int = 42,
    fresh: bool = False,
) -> dict:
    selected = _iter_symbols(symbols, universe_path, max_symbols)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        for path in out_dir.glob("*.parquet"):
            path.unlink()

    btc_frames = prepare_btc_frames()
    stats: list[dict] = []
    total_rows = 0
    for i, symbol in enumerate(selected, 1):
        shard = out_dir / f"{symbol}.parquet"
        if shard.exists() and not fresh:
            rows = len(pd.read_parquet(shard, columns=["symbol"]))
            total_rows += rows
            stats.append(SymbolBuildStats(symbol, "cached", rows=rows).__dict__)
        else:
            df, stat = build_symbol_frame(
                symbol,
                btc_frames=btc_frames,
                stride_min=stride_min,
                days=days,
                anchors_only=anchors_only,
                random_count=random_count,
                random_step_min=random_step_min,
                entry_delay_min=entry_delay_min,
                seed=seed,
            )
            if len(df):
                df.to_parquet(shard, index=False)
            total_rows += len(df)
            stats.append(stat.__dict__)
        if i % 10 == 0 or i == len(selected):
            ok = sum(1 for s in stats if s["status"] in {"ok", "cached"} and s["rows"] > 0)
            print(f"  hc dataset {i}/{len(selected)} shards_ok={ok} rows={total_rows}", flush=True)

    parquet_paths = sorted(out_dir.glob("*.parquet"))
    disk_bytes = sum(p.stat().st_size for p in parquet_paths)
    ok_stats = [s for s in stats if s.get("first_time") and s.get("last_time")]
    valid_min = min((s["first_time"] for s in ok_stats), default=None)
    valid_max = max((s["last_time"] for s in ok_stats), default=None)
    btc_source_max = btc_frames["5m"].index.max().isoformat() if len(btc_frames.get("5m", [])) else None
    warnings: list[str] = []
    if valid_max and btc_source_max:
        lag = pd.Timestamp(btc_source_max) - pd.Timestamp(valid_max)
        if lag > pd.Timedelta(days=1):
            warnings.append(
                "Valid HC snapshots stop more than 1 day before BTC 5m source max. "
                "This usually means strict 4h/BTC alignment rejected recent gaps; "
                "do not present the fold as latest-live until the candle gap is fixed."
            )
    summary = {
        "out_dir": str(out_dir),
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "symbols_requested": len(selected),
        "shards": len(parquet_paths),
        "rows": int(total_rows),
        "feature_columns": len(HC.FEATURE_COLUMNS),
        "disk_mb": round(disk_bytes / 1024 / 1024, 3),
        "valid_base_time_min": valid_min,
        "valid_base_time_max": valid_max,
        "btc_5m_source_max": btc_source_max,
        "warnings": warnings,
        "stride_min": stride_min,
        "days": days,
        "anchors_only": anchors_only,
        "random_count": random_count,
        "random_step_min": random_step_min,
        "entry_delay_min": int(entry_delay_min),
        "stats": stats,
    }
    (out_dir.parent / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_dataset(dataset_dir: Path, columns: list[str] | None = None) -> pd.DataFrame:
    paths = sorted(dataset_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No HC parquet shards under {dataset_dir}")
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    if "base_time" in df.columns:
        df["base_time"] = pd.to_datetime(df["base_time"], utc=True)
    return df
