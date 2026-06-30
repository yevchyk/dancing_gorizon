"""Recent Unicorn simulation on the original trained coin universe.

Unlike the older helper scripts, this one:
  * uses the current pulse00 profile exit horizon;
  * filters stale symbols per anchor, matching live snapshot freshness;
  * requires an exit candle to exist before a candidate is allowed;
  * runs on the original fast_v2 trained universe, not the okx_liquid list.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .run_test_engine_harvest_sim import PriceSeries, simulate_engine
from .trading.fast_combo_engine import FastComboEngine, WORTHY
from .trading.timeutil import index_to_ns


NS = 60_000_000_000


class StoreBook:
    def __init__(self, store: CandleStore) -> None:
        self.store = store
        self._cache: dict[str, PriceSeries | None] = {}

    def at(self, symbol: str, t: pd.Timestamp) -> float | None:
        if symbol not in self._cache:
            c = self.store.load(symbol)
            if c is None or c.empty:
                self._cache[symbol] = None
            else:
                c = c.sort_index()
                self._cache[symbol] = PriceSeries(
                    index_to_ns(c.index),
                    c["close"].to_numpy("float64"),
                )
        ps = self._cache[symbol]
        return None if ps is None else ps.at(t)


def trained_symbols(store: CandleStore) -> list[str]:
    trained = {p.stem for p in FC.FAST_CHUNKS_DIR.glob("*.parquet")}
    available = set(store.symbols())
    return sorted((trained & available) - set(C.BLACKLIST_SYMBOLS))


def available_window(store: CandleStore, symbols: list[str], hours: float, exit_min: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    rows = []
    for sym in symbols:
        try:
            c = store.load(sym)
            if c is None or c.empty:
                continue
            c = c.sort_index()
            rows.append({"symbol": sym, "min_ts": c.index.min(), "max_ts": c.index.max(), "rows": len(c)})
        except Exception:
            continue
    cov = pd.DataFrame(rows)
    if cov.empty:
        raise SystemExit("no candles for trained symbols")
    latest = pd.to_datetime(cov["max_ts"], utc=True).max()
    end = (latest - pd.Timedelta(minutes=exit_min)).floor("2min")
    start = end - pd.Timedelta(hours=hours)
    return start, end, cov


def build_candidates(
    eng: FastComboEngine,
    store: CandleStore,
    symbols: list[str],
    anchors: pd.DatetimeIndex,
    exit_h: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exit_min = eng.horizon_minutes[exit_h]
    ans = anchors.as_unit("ns").asi8
    rows: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []

    for sym in symbols:
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        ts = index_to_ns(c.index)
        close = c["close"].to_numpy("float64")
        feats, valid = eng.curve.build_matrix(ts, close, ans)
        if valid.sum() == 0:
            continue

        idx_all = np.where(valid)[0]
        a_ns_all = ans[idx_all]
        entry_idx = np.searchsorted(ts, a_ns_all, "right") - 1
        exit_idx = np.searchsorted(ts, a_ns_all + exit_min * NS, "right") - 1
        fresh_entry = (
            (entry_idx >= 0)
            & (ts[np.clip(entry_idx, 0, len(ts) - 1)] >= a_ns_all - NS)
            & (exit_idx > entry_idx)
        )
        idx = idx_all[fresh_entry]
        if len(idx) == 0:
            continue

        x = pd.DataFrame(feats[idx], columns=eng.columns)
        probs: dict[str, np.ndarray] = {}
        for model_name, (model, cols) in eng._models.items():
            probs[model_name] = model.predict_proba(x[cols])[:, 1]

        up_count = np.zeros(len(idx), dtype=int)
        down_count = np.zeros(len(idx), dtype=int)
        up_score = np.zeros(len(idx), dtype="float64")
        down_score = np.zeros(len(idx), dtype="float64")
        for _full, (model_name, _side_name, side, base_thr) in WORTHY.items():
            p = probs[model_name]
            active = p >= base_thr
            headroom = np.clip((p - base_thr) / max(1e-9, 1.0 - base_thr), 0, None)
            if side > 0:
                up_count += active.astype(int)
                up_score += np.where(active, headroom, 0.0)
            else:
                down_count += active.astype(int)
                down_score += np.where(active, headroom, 0.0)

        anchor_times = anchors[idx]
        long_ok = (up_count >= 3) & (down_count == 0)
        short_ok = (down_count >= 3) & (up_count == 0)
        parts = []
        for mask, side, score in ((long_ok, 1, up_score), (short_ok, -1, down_score)):
            if mask.sum() == 0:
                continue
            at = pd.to_datetime(anchor_times[mask], utc=True)
            parts.append(pd.DataFrame({
                "engine": "unicorn_old",
                "family": "old_trained",
                "source": "pulse00",
                "signal_model": f"PulseClean3_idx0.00_exit{exit_h}",
                "symbol": sym,
                "anchor_time": at.to_numpy(),
                "day": at.strftime("%m-%d").to_numpy(),
                "side": side,
                "exit": exit_h,
                "threshold": np.nan,
                "leverage": 1.0,
                "score": 100.0 + score[mask],
            }))
        if parts:
            rows.append(pd.concat(parts, ignore_index=True))

        by_hour = pd.Series(anchor_times).dt.floor("1h").value_counts()
        for hour, n in by_hour.items():
            coverage_rows.append({"hour": pd.Timestamp(hour), "symbol": sym, "fresh_anchors": int(n)})

    cand = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    return cand, coverage


def hourly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    x = trades.copy()
    x["utc"] = pd.to_datetime(x["opened_at"], utc=True).dt.hour
    x["kyiv"] = (x["utc"] + 3) % 24
    rows = []
    for (utc, kyiv), g in x.groupby(["utc", "kyiv"], sort=True):
        rows.append({
            "utc": int(utc),
            "kyiv": int(kyiv),
            "trades": int(len(g)),
            "long%": float((g["side"] == "long").mean()),
            "win": float(g["won"].mean()),
            "avg%": float(g["net_pnl_pct"].mean()),
            "total%": float(g["net_pnl_pct"].sum()),
            "symbols": int(g["symbol"].nunique()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--top-per-scan", type=int, default=3)
    ap.add_argument("--max-open", type=int, default=3)
    ap.add_argument("--cooldown-min", type=int, default=10)
    args = ap.parse_args()

    eng = FastComboEngine("pulse00")
    exit_h = str(eng.cfg.get("exit_horizon", "10m"))
    store = CandleStore(C.CANDLES_DIR)
    syms = trained_symbols(store)
    start, end, cov = available_window(store, syms, args.hours, eng.horizon_minutes[exit_h])
    anchors = pd.date_range(start.ceil("2min"), end.floor("2min"), freq="2min")
    print(
        f"old trained Unicorn: symbols={len(syms)} exit={exit_h} "
        f"window={start} -> {end} UTC anchors={len(anchors)}"
    )
    print(
        f"store coverage max: latest={pd.to_datetime(cov['max_ts'], utc=True).max()} "
        f"median={pd.to_datetime(cov['max_ts'], utc=True).median()} "
        f"oldest={pd.to_datetime(cov['max_ts'], utc=True).min()}"
    )
    print(
        f"sim params: top={args.top_per_scan} max_open={args.max_open} "
        f"cooldown={args.cooldown_min}m"
    )

    cand, coverage = build_candidates(eng, store, syms, anchors, exit_h)
    if cand.empty:
        print("no Unicorn candidates in window")
        return
    scan_times = [pd.Timestamp(t) for t in anchors]
    trades, blocks = simulate_engine(
        "unicorn_old",
        cand,
        scan_times,
        StoreBook(store),
        harvest=False,
        top_per_scan=args.top_per_scan,
        max_open=args.max_open,
        cooldown_min=args.cooldown_min,
    )
    if trades.empty:
        print(f"candidates={len(cand)} but no trades after throttle; blocks={blocks}")
        return

    p = trades["net_pnl_pct"]
    print("\nOVERALL")
    print(
        f"candidates={len(cand)} trades={len(trades)} win={trades['won'].mean():.3f} "
        f"avg={p.mean():+.4f}% total={p.sum():+.2f}% "
        f"long={(trades['side'] == 'long').mean():.0%} symbols={trades['symbol'].nunique()}"
    )
    print(f"blocks={blocks}")

    h = hourly(trades)
    print("\nHOURLY")
    print(h.to_string(index=False, formatters={
        "long%": "{:.0%}".format,
        "win": "{:.3f}".format,
        "avg%": "{:+.4f}".format,
        "total%": "{:+.2f}".format,
    }))

    if not coverage.empty:
        cov_h = coverage.groupby("hour").agg(
            fresh_symbols=("symbol", "nunique"),
            fresh_anchor_rows=("fresh_anchors", "sum"),
        ).reset_index()
        cov_h["utc"] = pd.to_datetime(cov_h["hour"], utc=True).dt.hour
        cov_h["kyiv"] = (cov_h["utc"] + 3) % 24
        print("\nFRESH DATA BY HOUR")
        print(cov_h[["utc", "kyiv", "fresh_symbols", "fresh_anchor_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
