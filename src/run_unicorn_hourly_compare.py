"""Compare fast_v3 Unicorn vs the older fast_v2 Unicorn over a recent window.

The report is chronological by hour, so it is useful for checking whether the
market regime is improving rather than only aggregating by UTC hour-of-day.

  python -m src.run_unicorn_hourly_compare --hours 48
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .run_unicorn_old_recent import StoreBook, build_candidates as build_old_candidates, trained_symbols
from .trading.fast_combo_engine import FastComboEngine
from .trading.fast_v3_engine import FastV3Engine, V3_DATASET, V3_LABELS
from .trading.timeutil import index_to_ns

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NS_MIN = 60_000_000_000
COST = FC.EVAL_COST


class PriceBook(StoreBook):
    pass


def _settings() -> dict:
    defaults = {
        "trade_size_usd": 10.0,
        "top_per_scan": 3,
        "max_concurrent": 999,
        "cooldown_min": 10,
    }
    path = C.ROOT / "settings.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in list(defaults):
            if key in data:
                defaults[key] = data[key]
    return defaults


def _store_latest(store: CandleStore, symbols: list[str]) -> pd.Timestamp:
    latest = []
    for sym in symbols:
        c = store.load(sym)
        if c is not None and not c.empty:
            latest.append(pd.to_datetime(c.index.max(), utc=True))
    if not latest:
        raise SystemExit("no candles in store")
    return max(latest)


def _v3_symbols(store: CandleStore) -> list[str]:
    if V3_DATASET.exists():
        symbols = list(pd.read_parquet(V3_DATASET, columns=["symbol"])["symbol"].unique())
    else:
        symbols = store.symbols()
    return sorted(sym for sym in symbols if store.load(sym) is not None)


def _entry_exit_ok(ts: np.ndarray, anchors_ns: np.ndarray, horizon_min: int) -> np.ndarray:
    entry_idx = np.searchsorted(ts, anchors_ns, "right") - 1
    exit_idx = np.searchsorted(ts, anchors_ns + horizon_min * NS_MIN, "right") - 1
    fresh = (
        (entry_idx >= 0)
        & (exit_idx > entry_idx)
        & (ts[np.clip(entry_idx, 0, len(ts) - 1)] >= anchors_ns - NS_MIN)
        & (exit_idx < len(ts))
    )
    return fresh


def build_v3_candidates(
    eng: FastV3Engine,
    store: CandleStore,
    symbols: list[str],
    anchors: pd.DatetimeIndex,
) -> pd.DataFrame:
    threshold = float(eng.cfg["agreement_threshold"])
    min_agree = int(eng.cfg["min_agree"])
    exit_h = str(eng.cfg["exit_horizon"])
    exit_min = int(eng.horizon_minutes[exit_h])
    notional = float(eng.cfg["notional_usd"])
    anchors_ns = anchors.as_unit("ns").asi8
    rows = []

    for si, sym in enumerate(symbols, 1):
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        ts = index_to_ns(c.index)
        close = c["close"].to_numpy("float64")
        feats, valid = eng.curve.build_matrix(ts, close, anchors_ns)
        ok = valid & _entry_exit_ok(ts, anchors_ns, exit_min)
        idx = np.where(ok)[0]
        if len(idx) == 0:
            continue

        x = pd.DataFrame(feats[idx], columns=eng.columns)
        probs: dict[str, np.ndarray] = {}
        for label in V3_LABELS:
            for side in ("up", "down"):
                model, cols = eng._models[f"{side}_{label}"]
                probs[f"{side}_{label}"] = model.predict_proba(x[cols])[:, 1]

        up_count = np.zeros(len(idx), dtype=int)
        down_count = np.zeros(len(idx), dtype=int)
        up_score = np.zeros(len(idx), dtype="float64")
        down_score = np.zeros(len(idx), dtype="float64")
        for label in V3_LABELS:
            up = probs[f"up_{label}"]
            down = probs[f"down_{label}"]
            up_active = up >= threshold
            down_active = down >= threshold
            up_count += up_active.astype(int)
            down_count += down_active.astype(int)
            up_score += np.where(up_active, (up - threshold) / max(1e-9, 1.0 - threshold), 0.0)
            down_score += np.where(down_active, (down - threshold) / max(1e-9, 1.0 - threshold), 0.0)

        anchor_times = anchors[idx]
        for mask, side_int, score in (
            ((up_count >= min_agree) & (down_count == 0), 1, up_score),
            ((down_count >= min_agree) & (up_count == 0), -1, down_score),
        ):
            if mask.sum() == 0:
                continue
            at = pd.to_datetime(anchor_times[mask], utc=True)
            rows.append(pd.DataFrame({
                "engine": "unicorn_v3",
                "family": "fast_v3",
                "source": "unicorn_v2",
                "signal_model": f"unicorn_v2_agree{min_agree}_exit{exit_h}",
                "symbol": sym,
                "anchor_time": at.to_numpy(),
                "side": side_int,
                "exit": exit_h,
                "score": 100.0 + score[mask],
                "size_usd": notional,
            }))

        if si % 25 == 0 or si == len(symbols):
            print(f"  v3 candidates {si}/{len(symbols)} last={sym}", flush=True)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def simulate(
    name: str,
    cand: pd.DataFrame,
    scan_times: list[pd.Timestamp],
    book: PriceBook,
    horizon_minutes: dict[str, int],
    *,
    top_per_scan: int,
    max_open: int,
    cooldown_min: int,
    default_size_usd: float,
) -> pd.DataFrame:
    if cand.empty:
        return pd.DataFrame()

    d = cand.copy()
    d["anchor_time"] = pd.to_datetime(d["anchor_time"], utc=True)
    by_time = {
        pd.Timestamp(t): g.sort_values("score", ascending=False).head(top_per_scan)
        for t, g in d.groupby("anchor_time", sort=False)
    }
    open_pos: dict[str, dict] = {}
    last_trade_at: dict[str, pd.Timestamp] = {}
    trades: list[dict] = []

    def close_due(now: pd.Timestamp, flush: bool = False) -> None:
        for sym, pos in list(open_pos.items()):
            if not flush and now < pos["deadline"]:
                continue
            close_at = pos["deadline"]
            px = book.at(sym, close_at)
            if px is None:
                continue
            side = int(pos["side"])
            gross = side * (px / pos["entry_price"] - 1.0)
            net = gross - COST
            trades.append({
                "engine": name,
                "symbol": sym,
                "side": "long" if side > 0 else "short",
                "opened_at": pos["opened_at"],
                "closed_at": close_at,
                "net_pnl_pct": net * 100.0,
                "won": int(net > 0),
                "size_usd": pos["size_usd"],
                "usd": net * pos["size_usd"],
            })
            del open_pos[sym]

    for now in scan_times:
        close_due(now)
        g = by_time.get(pd.Timestamp(now))
        if g is None or g.empty:
            continue
        for row in g.itertuples(index=False):
            sym = str(row.symbol)
            if sym in open_pos:
                continue
            if len(open_pos) >= max_open:
                continue
            last = last_trade_at.get(sym)
            if last is not None and now < last + pd.Timedelta(minutes=cooldown_min):
                continue
            entry = book.at(sym, now)
            if entry is None:
                continue
            exit_label = str(row.exit)
            size_usd = float(getattr(row, "size_usd", default_size_usd) or default_size_usd)
            open_pos[sym] = {
                "symbol": sym,
                "side": int(row.side),
                "exit": exit_label,
                "size_usd": size_usd,
                "entry_price": entry,
                "opened_at": now,
                "deadline": now + pd.Timedelta(minutes=horizon_minutes[exit_label]),
            }
            last_trade_at[sym] = now

    close_due(scan_times[-1], flush=True)
    return pd.DataFrame(trades)


def hourly(trades: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    cum = 0.0
    x = trades.copy()
    if not x.empty:
        x["hour"] = pd.to_datetime(x["opened_at"], utc=True).dt.floor("1h")
    for hour in hours:
        g = x[x["hour"] == hour] if not x.empty else x
        usd = float(g["usd"].sum()) if not g.empty else 0.0
        cum += usd
        rows.append({
            "hour": hour,
            "n": int(len(g)),
            "long": int((g["side"] == "long").sum()) if not g.empty else 0,
            "short": int((g["side"] == "short").sum()) if not g.empty else 0,
            "win": float(g["won"].mean()) if not g.empty else np.nan,
            "avg_pct": float(g["net_pnl_pct"].mean()) if not g.empty else np.nan,
            "usd": usd,
            "cum_usd": cum,
        })
    return pd.DataFrame(rows)


def print_summary(label: str, trades: pd.DataFrame, hours: float) -> None:
    if trades.empty:
        print(f"\n--- {label} ---\nno trades")
        return
    print(f"\n--- {label} ---")
    print(
        f"trades={len(trades)}  trades/day={len(trades)/hours*24:.1f}  "
        f"win={trades['won'].mean():.3f}  avg%={trades['net_pnl_pct'].mean():+.4f}  "
        f"total$={trades['usd'].sum():+.2f}  $/day={trades['usd'].sum()/hours*24:+.2f}  "
        f"long={(trades['side'] == 'long').sum()} short={(trades['side'] == 'short').sum()}  "
        f"symbols={trades['symbol'].nunique()}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=48.0)
    ap.add_argument("--cadence", type=int, default=2)
    ap.add_argument("--top-per-scan", type=int, default=None)
    ap.add_argument("--max-open", type=int, default=None)
    ap.add_argument("--cooldown-min", type=int, default=None)
    args = ap.parse_args()

    settings = _settings()
    top_per_scan = int(args.top_per_scan or settings["top_per_scan"])
    max_open = int(args.max_open or settings["max_concurrent"])
    cooldown_min = int(args.cooldown_min or settings["cooldown_min"])
    base_size = float(settings["trade_size_usd"])

    store = CandleStore(C.CANDLES_DIR)
    v3_eng = FastV3Engine("unicorn_v2")
    old_eng = FastComboEngine("pulse00")
    v3_symbols = _v3_symbols(store)
    old_symbols = trained_symbols(store)
    latest = min(_store_latest(store, v3_symbols), _store_latest(store, old_symbols))
    end = (latest - pd.Timedelta(minutes=20)).floor(f"{args.cadence}min")
    start = end - pd.Timedelta(hours=float(args.hours))
    anchors = pd.date_range(start, end, freq=f"{args.cadence}min")
    hour_index = pd.date_range(start.floor("1h"), end.floor("1h"), freq="1h")

    print(
        f"window={start} -> {end} UTC  hours={args.hours:g}  anchors={len(anchors)}  "
        f"top={top_per_scan} max_open={max_open} cooldown={cooldown_min}m"
    )
    print(
        f"sizes: v3=${float(v3_eng.cfg['notional_usd']):.0f}; "
        f"old=${base_size * float(old_eng.cfg.get('pulse_size_mult', 1.0)):.0f}"
    )

    print("\nbuilding fast_v3 unicorn candidates...", flush=True)
    v3_cand = build_v3_candidates(v3_eng, store, v3_symbols, anchors)
    print(f"v3 candidates={len(v3_cand)}", flush=True)

    print("\nbuilding old fast_v2 unicorn candidates...", flush=True)
    old_exit = str(old_eng.cfg.get("exit_horizon", "8m"))
    old_cand, _coverage = build_old_candidates(old_eng, store, old_symbols, anchors, old_exit)
    if not old_cand.empty:
        old_cand["size_usd"] = base_size * float(old_eng.cfg.get("pulse_size_mult", 1.0))
    print(f"old candidates={len(old_cand)}", flush=True)

    book = PriceBook(store)
    common_h = {**v3_eng.horizon_minutes, **old_eng.horizon_minutes}
    v3_trades = simulate(
        "unicorn_v3", v3_cand, list(anchors), book, common_h,
        top_per_scan=top_per_scan, max_open=max_open, cooldown_min=cooldown_min,
        default_size_usd=float(v3_eng.cfg["notional_usd"]),
    )
    old_trades = simulate(
        "unicorn_old", old_cand, list(anchors), book, common_h,
        top_per_scan=top_per_scan, max_open=max_open, cooldown_min=cooldown_min,
        default_size_usd=base_size * float(old_eng.cfg.get("pulse_size_mult", 1.0)),
    )

    print_summary("fast_v3 unicorn_v2", v3_trades, float(args.hours))
    print_summary("old fast_v2 pulse00", old_trades, float(args.hours))

    v3h = hourly(v3_trades, hour_index).add_prefix("v3_")
    oldh = hourly(old_trades, hour_index).add_prefix("old_")
    table = pd.concat([v3h, oldh.drop(columns=["old_hour"])], axis=1)
    table = table.rename(columns={"v3_hour": "hour"})
    out_dir = C.OUTPUTS_DIR / "analysis" / "fast_v3" / "unicorn_hourly"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "unicorn_hourly_compare_last48.csv"
    table.to_csv(out, index=False)

    print("\nHOURLY")
    display = table.copy()
    display["hour"] = pd.to_datetime(display["hour"], utc=True).dt.strftime("%m-%d %H:%M")
    print(display.to_string(index=False, formatters={
        "v3_win": lambda x: "" if pd.isna(x) else f"{x:.3f}",
        "v3_avg_pct": lambda x: "" if pd.isna(x) else f"{x:+.4f}",
        "v3_usd": "{:+.2f}".format,
        "v3_cum_usd": "{:+.2f}".format,
        "old_win": lambda x: "" if pd.isna(x) else f"{x:.3f}",
        "old_avg_pct": lambda x: "" if pd.isna(x) else f"{x:+.4f}",
        "old_usd": "{:+.2f}".format,
        "old_cum_usd": "{:+.2f}".format,
    }))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
