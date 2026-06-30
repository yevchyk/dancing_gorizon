"""Hourly last-day comparison on the isolated okx_liquid store.

Compares:
  1) Only Forward: long up_2m >= 0.93, exit 8m
  2) long up_2m >= 0.95, exit 8m
  3) long up_10m >= 0.90, exit 10m
  4) Unicorn/Pulse3, exit 8m

This is a simulation only. It does not touch live trading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .fast import config as FC
from .trading.fast_combo_engine import FastComboEngine, WORTHY
from .trading.timeutil import index_to_ns
from .run_okx_liquid_sim import OkxLiquidPriceBook
from .run_test_engine_harvest_sim import simulate_engine


OKX_DIR = Path("data/okx_liquid/candles_mixed")
MAX_EXIT_MIN = 10


def load_store(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = OKX_DIR / f"{sym}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            df = df.sort_index()
            if len(df) >= 300 and not df.empty:
                out[sym] = df
        except Exception:
            continue
    return out


def common_window(data: dict[str, pd.DataFrame], days: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    maxes = [df.index.max() for df in data.values() if df is not None and not df.empty]
    if not maxes:
        raise SystemExit("no candle data")
    end = min(maxes) - pd.Timedelta(minutes=MAX_EXIT_MIN)
    end = pd.Timestamp(end).floor("1min")
    start = end - pd.Timedelta(days=days)
    return start, end


def engine_input(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["day"] = pd.to_datetime(d["anchor_time"], utc=True).dt.strftime("%m-%d")
    d["family"] = "okx_liquid"
    d["source"] = d["engine"]
    d["threshold"] = np.nan
    d["leverage"] = 1.0
    return d[[
        "engine", "family", "source", "signal_model", "symbol", "anchor_time",
        "day", "side", "exit", "threshold", "leverage", "score",
    ]]


def build_candidates(
    eng: FastComboEngine,
    data: dict[str, pd.DataFrame],
    anchors: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    ans = anchors.as_unit("ns").asi8
    rows_only_forward: list[dict] = []
    rows_up2: list[dict] = []
    rows_up10: list[dict] = []
    rows_unicorn: list[dict] = []

    for sym, df in data.items():
        ts = index_to_ns(df.index)
        close = df["close"].to_numpy("float64")
        feats, valid = eng.curve.build_matrix(ts, close, ans)
        if valid.sum() == 0:
            continue

        idx = np.where(valid)[0]
        X = pd.DataFrame(feats[idx], columns=eng.columns)
        probs: dict[str, np.ndarray] = {}
        for model_name, (model, cols) in eng._models.items():
            probs[model_name] = model.predict_proba(X[cols])[:, 1]

        up_count = np.zeros(len(idx), dtype=int)
        down_count = np.zeros(len(idx), dtype=int)
        up_score = np.zeros(len(idx), dtype="float64")
        down_score = np.zeros(len(idx), dtype="float64")
        up_max = np.zeros(len(idx), dtype="float64")
        down_max = np.zeros(len(idx), dtype="float64")

        for _full, (model_name, _side_name, side, base) in WORTHY.items():
            p = probs[model_name]
            active = p >= base
            headroom = np.clip((p - base) / max(1e-9, 1.0 - base), 0, None)
            if side > 0:
                up_count += active.astype(int)
                up_score += np.where(active, headroom, 0.0)
                up_max = np.maximum(up_max, np.where(active, p, 0.0))
            else:
                down_count += active.astype(int)
                down_score += np.where(active, headroom, 0.0)
                down_max = np.maximum(down_max, np.where(active, p, 0.0))

        for j, anchor_pos in enumerate(idx):
            anchor = anchors[anchor_pos]
            p_up2 = float(probs["up_2m"][j])
            if p_up2 >= 0.93:
                rows_only_forward.append({
                    "engine": "only_forward",
                    "signal_model": "OnlyForward_up2m_p093_exit8m",
                    "symbol": sym,
                    "anchor_time": anchor,
                    "side": 1,
                    "exit": "8m",
                    "score": p_up2,
                })
            if p_up2 >= 0.95:
                rows_up2.append({
                    "engine": "up2_095_exit8",
                    "signal_model": "up_2m>=0.95",
                    "symbol": sym,
                    "anchor_time": anchor,
                    "side": 1,
                    "exit": "8m",
                    "score": p_up2,
                })

            p_up10 = float(probs["up_10m"][j])
            if p_up10 >= 0.90:
                rows_up10.append({
                    "engine": "up10_090_exit10",
                    "signal_model": "up_10m>=0.90",
                    "symbol": sym,
                    "anchor_time": anchor,
                    "side": 1,
                    "exit": "10m",
                    "score": p_up10,
                })

            long_ok = up_count[j] >= 3 and down_count[j] == 0
            short_ok = down_count[j] >= 3 and up_count[j] == 0
            if long_ok or short_ok:
                rows_unicorn.append({
                    "engine": "unicorn_p3_exit8",
                    "signal_model": "Pulse3_exit8",
                    "symbol": sym,
                    "anchor_time": anchor,
                    "side": 1 if long_ok else -1,
                    "exit": "8m",
                    "score": 100.0 + (up_score[j] if long_ok else down_score[j]),
                })

    return {
        "Only Forward @8m": engine_input(rows_only_forward),
        "up_2m>=0.95 @8m": engine_input(rows_up2),
        "up_10m>=0.90 @10m": engine_input(rows_up10),
        "Unicorn Pulse3 @8m": engine_input(rows_unicorn),
    }


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "win": np.nan, "avg%": np.nan, "total%": 0.0, "long%": np.nan}
    return {
        "trades": int(len(trades)),
        "win": float(trades["won"].mean()),
        "avg%": float(trades["net_pnl_pct"].mean()),
        "total%": float(trades["net_pnl_pct"].sum()),
        "long%": float((trades["side"] == "long").mean()),
    }


def hourly(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    x = trades.copy()
    t = pd.to_datetime(x["opened_at"], utc=True)
    x["utc"] = t.dt.hour
    x["kyiv"] = (x["utc"] + 3) % 24
    for (utc, kyiv), g in x.groupby(["utc", "kyiv"], sort=True):
        s = summarize(g)
        rows.append({"engine": label, "utc": int(utc), "kyiv": int(kyiv), **s})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--top-per-scan", type=int, default=3)
    ap.add_argument("--max-open", type=int, default=5)
    ap.add_argument("--cooldown-min", type=int, default=10)
    ap.add_argument("--apply-blacklist", action="store_true")
    args = ap.parse_args()

    symbols = sorted(p.stem for p in OKX_DIR.glob("*.parquet"))
    if args.apply_blacklist:
        symbols = [s for s in symbols if s not in set(C.BLACKLIST_SYMBOLS)]

    data = load_store(symbols)
    start, end = common_window(data, args.days)
    anchors = pd.date_range(start.ceil("2min"), end.floor("2min"), freq="2min")
    print(
        f"okx_liquid hourly compare: symbols={len(data)} days={args.days:g} "
        f"window={start} -> {end} UTC anchors={len(anchors)}"
    )
    print(
        f"sim params: top={args.top_per_scan} max_open={args.max_open} "
        f"cooldown={args.cooldown_min}m blacklist={args.apply_blacklist}"
    )

    eng = FastComboEngine("pulse00")
    cands = build_candidates(eng, data, anchors)
    book = OkxLiquidPriceBook()

    all_hourly = []
    print("\nOVERALL")
    for label, cand in cands.items():
        if cand.empty:
            trades = pd.DataFrame()
            blocks = {}
        else:
            scan_times = sorted(pd.Timestamp(t) for t in cand["anchor_time"].drop_duplicates())
            trades, blocks = simulate_engine(
                label,
                cand,
                scan_times,
                book,
                harvest=False,
                top_per_scan=args.top_per_scan,
                max_open=args.max_open,
                cooldown_min=args.cooldown_min,
            )
        s = summarize(trades)
        print(
            f"{label:22s} cands={len(cand):5d} trades={s['trades']:4d} "
            f"win={s['win']:.3f} avg={s['avg%']:+.4f}% total={s['total%']:+.2f}% "
            f"long={s['long%']:.0%}"
        )
        h = hourly(trades, label)
        if not h.empty:
            all_hourly.append(h)

    if not all_hourly:
        return
    out = pd.concat(all_hourly, ignore_index=True)
    print("\nHOURLY")
    for label in cands:
        d = out[out["engine"] == label]
        if d.empty:
            continue
        print(f"\n{label}")
        print(d[["utc", "kyiv", "trades", "win", "avg%", "total%", "long%"]].to_string(
            index=False,
            formatters={
                "win": "{:.3f}".format,
                "avg%": "{:+.4f}".format,
                "total%": "{:+.2f}".format,
                "long%": "{:.0%}".format,
            },
        ))


if __name__ == "__main__":
    main()
