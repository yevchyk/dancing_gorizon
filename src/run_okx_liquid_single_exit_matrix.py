"""Single-model exit matrix for okx_liquid.

Scores up_2m/up_8m/up_10m on the isolated okx_liquid store, then compares each
model at one probability threshold across 2m/8m/10m exits. Read-only simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .fast import config as FC
from .run_okx_liquid_sim import OkxLiquidPriceBook
from .run_test_engine_harvest_sim import simulate_engine
from .trading.fast_combo_engine import FastComboEngine
from .trading.timeutil import index_to_ns


OKX_DIR = Path("data/okx_liquid/candles_mixed")
NS = 60_000_000_000
EVAL = FC.EVAL_COST
MODELS = ("up_2m", "up_8m", "up_10m")
EXITS = {"2m": 2, "8m": 8, "10m": 10}


def fwd_ret(ts: np.ndarray, close: np.ndarray, anchors: np.ndarray, minutes: int) -> np.ndarray:
    ei = np.searchsorted(ts, anchors, "right") - 1
    xj = np.searchsorted(ts, anchors + minutes * NS, "right") - 1
    ok = (ei >= 0) & (xj > ei)
    out = np.full(len(anchors), np.nan, dtype="float64")
    entry = close[np.clip(ei, 0, len(close) - 1)]
    exitp = close[np.clip(xj, 0, len(close) - 1)]
    ok &= np.isfinite(entry) & np.isfinite(exitp) & (entry > 0)
    out[ok] = exitp[ok] / entry[ok] - 1.0
    return out


def build_frame(days: float) -> tuple[pd.DataFrame, float]:
    eng = FastComboEngine("pulse00")
    symbols = sorted(p.stem for p in OKX_DIR.glob("*.parquet"))
    maxes: list[pd.Timestamp] = []
    for sym in symbols:
        try:
            df = pd.read_parquet(OKX_DIR / f"{sym}.parquet", columns=["timestamp"])
            ts = pd.to_datetime(df["timestamp"], utc=True)
            if len(ts):
                maxes.append(ts.max())
        except Exception:
            continue
    if not maxes:
        raise SystemExit("no okx_liquid candles")

    end = min(maxes) - pd.Timedelta(minutes=max(EXITS.values()))
    end = pd.Timestamp(end).floor("1min")
    start = end - pd.Timedelta(days=days)
    anchors = pd.date_range(start.ceil("2min"), end.floor("2min"), freq="2min")
    ans = anchors.as_unit("ns").asi8
    window_days = max(1e-9, (end - start).total_seconds() / 86400.0)
    print(f"window {start} -> {end} UTC anchors={len(anchors)} symbols={len(symbols)}")

    frames: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            df = pd.read_parquet(OKX_DIR / f"{sym}.parquet")
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            df = df.sort_index()
            if len(df) < 300:
                continue
        except Exception:
            continue

        ts = index_to_ns(df.index)
        close = df["close"].to_numpy("float64")
        feats, valid = eng.curve.build_matrix(ts, close, ans)
        if valid.sum() == 0:
            continue

        idx = np.where(valid)[0]
        x = pd.DataFrame(feats[idx], columns=eng.columns)
        row = {
            "symbol": sym,
            "anchor_time": anchors[idx],
        }
        for model_name in MODELS:
            model, cols = eng._models[model_name]
            row[f"p_{model_name}"] = model.predict_proba(x[cols])[:, 1].astype("float32")
        for exit_label, minutes in EXITS.items():
            row[f"r_{exit_label}"] = fwd_ret(ts, close, ans[idx], minutes).astype("float32")
        frames.append(pd.DataFrame(row))

    if not frames:
        raise SystemExit("no scored rows")
    d = pd.concat(frames, ignore_index=True)
    d["anchor_time"] = pd.to_datetime(d["anchor_time"], utc=True)
    print(f"scored rows={len(d)}")
    return d, window_days


def make_engine_input(d: pd.DataFrame, model_name: str, exit_label: str, threshold: float) -> pd.DataFrame:
    pcol = f"p_{model_name}"
    mask = d[pcol].to_numpy("float64") > threshold
    x = d.loc[mask, ["symbol", "anchor_time", pcol]].copy()
    if x.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "engine": f"{model_name}_gt{threshold:.2f}_exit{exit_label}",
        "family": "okx_liquid_single",
        "source": model_name,
        "signal_model": f"{model_name}>{threshold:.2f}",
        "symbol": x["symbol"].to_numpy(),
        "anchor_time": x["anchor_time"].to_numpy(),
        "day": pd.to_datetime(x["anchor_time"], utc=True).dt.strftime("%m-%d").to_numpy(),
        "side": 1,
        "exit": exit_label,
        "threshold": threshold,
        "leverage": 1.0,
        "score": x[pcol].to_numpy("float64"),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--threshold", type=float, default=0.93)
    ap.add_argument("--threshold-up2", type=float)
    ap.add_argument("--threshold-up8", type=float)
    ap.add_argument("--threshold-up10", type=float)
    ap.add_argument("--top-per-scan", type=int, default=3)
    ap.add_argument("--max-open", type=int, default=3)
    ap.add_argument("--cooldown-min", type=int, default=10)
    args = ap.parse_args()

    d, window_days = build_frame(args.days)
    book = OkxLiquidPriceBook()
    rows: list[dict] = []
    thresholds = {
        "up_2m": args.threshold_up2 if args.threshold_up2 is not None else args.threshold,
        "up_8m": args.threshold_up8 if args.threshold_up8 is not None else args.threshold,
        "up_10m": args.threshold_up10 if args.threshold_up10 is not None else args.threshold,
    }

    for model_name in MODELS:
        threshold = float(thresholds[model_name])
        pcol = f"p_{model_name}"
        base_mask = d[pcol].to_numpy("float64") > threshold
        for exit_label in EXITS:
            rcol = f"r_{exit_label}"
            mask = base_mask & np.isfinite(d[rcol].to_numpy("float64"))
            pnl = d.loc[mask, rcol].to_numpy("float64") - EVAL
            cand = make_engine_input(d, model_name, exit_label, threshold)
            if cand.empty:
                trades = pd.DataFrame()
                blocks = {}
            else:
                scan_times = sorted(pd.Timestamp(t) for t in cand["anchor_time"].drop_duplicates())
                trades, blocks = simulate_engine(
                    cand["engine"].iloc[0],
                    cand,
                    scan_times,
                    book,
                    harvest=False,
                    top_per_scan=args.top_per_scan,
                    max_open=args.max_open,
                    cooldown_min=args.cooldown_min,
                )

            rows.append({
                "model": model_name,
                "thr": threshold,
                "exit": exit_label,
                "raw": int(mask.sum()),
                "raw/day": float(mask.sum() / window_days),
                "sig_win": float((pnl > 0).mean()) if len(pnl) else np.nan,
                "sig_avg%": float(pnl.mean() * 100.0) if len(pnl) else np.nan,
                "sig_total%": float(pnl.sum() * 100.0) if len(pnl) else 0.0,
                "trades": int(len(trades)),
                "trades/day": float(len(trades) / window_days),
                "sim_win": float(trades["won"].mean()) if len(trades) else np.nan,
                "sim_avg%": float(trades["net_pnl_pct"].mean()) if len(trades) else np.nan,
                "sim_total%": float(trades["net_pnl_pct"].sum()) if len(trades) else 0.0,
                "blocked_open": int(blocks.get("block_max_open", 0)),
                "blocked_cooldown": int(blocks.get("block_cooldown", 0)),
            })

    out = pd.DataFrame(rows)
    print(
        "\nSINGLE MODEL EXIT MATRIX: "
        f"up2>{thresholds['up_2m']:.2f}, "
        f"up8>{thresholds['up_8m']:.2f}, "
        f"up10>{thresholds['up_10m']:.2f}, "
        f"top={args.top_per_scan}, max_open={args.max_open}, cooldown={args.cooldown_min}m"
    )
    print(out.to_string(index=False, formatters={
        "thr": "{:.2f}".format,
        "raw/day": "{:.1f}".format,
        "sig_win": "{:.3f}".format,
        "sig_avg%": "{:+.4f}".format,
        "sig_total%": "{:+.2f}".format,
        "trades/day": "{:.1f}".format,
        "sim_win": "{:.3f}".format,
        "sim_avg%": "{:+.4f}".format,
        "sim_total%": "{:+.2f}".format,
    }))

    best = out.sort_values(["sim_total%", "sim_avg%"], ascending=[False, False]).head(5)
    print("\nBEST BY LIVE-LIKE TOTAL")
    print(best[["model", "thr", "exit", "raw/day", "trades/day", "sim_win", "sim_avg%", "sim_total%"]].to_string(
        index=False,
        formatters={
            "thr": "{:.2f}".format,
            "raw/day": "{:.1f}".format,
            "trades/day": "{:.1f}".format,
            "sim_win": "{:.3f}".format,
            "sim_avg%": "{:+.4f}".format,
            "sim_total%": "{:+.2f}".format,
        },
    ))


if __name__ == "__main__":
    main()
