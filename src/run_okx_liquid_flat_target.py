"""Tune single-model UP flows on okx_liquid to a target raw signal count.

Models compared:
  - up_2m, exit 8m
  - up_8m, exit 8m
  - up_10m, exit 10m

The grid first finds probability triplets near a target raw candidate rate, then
runs the normal live-like simulator for the best candidates. This is a read-only
simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .fast import config as FC
from .trading.fast_combo_engine import FastComboEngine
from .trading.timeutil import index_to_ns
from .run_okx_liquid_sim import OkxLiquidPriceBook
from .run_test_engine_harvest_sim import simulate_engine


OKX_DIR = Path("data/okx_liquid/candles_mixed")
NS = 60_000_000_000
EVAL = FC.EVAL_COST


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
    frames: list[pd.DataFrame] = []

    maxes = []
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

    end = min(maxes) - pd.Timedelta(minutes=10)
    end = pd.Timestamp(end).floor("1min")
    start = end - pd.Timedelta(days=days)
    anchors = pd.date_range(start.ceil("2min"), end.floor("2min"), freq="2min")
    ans = anchors.as_unit("ns").asi8
    window_days = max(1e-9, (end - start).total_seconds() / 86400.0)
    print(f"window {start} -> {end} UTC anchors={len(anchors)} symbols={len(symbols)}")

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
        X = pd.DataFrame(feats[idx], columns=eng.columns)
        p2 = eng._models["up_2m"][0].predict_proba(X[eng._models["up_2m"][1]])[:, 1]
        p8 = eng._models["up_8m"][0].predict_proba(X[eng._models["up_8m"][1]])[:, 1]
        p10 = eng._models["up_10m"][0].predict_proba(X[eng._models["up_10m"][1]])[:, 1]
        a = ans[idx]
        frames.append(pd.DataFrame({
            "symbol": sym,
            "anchor_time": anchors[idx],
            "p2": p2.astype("float32"),
            "p8": p8.astype("float32"),
            "p10": p10.astype("float32"),
            "r8": fwd_ret(ts, close, a, 8).astype("float32"),
            "r10": fwd_ret(ts, close, a, 10).astype("float32"),
        }))

    if not frames:
        raise SystemExit("no scored rows")
    d = pd.concat(frames, ignore_index=True)
    d["anchor_time"] = pd.to_datetime(d["anchor_time"], utc=True)
    print(f"scored rows={len(d)}")
    return d, window_days


def choice_for(d: pd.DataFrame, t2: float, t8: float, t10: float) -> pd.DataFrame:
    p2 = d["p2"].to_numpy("float64")
    p8 = d["p8"].to_numpy("float64")
    p10 = d["p10"].to_numpy("float64")
    s2 = np.where(p2 >= t2, p2, -1.0)
    s8 = np.where(p8 >= t8, p8, -1.0)
    s10 = np.where(p10 >= t10, p10, -1.0)
    scores = np.vstack([s2, s8, s10])
    pick = scores.argmax(axis=0)
    best = scores[pick, np.arange(len(d))]
    mask = best >= 0
    x = d.loc[mask, ["symbol", "anchor_time", "r8", "r10"]].copy()
    pick_m = pick[mask]
    best_m = best[mask]
    x["model"] = np.where(pick_m == 0, "up_2m", np.where(pick_m == 1, "up_8m", "up_10m"))
    x["exit"] = np.where(pick_m == 2, "10m", "8m")
    x["score"] = best_m
    x["ret"] = np.where(pick_m == 2, x["r10"].to_numpy("float64"), x["r8"].to_numpy("float64"))
    x = x[np.isfinite(x["ret"])].copy()
    x["pnl"] = x["ret"] - EVAL
    return x


def as_engine_input(x: pd.DataFrame, name: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "engine": name,
        "family": "okx_liquid_flat",
        "source": x["model"].to_numpy(),
        "signal_model": x["model"].to_numpy(),
        "symbol": x["symbol"].to_numpy(),
        "anchor_time": x["anchor_time"].to_numpy(),
        "day": pd.to_datetime(x["anchor_time"], utc=True).dt.strftime("%m-%d").to_numpy(),
        "side": 1,
        "exit": x["exit"].to_numpy(),
        "threshold": np.nan,
        "leverage": 1.0,
        "score": x["score"].to_numpy("float64"),
    })
    return out


def trade_stats(trades: pd.DataFrame, window_days: float) -> dict:
    if trades.empty:
        return {"trades": 0, "trades/day": 0.0, "win": np.nan, "avg%": np.nan, "total%": 0.0}
    return {
        "trades": int(len(trades)),
        "trades/day": float(len(trades) / window_days),
        "win": float(trades["won"].mean()),
        "avg%": float(trades["net_pnl_pct"].mean()),
        "total%": float(trades["net_pnl_pct"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=10.0)
    ap.add_argument("--target-per-day", type=float, default=240.0)
    ap.add_argument("--band", type=float, default=35.0)
    ap.add_argument("--top-per-scan", type=int, default=3)
    ap.add_argument("--max-open", type=int, default=3)
    ap.add_argument("--cooldown-min", type=int, default=10)
    ap.add_argument("--simulate-top", type=int, default=25)
    args = ap.parse_args()

    d, window_days = build_frame(args.days)
    grid2 = np.round(np.arange(0.85, 0.971, 0.01), 2)
    grid8 = np.round(np.arange(0.80, 0.931, 0.01), 2)
    grid10 = np.round(np.arange(0.80, 0.931, 0.01), 2)

    rows = []
    p2 = d["p2"].to_numpy("float64")
    p8 = d["p8"].to_numpy("float64")
    p10 = d["p10"].to_numpy("float64")
    r8 = d["r8"].to_numpy("float64")
    r10 = d["r10"].to_numpy("float64")

    for t2 in grid2:
        m2 = p2 >= t2
        for t8 in grid8:
            m8 = p8 >= t8
            for t10 in grid10:
                m10 = p10 >= t10
                mask = m2 | m8 | m10
                n = int(mask.sum())
                raw_day = n / window_days
                if abs(raw_day - args.target_per_day) > args.band:
                    continue
                s2 = np.where(m2, p2, -1.0)
                s8 = np.where(m8, p8, -1.0)
                s10 = np.where(m10, p10, -1.0)
                scores = np.vstack([s2, s8, s10])
                pick = scores.argmax(axis=0)
                ret = np.where(pick == 2, r10, r8)
                ok = mask & np.isfinite(ret)
                pnl = ret[ok] - EVAL
                if len(pnl) == 0:
                    continue
                rows.append({
                    "t2": float(t2), "t8": float(t8), "t10": float(t10),
                    "raw": int(ok.sum()), "raw/day": float(ok.sum() / window_days),
                    "sig_win": float((pnl > 0).mean()),
                    "sig_avg%": float(pnl.mean() * 100),
                    "sig_total%": float(pnl.sum() * 100),
                })

    if not rows:
        print("no threshold triplets in target band")
        return

    cand = pd.DataFrame(rows)
    cand["target_dist"] = (cand["raw/day"] - args.target_per_day).abs()
    cand = cand.sort_values(["sig_total%", "sig_avg%", "target_dist"], ascending=[False, False, True])
    print("\nBEST SIGNAL-LEVEL TRIPLETS NEAR TARGET")
    print(cand.head(15).to_string(index=False, formatters={
        "t2": "{:.2f}".format, "t8": "{:.2f}".format, "t10": "{:.2f}".format,
        "raw/day": "{:.1f}".format, "sig_win": "{:.3f}".format,
        "sig_avg%": "{:+.4f}".format, "sig_total%": "{:+.1f}".format,
        "target_dist": "{:.1f}".format,
    }))

    book = OkxLiquidPriceBook()
    sim_rows = []
    seen = set()
    for r in cand.head(args.simulate_top).itertuples(index=False):
        key = (float(r.t2), float(r.t8), float(r.t10))
        if key in seen:
            continue
        seen.add(key)
        x = choice_for(d, *key)
        inp = as_engine_input(x, f"flat_t2{key[0]:.2f}_t8{key[1]:.2f}_t10{key[2]:.2f}")
        scan_times = sorted(pd.Timestamp(t) for t in inp["anchor_time"].drop_duplicates())
        trades, blocks = simulate_engine(
            inp["engine"].iloc[0], inp, scan_times, book,
            harvest=False,
            top_per_scan=args.top_per_scan,
            max_open=args.max_open,
            cooldown_min=args.cooldown_min,
        )
        st = trade_stats(trades, window_days)
        mix = x["model"].value_counts().to_dict()
        sim_rows.append({
            "t2": key[0], "t8": key[1], "t10": key[2],
            "raw/day": len(x) / window_days,
            **st,
            "up2_raw": int(mix.get("up_2m", 0)),
            "up8_raw": int(mix.get("up_8m", 0)),
            "up10_raw": int(mix.get("up_10m", 0)),
            "blocked_open": blocks.get("block_max_open", 0),
            "blocked_cooldown": blocks.get("block_cooldown", 0),
        })

    sim = pd.DataFrame(sim_rows).sort_values(["total%", "avg%"], ascending=[False, False])
    print("\nBEST LIVE-LIKE SIMS")
    print(sim.head(15).to_string(index=False, formatters={
        "t2": "{:.2f}".format, "t8": "{:.2f}".format, "t10": "{:.2f}".format,
        "raw/day": "{:.1f}".format, "trades/day": "{:.1f}".format,
        "win": "{:.3f}".format, "avg%": "{:+.4f}".format, "total%": "{:+.2f}".format,
    }))


if __name__ == "__main__":
    main()
