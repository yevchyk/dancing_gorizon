"""Simulate the fast_v3 spread20 engine on scored rows.

Rule:
  LONG  when p_up_20m - p_down_20m >= threshold
  SHORT when p_down_20m - p_up_20m >= threshold
  close after 20m, net of the fast eval cost.

Example:
  python -m src.run_spread20_engine_sim --score-file outputs/analysis/fast_v3/holdout_scores_wf_spread20_last7_final.parquet --rolling-days 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .fast import config as FC

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load_scores(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], utc=True)
        frame["day"] = frame["anchor_time"].dt.strftime("%Y-%m-%d")
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True).sort_values("anchor_time")
    return scores.drop_duplicates(["symbol", "anchor_time"], keep="last").reset_index(drop=True)


def _pick_trades(
    scores: pd.DataFrame,
    *,
    threshold: float,
    top_per_anchor: int,
) -> pd.DataFrame:
    spread = scores["p_up_20m"].to_numpy("float64") - scores["p_down_20m"].to_numpy("float64")
    fired = np.abs(spread) >= threshold
    trades = scores.loc[fired, ["symbol", "anchor_time", "day", "p_up_20m", "p_down_20m", "real_ret_20m"]].copy()
    trades["spread"] = spread[fired]
    trades["side"] = np.where(trades["spread"] > 0, "long", "short")
    trades["score"] = trades["spread"].abs()
    if top_per_anchor > 0 and not trades.empty:
        trades = (
            trades.sort_values(["anchor_time", "score"], ascending=[True, False])
            .groupby("anchor_time", sort=False)
            .head(top_per_anchor)
            .sort_values("anchor_time")
            .reset_index(drop=True)
        )
    sign = np.where(trades["side"].to_numpy() == "long", 1.0, -1.0)
    trades["pnl_pct"] = (sign * trades["real_ret_20m"].to_numpy("float64") - FC.EVAL_COST) * 100.0
    return trades


def _print_summary(trades: pd.DataFrame, *, days: float, notional: float) -> None:
    if trades.empty:
        print("no trades")
        return
    usd = trades["pnl_pct"].to_numpy("float64") / 100.0 * notional
    print(
        f"signals={len(trades)}  signals/day={len(trades)/days:.1f}  "
        f"win={(trades.pnl_pct > 0).mean():.3f}  avg%={trades.pnl_pct.mean():+.4f}  "
        f"total%={trades.pnl_pct.sum():+.2f}  total$={usd.sum():+.2f}  $/day={usd.sum()/days:+.2f}"
    )
    print(
        f"long={int((trades.side == 'long').sum())}  "
        f"short={int((trades.side == 'short').sum())}  "
        f"symbols={trades.symbol.nunique()}  "
        f"spread_min={trades.score.min():.4f}  spread_med={trades.score.median():.4f}  "
        f"spread_max={trades.score.max():.4f}"
    )


def _breakdowns(trades: pd.DataFrame, notional: float) -> None:
    if trades.empty:
        return
    trades = trades.copy()
    trades["usd"] = trades["pnl_pct"] / 100.0 * notional
    print(f"\n{'day':<12}{'n':>7}{'long':>7}{'short':>7}{'win':>7}{'avg%':>10}{'total$':>10}")
    for day, group in trades.groupby("day", sort=True):
        print(
            f"{day:<12}{len(group):>7}{int((group.side == 'long').sum()):>7}"
            f"{int((group.side == 'short').sum()):>7}{(group.pnl_pct > 0).mean():>7.3f}"
            f"{group.pnl_pct.mean():>+10.4f}{group.usd.sum():>+10.2f}"
        )
    by_symbol = (
        trades.groupby("symbol")
        .agg(n=("pnl_pct", "size"), avg_pct=("pnl_pct", "mean"), total_usd=("usd", "sum"))
        .sort_values("total_usd", ascending=False)
    )
    print("\nTop symbols:")
    print(by_symbol.head(12).to_string(formatters={"avg_pct": "{:+.4f}".format, "total_usd": "{:+.2f}".format}))
    print("\nWorst symbols:")
    print(by_symbol.tail(12).sort_values("total_usd").to_string(formatters={"avg_pct": "{:+.4f}".format, "total_usd": "{:+.2f}".format}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-file", action="append", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.38)
    ap.add_argument("--rolling-days", type=float, default=7.0)
    ap.add_argument("--size", type=float, default=10.0)
    ap.add_argument("--leverage", type=float, default=3.0)
    ap.add_argument("--top-per-anchor", type=int, default=0, help="0 means trade every fired symbol")
    ap.add_argument("--out", type=Path, default=Path("outputs/analysis/fast_v3/anti_signal_sweep/spread20_engine_last7_trades.csv"))
    args = ap.parse_args()

    scores = _load_scores(args.score_file)
    end = scores["anchor_time"].max()
    start = end - pd.Timedelta(days=float(args.rolling_days))
    scores = scores[(scores["anchor_time"] >= start) & (scores["anchor_time"] <= end)].copy()
    elapsed_days = max((end - start).total_seconds() / 86400.0, 1e-9)
    notional = float(args.size) * float(args.leverage)
    print(
        f"spread20 engine: threshold={args.threshold:.2f}  "
        f"notional=${notional:.2f} (${args.size:.2f} x {args.leverage:.1f}x)  "
        f"cost={FC.EVAL_COST*100:.2f}%"
    )
    print(f"window {start} -> {end}  rows={len(scores)}  symbols={scores.symbol.nunique()}")
    if args.top_per_anchor > 0:
        print(f"cap: top {args.top_per_anchor} fired symbols per anchor")
    else:
        print("cap: all fired symbols")
    trades = _pick_trades(scores, threshold=float(args.threshold), top_per_anchor=int(args.top_per_anchor))
    _print_summary(trades, days=elapsed_days, notional=notional)
    _breakdowns(trades, notional)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.out, index=False)
    print(f"\ntrades -> {args.out}")


if __name__ == "__main__":
    main()
