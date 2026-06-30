"""Replay HC per-horizon threshold signals over a recent live-like window.

This sidecar uses per-horizon probability floors produced by
run_hc_horizon_threshold_optimizer, then re-scores a requested mature window and
simulates independent fixed-horizon trades.  It is not a live order runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .fast import config as FC
from .hc import config as HC
from .hc.data import read_json_symbols
from .markets import get
from .run_hc_horizon_threshold_optimizer import attach_exact_outcomes
from .run_hc_offgrid_sim import build_feature_rows, score_ensemble


OUT_DIR = C.OUTPUTS_DIR / "analysis" / "hc_threshold_live_sim"


def load_thresholds(path: Path) -> dict[int, float]:
    df = pd.read_csv(path)
    if "horizon" not in df.columns or "threshold" not in df.columns:
        raise ValueError(f"{path} must have horizon,threshold columns")
    out = {
        int(r.horizon): float(r.threshold)
        for r in df.itertuples(index=False)
        if np.isfinite(float(r.threshold)) and float(r.threshold) <= 1.0
    }
    if not out:
        raise ValueError(f"No usable thresholds in {path}")
    return dict(sorted(out.items()))


def latest_common_price_time(symbols: list[str]) -> pd.Timestamp:
    latest: list[pd.Timestamp] = []
    for sym in symbols:
        df = get(HC.STORE_KEY).load(sym)
        if df is None or df.empty:
            continue
        latest.append(pd.Timestamp(df.index.max()).tz_convert("UTC"))
    if not latest:
        raise RuntimeError("No candle data found for threshold live sim")
    return min(latest)


def mature_entry_grid(
    *,
    symbols: list[str],
    max_horizon: int,
    hours: float,
    scan_stride_min: int,
) -> pd.DatetimeIndex:
    latest = latest_common_price_time(symbols)
    end = (latest - pd.Timedelta(minutes=int(max_horizon))).floor(f"{int(scan_stride_min)}min")
    count = int(round(float(hours) * 60.0 / int(scan_stride_min)))
    start = end - pd.Timedelta(minutes=int(scan_stride_min) * (count - 1))
    return pd.date_range(start, end, freq=f"{int(scan_stride_min)}min", tz="UTC")


def make_threshold_candidates(
    scored: pd.DataFrame,
    thresholds: dict[int, float],
    *,
    opp_cap: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    base = scored[scored["horizon_minutes"].isin(thresholds)].copy()
    for side_name, side_int, prob_col, opp_col in (
        ("long", 1, "up_prob", "down_prob"),
        ("short", -1, "down_prob", "up_prob"),
    ):
        d = base[["symbol", "base_time", "horizon_minutes", prob_col, opp_col]].copy()
        if d.empty:
            continue
        d["threshold"] = d["horizon_minutes"].astype(int).map(thresholds).astype("float64")
        mask = d[prob_col].astype("float64").ge(d["threshold"]) & d[opp_col].astype("float64").le(float(opp_cap))
        d = d.loc[mask].copy()
        if d.empty:
            continue
        d["side"] = side_int
        d["side_name"] = side_name
        d["p_dir"] = d[prob_col].astype("float64")
        d["p_opp"] = d[opp_col].astype("float64")
        d["score"] = d["p_dir"] - d["p_opp"]
        d["over_threshold"] = d["p_dir"] - d["threshold"]
        d["day_key"] = pd.to_datetime(d["base_time"], utc=True).dt.strftime("%Y%m%d")
        parts.append(
            d[
                [
                    "day_key",
                    "symbol",
                    "base_time",
                    "horizon_minutes",
                    "side",
                    "side_name",
                    "p_dir",
                    "p_opp",
                    "threshold",
                    "over_threshold",
                    "score",
                ]
            ]
        )
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["anchor_time"] = pd.to_datetime(out["base_time"], utc=True) + pd.Timedelta(minutes=HC.EXEC_ENTRY_DELAY_MIN)
    return out.sort_values(["anchor_time", "score", "p_dir"], ascending=[True, False, False]).reset_index(drop=True)


def select_mode(
    candidates: pd.DataFrame,
    *,
    mode: str,
    top_n: int,
    strong_margin: float,
    global_max_per_scan: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    d = candidates.sort_values(["anchor_time", "score", "p_dir"], ascending=[True, False, False]).copy()
    if mode == "top3":
        rows = []
        for (_t, _sym), g in d.groupby(["anchor_time", "symbol"], sort=False):
            rows.append(g.head(int(top_n)).copy())
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    elif mode == "top4_strong":
        rows = []
        for (_t, _sym), g in d.groupby(["anchor_time", "symbol"], sort=False):
            head = g.head(int(top_n)).copy()
            rows.append(head)
            fourth = g.iloc[int(top_n): int(top_n) + 1]
            if not fourth.empty and float(fourth["over_threshold"].iloc[0]) >= float(strong_margin):
                rows.append(fourth.copy())
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    else:
        raise ValueError(f"unknown mode: {mode}")

    cap = int(global_max_per_scan)
    if cap > 0 and not out.empty:
        out = (
            out.sort_values(["anchor_time", "score", "p_dir"], ascending=[True, False, False])
            .groupby("anchor_time", sort=False, group_keys=False)
            .head(cap)
        )
    return out


def summarize_trades(label: str, selected: pd.DataFrame, *, stake_margin: float, leverage: float, initial_balance: float) -> tuple[pd.DataFrame, dict]:
    if selected.empty:
        return selected.copy(), {
            "mode": label,
            "signals": 0,
            "scans": 0,
            "signals_per_day": 0.0,
            "win_pct": np.nan,
            "avg_net_pct": np.nan,
            "avg_levered_pct": np.nan,
            "pnl_usd": 0.0,
            "final_balance": initial_balance,
            "max_drawdown_pct": 0.0,
            "max_open": 0,
        }
    out = selected.sort_values(["entry_time", "score"], ascending=[True, False]).copy()
    out["levered_pnl_pct"] = out["net_pnl_pct"].astype("float64") * float(leverage)
    out["pnl_usd"] = float(stake_margin) * out["levered_pnl_pct"] / 100.0
    out["balance_after"] = float(initial_balance) + out["pnl_usd"].cumsum()
    peaks = pd.concat(
        [pd.Series([float(initial_balance)]), out["balance_after"].reset_index(drop=True)],
        ignore_index=True,
    ).cummax().iloc[1:].to_numpy()
    dd = out["balance_after"].to_numpy("float64") - peaks

    events = []
    for r in out.itertuples(index=False):
        events.append((pd.Timestamp(r.entry_time), 1))
        events.append((pd.Timestamp(r.exit_time), -1))
    open_now = 0
    max_open = 0
    for _t, delta in sorted(events, key=lambda x: (x[0], x[1])):
        open_now += delta
        max_open = max(max_open, open_now)

    hours = max(1e-9, (out["entry_time"].max() - out["entry_time"].min()).total_seconds() / 3600.0)
    summary = {
        "mode": label,
        "signals": int(len(out)),
        "scans": int(out["anchor_time"].nunique()),
        "signals_per_day": float(len(out) / hours * 24.0),
        "win_pct": float(out["won"].mean() * 100.0),
        "avg_net_pct": float(out["net_pnl_pct"].mean()),
        "avg_levered_pct": float(out["levered_pnl_pct"].mean()),
        "pnl_usd": float(out["pnl_usd"].sum()),
        "final_balance": float(initial_balance + out["pnl_usd"].sum()),
        "max_drawdown_pct": float(dd.min() / float(initial_balance) * 100.0) if len(dd) else 0.0,
        "max_open": int(max_open),
    }
    return out, summary


def horizon_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = (
        trades.groupby("horizon_minutes")
        .agg(
            signals=("symbol", "size"),
            win_pct=("won", lambda s: float(s.mean() * 100.0)),
            avg_net_pct=("net_pnl_pct", "mean"),
            total_net_pct=("net_pnl_pct", "sum"),
            avg_prob=("p_dir", "mean"),
            symbols=("symbol", "nunique"),
        )
        .reset_index()
        .rename(columns={"horizon_minutes": "horizon"})
    )
    return out.sort_values(["signals", "horizon"], ascending=[False, True])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=Path, default=C.OUTPUTS_DIR / "analysis" / "hc_offgrid" / "threshold_optimizer_300pd_top50" / "per_horizon_thresholds.csv")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--scan-stride-min", type=int, default=5)
    ap.add_argument("--opp-cap", type=float, default=0.20)
    ap.add_argument("--stake-margin", type=float, default=10.0)
    ap.add_argument("--leverage", type=float, default=4.0)
    ap.add_argument("--initial-balance", type=float, default=100.0)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--modes", default="top3,top4_strong")
    ap.add_argument("--strong-margin", type=float, default=0.02)
    ap.add_argument("--global-max-per-scan", type=int, default=0, help="0 = no market-wide scan cap")
    ap.add_argument("--price-max-stale-min", type=float, default=2.0)
    ap.add_argument("--model-dir", type=Path, default=Path("models/hc_exec_stride120_nonoverlap"))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    thresholds = load_thresholds(args.thresholds)
    symbols = sorted(set(read_json_symbols()) - C.hc_blacklist_symbols())
    entries = mature_entry_grid(
        symbols=symbols,
        max_horizon=max(thresholds),
        hours=args.hours,
        scan_stride_min=args.scan_stride_min,
    )
    print(f"window entries={len(entries)} {entries.min()} -> {entries.max()}", flush=True)
    print(f"horizons={len(thresholds)} min={min(thresholds)} max={max(thresholds)}", flush=True)

    features = build_feature_rows(
        symbols=symbols,
        entries=entries,
        horizons=tuple(thresholds.keys()),
        entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN,
    )
    if features.empty:
        raise RuntimeError("No feature rows built")
    scored = score_ensemble(features, args.model_dir)
    candidates = make_threshold_candidates(scored, thresholds, opp_cap=args.opp_cap)
    outcomes = attach_exact_outcomes(candidates, max_stale_min=args.price_max_stale_min)
    print(f"candidates={len(candidates)} outcomes={len(outcomes)} scans={outcomes['anchor_time'].nunique() if len(outcomes) else 0}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"last{int(args.hours)}h_m{int(args.stake_margin)}_x{int(args.leverage)}"
    scored.to_parquet(args.out_dir / f"{tag}_scored.parquet", index=False)
    outcomes.to_parquet(args.out_dir / f"{tag}_all_candidates.parquet", index=False)

    rows = []
    trade_frames = []
    modes = [m.strip() for m in args.modes.replace(";", ",").split(",") if m.strip()]
    for mode in modes:
        selected = select_mode(
            outcomes,
            mode=mode,
            top_n=args.top_n,
            strong_margin=args.strong_margin,
            global_max_per_scan=args.global_max_per_scan,
        )
        trades, summary = summarize_trades(
            mode,
            selected,
            stake_margin=args.stake_margin,
            leverage=args.leverage,
            initial_balance=args.initial_balance,
        )
        rows.append(summary)
        if len(trades):
            trades["mode"] = mode
            trade_frames.append(trades)
            trades.to_parquet(args.out_dir / f"{tag}_{mode}_trades.parquet", index=False)
            horizon_summary(trades).to_csv(args.out_dir / f"{tag}_{mode}_horizons.csv", index=False)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.out_dir / f"{tag}_summary.csv", index=False)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    print("\nSUMMARY")
    print(summary_df.to_string(index=False, formatters={
        "signals_per_day": "{:.1f}".format,
        "win_pct": "{:.1f}%".format,
        "avg_net_pct": "{:+.2f}%".format,
        "avg_levered_pct": "{:+.2f}%".format,
        "pnl_usd": "{:+.2f}$".format,
        "final_balance": "{:.2f}$".format,
        "max_drawdown_pct": "{:+.2f}%".format,
    }))
    if len(all_trades):
        for mode in modes:
            print(f"\nHORIZONS {mode}")
            ht = horizon_summary(all_trades[all_trades["mode"].eq(mode)])
            print(ht.to_string(index=False, formatters={
                "win_pct": "{:.1f}%".format,
                "avg_net_pct": "{:+.2f}%".format,
                "total_net_pct": "{:+.1f}%".format,
                "avg_prob": "{:.3f}".format,
            }))
    print(f"\nout -> {args.out_dir}")


if __name__ == "__main__":
    main()
