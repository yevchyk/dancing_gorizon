"""Live-like HC hourly report for a fixed local cutoff.

This is intentionally narrower than the broader research sims:
* per-horizon thresholds match run_hc_live;
* one best signal per symbol per scan;
* one active position per symbol, max-open cap, cooldown;
* only trades whose fixed-horizon deadline is at or before the cutoff are scored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .hc import config as HC
from .hc.data import read_json_symbols
from .run_hc_live import _load_horizon_thresholds
from .run_hc_offgrid_sim import build_feature_rows, score_ensemble
from .run_hc_classic_sim import ProductionPriceBook
from .trading.hc_live_engine import HCLiveEngine


OUT_DIR = C.OUTPUTS_DIR / "analysis" / "hc_live_hourly"
FEE_AND_SLIP = 0.0015


def _parse_local_end(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Europe/Kiev")
    return ts.tz_convert("UTC")


def make_candidates(
    scored: pd.DataFrame,
    thresholds: dict[int, float],
    *,
    opp_cap: float,
    leverage: float,
    conviction: bool,
) -> pd.DataFrame:
    columns = [
        "symbol", "anchor_time", "deadline", "base_time", "horizon_minutes",
        "side", "side_name", "exit", "threshold", "p_dir", "p_opp",
        "spread", "score", "signal_model", "size_mult", "leverage",
    ]
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
        m = d[prob_col].ge(d["threshold"]) & d[opp_col].le(float(opp_cap))
        d = d.loc[m].copy()
        if d.empty:
            continue
        d["side"] = side_int
        d["side_name"] = side_name
        d["p_dir"] = d[prob_col].astype("float64")
        d["p_opp"] = d[opp_col].astype("float64")
        d["spread"] = d["p_dir"] - d["p_opp"]
        d["score"] = (d["p_dir"] - d["threshold"]) + d["spread"]
        d["exit"] = d["horizon_minutes"].astype(int).astype(str) + "m"
        d["signal_model"] = "hc_live_" + d["exit"].astype(str)
        d["anchor_time"] = pd.to_datetime(d["base_time"], utc=True) + pd.Timedelta(
            minutes=HC.EXEC_ENTRY_DELAY_MIN
        )
        d["deadline"] = d["anchor_time"] + pd.to_timedelta(
            d["horizon_minutes"].astype("int64"), unit="min"
        )
        d["size_mult"] = (
            d["spread"].map(HCLiveEngine.conviction_mult).astype("float64")
            if conviction else 1.0
        )
        d["leverage"] = float(leverage)
        parts.append(d)

    if not parts:
        return pd.DataFrame(columns=columns)
    cand = pd.concat(parts, ignore_index=True)
    cand = cand.sort_values(["anchor_time", "symbol", "score"], ascending=[True, True, False])
    cand = cand.drop_duplicates(["anchor_time", "symbol"], keep="first")
    cand = cand.sort_values(["anchor_time", "score"], ascending=[True, False]).reset_index(drop=True)
    return cand[columns]


def price_at(book: ProductionPriceBook, symbol: str, t: pd.Timestamp) -> float | None:
    px = book.at(symbol, t)
    if px is None or not np.isfinite(px) or px <= 0:
        return None
    return float(px)


def close_record(pos: dict, close_time: pd.Timestamp, exit_px: float, stake_margin: float) -> dict:
    side = int(pos["side"])
    gross = side * (exit_px / float(pos["entry_price"]) - 1.0)
    net = gross - FEE_AND_SLIP
    notional = float(stake_margin) * float(pos["leverage"]) * float(pos["size_mult"])
    return {
        **{k: pos[k] for k in (
            "symbol", "side_name", "exit", "score", "threshold", "p_dir",
            "p_opp", "spread", "size_mult", "leverage",
        )},
        "opened_at": pos["opened_at"],
        "closed_at": close_time,
        "entry_price": float(pos["entry_price"]),
        "exit_price": float(exit_px),
        "net_pnl_pct": net * 100.0,
        "levered_pnl_pct": net * 100.0 * float(pos["leverage"]),
        "pnl_usd": notional * net,
        "won": int(net > 0),
        "held_min": (close_time - pd.Timestamp(pos["opened_at"])).total_seconds() / 60.0,
    }


def simulate(
    candidates: pd.DataFrame,
    scan_times: list[pd.Timestamp],
    *,
    end_utc: pd.Timestamp,
    top_per_scan: int,
    max_open: int,
    cooldown_min: int,
    stake_margin: float,
) -> tuple[pd.DataFrame, dict]:
    book = ProductionPriceBook()
    candidates = candidates[candidates["deadline"].le(end_utc)].copy()
    by_time = {
        pd.Timestamp(t): g.sort_values("score", ascending=False).head(int(top_per_scan))
        for t, g in candidates.groupby("anchor_time", sort=False)
    }
    open_pos: dict[str, dict] = {}
    last_trade_at: dict[str, pd.Timestamp] = {}
    trades: list[dict] = []
    blocks = {
        "candidate_seen": 0,
        "block_already_open": 0,
        "block_max_open": 0,
        "block_cooldown": 0,
        "block_no_price": 0,
        "max_open_used": 0,
    }

    def close_due(now: pd.Timestamp) -> None:
        for sym, pos in list(open_pos.items()):
            if now < pos["deadline"]:
                continue
            px = price_at(book, sym, pos["deadline"])
            if px is None:
                blocks["block_no_price"] += 1
                continue
            trades.append(close_record(pos, pos["deadline"], px, stake_margin))
            del open_pos[sym]

    for now in scan_times:
        if now > end_utc:
            break
        close_due(now)
        g = by_time.get(now)
        if g is None or g.empty:
            blocks["max_open_used"] = max(blocks["max_open_used"], len(open_pos))
            continue
        for row in g.itertuples(index=False):
            blocks["candidate_seen"] += 1
            sym = str(row.symbol)
            if sym in open_pos:
                blocks["block_already_open"] += 1
                continue
            if len(open_pos) >= int(max_open):
                blocks["block_max_open"] += 1
                continue
            last = last_trade_at.get(sym)
            if last is not None and now < last + pd.Timedelta(minutes=int(cooldown_min)):
                blocks["block_cooldown"] += 1
                continue
            entry = price_at(book, sym, now)
            if entry is None:
                blocks["block_no_price"] += 1
                continue
            pos = row._asdict()
            pos["opened_at"] = now
            pos["entry_price"] = entry
            open_pos[sym] = pos
            last_trade_at[sym] = now
            blocks["max_open_used"] = max(blocks["max_open_used"], len(open_pos))

    close_due(end_utc)
    return pd.DataFrame(trades), blocks


def summarize(trades: pd.DataFrame, *, hours: float, initial_balance: float) -> dict:
    if trades.empty:
        return {
            "trades": 0, "trades_per_day": 0.0, "win": np.nan,
            "avg_net_pct": np.nan, "pnl_usd": 0.0,
            "final_balance": initial_balance, "max_drawdown_usd": 0.0,
        }
    out = trades.sort_values("closed_at").copy()
    bal = float(initial_balance) + out["pnl_usd"].cumsum()
    peaks = pd.concat([pd.Series([float(initial_balance)]), bal.reset_index(drop=True)], ignore_index=True).cummax().iloc[1:]
    dd = bal.to_numpy("float64") - peaks.to_numpy("float64")
    return {
        "trades": int(len(out)),
        "trades_per_day": float(len(out) / max(hours, 1e-9) * 24.0),
        "win": float(out["won"].mean()),
        "avg_net_pct": float(out["net_pnl_pct"].mean()),
        "avg_levered_pct": float(out["levered_pnl_pct"].mean()),
        "pnl_usd": float(out["pnl_usd"].sum()),
        "final_balance": float(initial_balance + out["pnl_usd"].sum()),
        "max_drawdown_usd": float(dd.min()) if len(dd) else 0.0,
    }


def grouped_reports(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = trades.copy()
    x["hour_kyiv"] = pd.to_datetime(x["opened_at"], utc=True).dt.tz_convert("Europe/Kiev").dt.floor("1h")
    hourly = (
        x.groupby("hour_kyiv")
        .agg(
            trades=("symbol", "size"),
            win=("won", "mean"),
            avg_net_pct=("net_pnl_pct", "mean"),
            pnl_usd=("pnl_usd", "sum"),
            symbols=("symbol", "nunique"),
        )
        .reset_index()
    )
    hourly["cum_usd"] = hourly["pnl_usd"].cumsum()
    symbols = (
        x.groupby("symbol")
        .agg(
            trades=("symbol", "size"),
            win=("won", "mean"),
            avg_net_pct=("net_pnl_pct", "mean"),
            pnl_usd=("pnl_usd", "sum"),
            worst_net_pct=("net_pnl_pct", "min"),
        )
        .reset_index()
        .sort_values(["pnl_usd", "avg_net_pct"], ascending=[True, True])
    )
    return hourly, symbols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=Path, default=C.OUTPUTS_DIR / "analysis" / "hc_offgrid" / "threshold_optimizer_300pd_top50" / "per_horizon_thresholds.csv")
    ap.add_argument("--threshold-shift", type=float, default=-0.02)
    ap.add_argument("--end-local", default="2026-06-05 12:00")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--scan-stride-min", type=int, default=5)
    ap.add_argument("--opp-cap", type=float, default=0.20)
    ap.add_argument("--top-per-scan", type=int, default=50)
    ap.add_argument("--max-open", type=int, default=6)
    ap.add_argument("--cooldown-min", type=int, default=0)
    ap.add_argument("--stake-margin", type=float, default=8.0)
    ap.add_argument("--leverage", type=float, default=5.0)
    ap.add_argument("--initial-balance", type=float, default=100.0)
    ap.add_argument("--conviction", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=Path("models/hc_exec_stride120_nonoverlap"))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    end_utc = _parse_local_end(args.end_local)
    start_utc = end_utc - pd.Timedelta(hours=float(args.hours))
    entries = pd.date_range(
        start_utc.floor(f"{int(args.scan_stride_min)}min"),
        end_utc.floor(f"{int(args.scan_stride_min)}min"),
        freq=f"{int(args.scan_stride_min)}min",
        tz="UTC",
    )
    thresholds = _load_horizon_thresholds(args.thresholds, args.threshold_shift)
    symbols = sorted(set(read_json_symbols()) - C.hc_blacklist_symbols())
    print(f"window {entries[0]} -> {entries[-1]} end={end_utc} entries={len(entries)} symbols={len(symbols)} horizons={len(thresholds)}", flush=True)

    features = build_feature_rows(
        symbols=symbols,
        entries=entries,
        horizons=tuple(thresholds),
        entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN,
    )
    print(f"features={len(features)}", flush=True)
    scored = score_ensemble(features, args.model_dir)
    cand = make_candidates(
        scored,
        thresholds,
        opp_cap=args.opp_cap,
        leverage=args.leverage,
        conviction=args.conviction,
    )
    cand = cand[cand["deadline"].le(end_utc)].copy()
    print(f"candidates_completed_by_end={len(cand)} scans={cand['anchor_time'].nunique() if len(cand) else 0}", flush=True)
    trades, blocks = simulate(
        cand,
        list(entries),
        end_utc=end_utc,
        top_per_scan=args.top_per_scan,
        max_open=args.max_open,
        cooldown_min=args.cooldown_min,
        stake_margin=args.stake_margin,
    )
    summary = summarize(trades, hours=args.hours, initial_balance=args.initial_balance)
    summary.update(blocks)
    hourly, symbols_df = grouped_reports(trades)

    tag = f"to_{end_utc.strftime('%Y%m%d_%H%M')}_h{int(args.hours)}_top{args.top_per_scan}_cap{args.max_open}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.out_dir / f"{tag}_summary.csv", index=False)
    cand.to_parquet(args.out_dir / f"{tag}_candidates.parquet", index=False)
    trades.to_parquet(args.out_dir / f"{tag}_trades.parquet", index=False)
    hourly.to_csv(args.out_dir / f"{tag}_hourly.csv", index=False)
    symbols_df.to_csv(args.out_dir / f"{tag}_symbols.csv", index=False)

    print("\nSUMMARY")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nHOURLY KYIV")
    print(hourly.to_string(index=False, formatters={
        "win": "{:.1%}".format,
        "avg_net_pct": "{:+.2f}".format,
        "pnl_usd": "{:+.2f}".format,
        "cum_usd": "{:+.2f}".format,
    }))
    print("\nWORST SYMBOLS")
    print(symbols_df.head(20).to_string(index=False, formatters={
        "win": "{:.1%}".format,
        "avg_net_pct": "{:+.2f}".format,
        "pnl_usd": "{:+.2f}".format,
        "worst_net_pct": "{:+.2f}".format,
    }))
    print(f"\nout -> {args.out_dir}")


if __name__ == "__main__":
    main()
