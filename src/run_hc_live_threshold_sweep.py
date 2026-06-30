"""Sweep HC live threshold shifts on a fixed recent window.

Used for answering: "can we lower thresholds to get more distinct-symbol trades
while keeping winrate above a floor?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config as C
from .hc import config as HC
from .hc.data import read_json_symbols
from .run_hc_live import _load_horizon_thresholds
from .run_hc_live_hourly_report import (
    _parse_local_end,
    grouped_reports,
    make_candidates,
    simulate,
    summarize,
)
from .run_hc_offgrid_sim import build_feature_rows, score_ensemble


OUT_DIR = C.OUTPUTS_DIR / "analysis" / "hc_live_threshold_sweep"


def parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=Path, default=C.OUTPUTS_DIR / "analysis" / "hc_offgrid" / "threshold_optimizer_300pd_top50" / "per_horizon_thresholds.csv")
    ap.add_argument("--shifts", default="-0.02,-0.04,-0.06,-0.08,-0.10,-0.12,-0.14,-0.16,-0.18,-0.20,-0.25,-0.30")
    ap.add_argument("--end-local", default="2026-06-05 12:00")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--scan-stride-min", type=int, default=5)
    ap.add_argument("--opp-cap", type=float, default=0.20)
    ap.add_argument("--top-per-scan", default="50")
    ap.add_argument("--max-open", default="6,8,10,12,16")
    ap.add_argument("--cooldown-min", type=int, default=0)
    ap.add_argument("--stake-margin", type=float, default=8.0)
    ap.add_argument("--leverage", type=float, default=5.0)
    ap.add_argument("--initial-balance", type=float, default=100.0)
    ap.add_argument("--model-dir", type=Path, default=Path("models/hc_exec_stride120_nonoverlap"))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--save-trades", action="store_true")
    args = ap.parse_args()

    shifts = parse_floats(args.shifts)
    top_values = parse_ints(args.top_per_scan)
    max_open_values = parse_ints(args.max_open)
    end_utc = _parse_local_end(args.end_local)
    start_utc = end_utc - pd.Timedelta(hours=float(args.hours))
    entries = pd.date_range(
        start_utc.floor(f"{int(args.scan_stride_min)}min"),
        end_utc.floor(f"{int(args.scan_stride_min)}min"),
        freq=f"{int(args.scan_stride_min)}min",
        tz="UTC",
    )
    symbols = sorted(set(read_json_symbols()) - C.hc_blacklist_symbols())
    base_thresholds = _load_horizon_thresholds(args.thresholds, 0.0)
    print(
        f"window {entries[0]} -> {entries[-1]} end={end_utc} "
        f"entries={len(entries)} symbols={len(symbols)} horizons={len(base_thresholds)}",
        flush=True,
    )

    features = build_feature_rows(
        symbols=symbols,
        entries=entries,
        horizons=tuple(base_thresholds),
        entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN,
    )
    print(f"features={len(features)}", flush=True)
    scored = score_ensemble(features, args.model_dir)

    rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    hourly_frames: list[pd.DataFrame] = []
    symbol_frames: list[pd.DataFrame] = []
    for shift in shifts:
        thresholds = {
            int(h): min(0.999, max(0.0, float(v) + float(shift)))
            for h, v in base_thresholds.items()
        }
        for conviction in (False, True):
            cand = make_candidates(
                scored,
                thresholds,
                opp_cap=args.opp_cap,
                leverage=args.leverage,
                conviction=conviction,
            )
            cand = cand[cand["deadline"].le(end_utc)].copy() if not cand.empty else cand
            for top in top_values:
                for max_open in max_open_values:
                    trades, blocks = simulate(
                        cand,
                        list(entries),
                        end_utc=end_utc,
                        top_per_scan=top,
                        max_open=max_open,
                        cooldown_min=args.cooldown_min,
                        stake_margin=args.stake_margin,
                    )
                    summary = summarize(trades, hours=args.hours, initial_balance=args.initial_balance)
                    label = f"shift{shift:+.3f}_top{top}_cap{max_open}_{'conv' if conviction else 'flat'}"
                    summary.update(blocks)
                    summary.update({
                        "label": label,
                        "shift": shift,
                        "top_per_scan": top,
                        "max_open": max_open,
                        "conviction": conviction,
                        "candidates": int(len(cand)),
                        "candidate_scans": int(cand["anchor_time"].nunique()) if len(cand) else 0,
                    })
                    rows.append(summary)
                    if args.save_trades and len(trades):
                        tr = trades.copy()
                        tr["label"] = label
                        trade_frames.append(tr)
                    if len(trades):
                        hourly, symbols_df = grouped_reports(trades)
                        hourly["label"] = label
                        symbols_df["label"] = label
                        hourly_frames.append(hourly)
                        symbol_frames.append(symbols_df)
        print(f"shift {shift:+.3f} done", flush=True)

    out = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"to_{end_utc.strftime('%Y%m%d_%H%M')}_h{int(args.hours)}"
    out.to_csv(args.out_dir / f"{tag}_summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_parquet(args.out_dir / f"{tag}_trades.parquet", index=False)
    if hourly_frames:
        pd.concat(hourly_frames, ignore_index=True).to_csv(args.out_dir / f"{tag}_hourly.csv", index=False)
    if symbol_frames:
        pd.concat(symbol_frames, ignore_index=True).to_csv(args.out_dir / f"{tag}_symbols.csv", index=False)

    ranked = out[
        (out["trades"] >= 4)
        & (out["win"].fillna(0) >= 0.72)
    ].sort_values(["trades", "pnl_usd", "max_drawdown_usd"], ascending=[False, False, False])
    print("\nBEST WIN>=72%, >=4 TRADES")
    cols = [
        "label", "trades", "trades_per_day", "win", "avg_net_pct",
        "pnl_usd", "max_drawdown_usd", "candidate_seen",
        "block_already_open", "block_max_open", "max_open_used",
    ]
    print(ranked[cols].head(30).to_string(index=False))
    print(f"\nout -> {args.out_dir}")


if __name__ == "__main__":
    main()
