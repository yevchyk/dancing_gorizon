"""Probe arbitrary HC horizons against exact 1m exits.

This is a research sidecar for "what if the horizon feature is smooth enough to
score 23m/27m even though the original HC labels were 5m-aligned?".  It does not
change live trading defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .hc import config as HC
from .hc.data import (
    _build_feature_matrix,
    prepare_btc_frames,
    prepare_timeframes,
    read_json_symbols,
    to_ns,
)
from .markets import get
from .run_hc_classic_sim import (
    ClassicConfig,
    _md_table,
    _pct_label,
    account_summary,
    make_candidates,
    score_ensemble,
)
from .run_test_engine_harvest_sim import simulate_engine
from .run_test_engines_compare import EXIT_MIN


OUT_DIR = C.OUTPUTS_DIR / "analysis" / "hc_offgrid"

DEFAULT_GRID5 = "10,15,20,25,30,35,40,45,50,60,75,90,120"
DEFAULT_OFFGRID = "10,15,20,23,25,27,30,33,35,37,40,43,45,47,50,55,60,75,90,120"
NS_PER_MIN = 60_000_000_000


@dataclass
class PriceSeries:
    ts_ns: np.ndarray
    close: np.ndarray
    max_stale_min: float

    def at(self, t: pd.Timestamp) -> float | None:
        t_ns = int(pd.Timestamp(t).value)
        idx = int(np.searchsorted(self.ts_ns, t_ns, side="right")) - 1
        if idx < 0:
            return None
        stale_min = (t_ns - int(self.ts_ns[idx])) / NS_PER_MIN
        if stale_min > self.max_stale_min:
            return None
        px = float(self.close[idx])
        return px if np.isfinite(px) and px > 0 else None


class StrictProductionPriceBook:
    """Price book from crypto_feature with stale-price protection.

    For the latest day the production store contains 1m candles, so 23m/27m
    exits should resolve to an exact minute close.  If a symbol only has 5m data
    at that timestamp, off-grid exits will be blocked instead of silently using
    an old close.
    """

    def __init__(self, max_stale_min: float = 2.0) -> None:
        self.max_stale_min = float(max_stale_min)
        self._cache: dict[str, PriceSeries | None] = {}

    def at(self, symbol: str, t: pd.Timestamp) -> float | None:
        if symbol not in self._cache:
            df = get(HC.STORE_KEY).load(symbol)
            if df is None or df.empty:
                self._cache[symbol] = None
            else:
                df = df.sort_index()
                self._cache[symbol] = PriceSeries(
                    to_ns(df.index),
                    df["close"].to_numpy("float64"),
                    self.max_stale_min,
                )
        series = self._cache[symbol]
        return None if series is None else series.at(t)


def parse_horizons(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    horizons = tuple(sorted(dict.fromkeys(int(p) for p in parts)))
    if not horizons:
        raise ValueError("horizons must not be empty")
    bad = [h for h in horizons if h <= 0]
    if bad:
        raise ValueError(f"horizons must be positive minute values, got {bad}")
    return horizons


def entry_grid(date: pd.Timestamp, stride_min: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(date).tz_convert("UTC") if pd.Timestamp(date).tzinfo else pd.Timestamp(date, tz="UTC")
    end = start + pd.Timedelta(days=1)
    return pd.date_range(
        start,
        end - pd.Timedelta(minutes=1),
        freq=f"{int(stride_min)}min",
        tz="UTC",
    )


def apply_fetch_delay(
    entries: pd.DatetimeIndex,
    *,
    min_sec: float,
    max_sec: float,
    seed: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    lo = max(0, int(round(float(min_sec))))
    hi = max(lo, int(round(float(max_sec))))
    if len(entries) == 0 or hi <= 0:
        return entries, np.zeros(len(entries), dtype="int32")
    rng = np.random.default_rng(int(seed))
    delays = rng.integers(lo, hi + 1, size=len(entries), dtype="int32")
    delayed = pd.DatetimeIndex(entries + pd.to_timedelta(delays, unit="s"))
    delayed = delayed.tz_convert("UTC") if delayed.tz is not None else delayed.tz_localize("UTC")
    return delayed, delays


def build_feature_rows(
    *,
    symbols: list[str],
    entries: pd.DatetimeIndex,
    horizons: tuple[int, ...],
    entry_delay_min: int,
) -> pd.DataFrame:
    btc_frames = prepare_btc_frames()
    base_times = entries - pd.Timedelta(minutes=int(entry_delay_min))
    horizon_arr = np.array(horizons, dtype="int16")
    h_count = len(horizon_arr)
    no_horizon_cols = HC.FEATURE_COLUMNS[:-2]
    frames: list[pd.DataFrame] = []

    for idx, symbol in enumerate(symbols, start=1):
        if idx == 1 or idx % 25 == 0 or idx == len(symbols):
            print(f"  features {idx}/{len(symbols)} {symbol}", flush=True)
        raw = get(HC.STORE_KEY).load(symbol)
        if raw is None or raw.empty:
            continue
        try:
            prepared = prepare_timeframes(raw, btc_frames)
            if not prepared:
                continue
            features, valid = _build_feature_matrix(base_times, prepared, HC.N_POINTS)
        except Exception as exc:
            print(f"  skip {symbol}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not bool(valid.any()):
            continue

        base_valid = base_times[valid]
        entry_valid = entries[valid]
        feature_rows = np.repeat(features[valid], h_count, axis=0)
        h = np.tile(horizon_arr, int(valid.sum()))

        data: dict[str, object] = {
            "symbol": np.repeat(symbol, len(h)),
            "base_time": np.repeat(base_valid.to_numpy(), h_count),
            "entry_time": np.repeat(entry_valid.to_numpy(), h_count),
        }
        for col_idx, col in enumerate(no_horizon_cols):
            data[col] = feature_rows[:, col_idx].astype("float32", copy=False)
        data["horizon_minutes"] = h
        data["horizon_log"] = np.log1p(h.astype("float32")).astype("float32")
        frames.append(pd.DataFrame(data))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["base_time"] = pd.to_datetime(out["base_time"], utc=True)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    return out[["symbol", "base_time", "entry_time", *HC.FEATURE_COLUMNS]]


def smoothness_table(scored: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    horizon_set = set(int(h) for h in horizons)
    rows: list[dict] = []
    for h in sorted(horizon_set):
        if h % 5 == 0:
            continue
        lo = (h // 5) * 5
        hi = lo + 5
        if lo not in horizon_set or hi not in horizon_set:
            continue
        sub = scored[scored["horizon_minutes"].isin([lo, h, hi])]
        for side, col in (("long", "up_prob"), ("short", "down_prob")):
            piv = sub.pivot_table(
                index=["symbol", "base_time"],
                columns="horizon_minutes",
                values=col,
                aggfunc="first",
            )
            needed = [lo, h, hi]
            if any(x not in piv.columns for x in needed):
                continue
            piv = piv[needed].dropna()
            if piv.empty:
                continue
            interp = piv[lo] + (piv[hi] - piv[lo]) * ((h - lo) / (hi - lo))
            diff = piv[h] - interp
            rows.append(
                {
                    "side": side,
                    "horizon": h,
                    "bracket": f"{lo}-{hi}",
                    "n": int(len(diff)),
                    "mean_diff_pp": float(diff.mean() * 100.0),
                    "mae_pp": float(diff.abs().mean() * 100.0),
                    "p95_abs_pp": float(diff.abs().quantile(0.95) * 100.0),
                    "max_abs_pp": float(diff.abs().max() * 100.0),
                }
            )
    return pd.DataFrame(rows)


def candidate_horizon_table(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    out = (
        candidates.groupby(["engine", "exit"])
        .agg(
            candidates=("symbol", "size"),
            scans=("anchor_time", "nunique"),
            symbols=("symbol", "nunique"),
            avg_p=("p_dir", "mean"),
            avg_opp=("p_opp", "mean"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )
    out["h_min"] = out["exit"].str.replace("m", "", regex=False).astype(int)
    return out.sort_values(["engine", "h_min"]).drop(columns=["h_min"])


def trade_horizon_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    out = (
        trades.groupby(["engine", "exit"])
        .agg(
            trades=("symbol", "size"),
            win=("won", "mean"),
            avg_net_pct=("net_pnl_pct", "mean"),
            median_net_pct=("net_pnl_pct", "median"),
            total_net_pct=("net_pnl_pct", "sum"),
            avg_hold_min=("held_min", "mean"),
        )
        .reset_index()
    )
    out["h_min"] = out["exit"].str.replace("m", "", regex=False).astype(int)
    return out.sort_values(["engine", "h_min"]).drop(columns=["h_min"])


def run_profile(
    *,
    name: str,
    scored: pd.DataFrame,
    high: float,
    opp_cap: float,
    top_per_scan: int,
    cooldown_min: int,
    max_open: int,
    leverage: float,
    initial_balance: float,
    stake_usd: float,
    book: StrictProductionPriceBook,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    cfg = ClassicConfig(
        name=name,
        high=high,
        opp_cap=opp_cap,
        horizon_min=int(scored["horizon_minutes"].min()),
        horizon_max=int(scored["horizon_minutes"].max()),
        entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN,
        top_per_scan=top_per_scan,
        cooldown_min=cooldown_min,
    )
    cand = make_candidates(scored, cfg, book=book, leverage=leverage)
    scan_times = sorted(pd.Timestamp(t) for t in cand["anchor_time"].drop_duplicates()) if len(cand) else []
    trades, blocks = simulate_engine(
        name,
        cand,
        scan_times,
        book,
        harvest=False,
        top_per_scan=top_per_scan,
        max_open=max_open,
        cooldown_min=cooldown_min,
    )
    trades, account = account_summary(trades, initial_balance=initial_balance, stake_usd=stake_usd)
    if len(trades):
        trades["profile"] = name
    return cand, trades, account, blocks


def write_report(
    *,
    path: Path,
    metadata: dict,
    summary: pd.DataFrame,
    smoothness: pd.DataFrame,
    candidate_by_horizon: pd.DataFrame,
    trade_by_horizon: pd.DataFrame,
    top_trades: pd.DataFrame,
) -> None:
    lines = [
        "# HC Off-Grid Horizon Simulation",
        "",
        f"Generated: {pd.Timestamp.now('UTC').isoformat()}",
        "",
        "## Setup",
        "",
        _md_table(pd.DataFrame([metadata])),
        "",
        "## Summary",
        "",
        _md_table(summary),
        "",
        "## Horizon Smoothness",
        "",
        _md_table(smoothness),
        "",
        "## Candidates By Horizon",
        "",
        _md_table(candidate_by_horizon, max_rows=120),
        "",
        "## Trades By Horizon",
        "",
        _md_table(trade_by_horizon, max_rows=120),
        "",
        "## Top Trades",
        "",
        _md_table(top_trades, max_rows=40),
        "",
        "## Notes",
        "",
        "- Features are built at `entry_time - 5m`; entry is at `entry_time`.",
        "- Optional fetch-delay mode shifts each scheduled scan by a deterministic random delay before feature building/scoring.",
        "- Off-grid horizons are used only as model input plus exact deadline exits.",
        "- PnL uses `crypto_feature` prices with stale-price guard; for 2026-06-04 this store has 1m candles.",
        "- Candidate selection uses only probabilities, side, horizon, symbol, and time.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-06-04")
    ap.add_argument("--scan-stride-min", type=int, default=5)
    ap.add_argument("--grid5-horizons", default=DEFAULT_GRID5)
    ap.add_argument("--offgrid-horizons", default=DEFAULT_OFFGRID)
    ap.add_argument("--model-dir", type=Path, default=Path("models/hc_exec_stride120_nonoverlap"))
    ap.add_argument("--high", type=float, default=0.88)
    ap.add_argument("--opp-cap", type=float, default=0.20)
    ap.add_argument("--top-per-scan", type=int, default=6)
    ap.add_argument("--cooldown-min", type=int, default=30)
    ap.add_argument("--max-open", type=int, default=10)
    ap.add_argument("--balance", type=float, default=100.0)
    ap.add_argument("--stake", type=float, default=5.0)
    ap.add_argument("--leverage", type=float, default=3.0)
    ap.add_argument("--price-max-stale-min", type=float, default=2.0)
    ap.add_argument("--fetch-delay-min-sec", type=float, default=0.0)
    ap.add_argument("--fetch-delay-max-sec", type=float, default=0.0)
    ap.add_argument("--fetch-delay-seed", type=int, default=42)
    ap.add_argument("--analysis-dir", type=Path, default=None)
    ap.add_argument("--report-md", type=Path, default=None)
    ap.add_argument("--print-smoothness", action="store_true")
    args = ap.parse_args()

    date = pd.Timestamp(args.date, tz="UTC")
    grid5 = parse_horizons(args.grid5_horizons)
    offgrid = parse_horizons(args.offgrid_horizons)
    all_horizons = tuple(sorted(set(grid5).union(offgrid)))
    for h in all_horizons:
        EXIT_MIN[f"{int(h)}m"] = int(h)

    tag = date.strftime("%Y%m%d")
    prob_label = _pct_label(args.high)
    delay_tag = ""
    if args.fetch_delay_max_sec > 0:
        delay_tag = f"_fd{int(round(args.fetch_delay_min_sec))}-{int(round(args.fetch_delay_max_sec))}s"
    run_label = f"{tag}_s{args.scan_stride_min}_p{prob_label}_x{len(all_horizons)}{delay_tag}"
    analysis_dir = args.analysis_dir or (OUT_DIR / run_label)
    report_md = args.report_md or (C.ROOT / "docs" / f"HC_OFFGRID_{run_label.upper()}.md")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(set(read_json_symbols()) - C.hc_blacklist_symbols())
    scheduled_entries = entry_grid(date, args.scan_stride_min)
    entries, fetch_delays = apply_fetch_delay(
        scheduled_entries,
        min_sec=args.fetch_delay_min_sec,
        max_sec=args.fetch_delay_max_sec,
        seed=args.fetch_delay_seed,
    )
    print(f"entries={len(entries)} {entries.min()} -> {entries.max()}", flush=True)
    if len(fetch_delays) and int(fetch_delays.max()) > 0:
        print(
            f"fetch_delay_sec min={int(fetch_delays.min())} "
            f"p50={float(np.median(fetch_delays)):.0f} max={int(fetch_delays.max())} "
            f"seed={args.fetch_delay_seed}",
            flush=True,
        )
    print(f"symbols={len(symbols)} horizons={','.join(str(h) for h in all_horizons)}", flush=True)

    features = build_feature_rows(
        symbols=symbols,
        entries=entries,
        horizons=all_horizons,
        entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN,
    )
    if features.empty:
        raise RuntimeError("No off-grid feature rows built")
    print(
        f"feature rows={len(features)} symbols={features['symbol'].nunique()} "
        f"base={features['base_time'].min()} -> {features['base_time'].max()}",
        flush=True,
    )

    scored = score_ensemble(features, args.model_dir)
    scored_path = analysis_dir / "hc_offgrid_scored.parquet"
    scored.to_parquet(scored_path, index=False)

    book = StrictProductionPriceBook(max_stale_min=args.price_max_stale_min)
    opp_label = _pct_label(args.opp_cap)
    profiles = [
        (f"hc_grid5_p{prob_label}_opp{opp_label}", set(grid5)),
        (f"hc_offgrid_p{prob_label}_opp{opp_label}", set(offgrid)),
    ]
    cand_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    block_rows: list[dict] = []
    for name, hset in profiles:
        subset = scored[scored["horizon_minutes"].isin(hset)].copy()
        print(f"simulate {name}: rows={len(subset)} horizons={len(hset)}", flush=True)
        cand, trades, account, blocks = run_profile(
            name=name,
            scored=subset,
            high=args.high,
            opp_cap=args.opp_cap,
            top_per_scan=args.top_per_scan,
            cooldown_min=args.cooldown_min,
            max_open=args.max_open,
            leverage=args.leverage,
            initial_balance=args.balance,
            stake_usd=args.stake,
            book=book,
        )
        if len(cand):
            cand_frames.append(cand)
            cand.to_parquet(analysis_dir / f"{name}_candidates.parquet", index=False)
        if len(trades):
            trade_frames.append(trades)
            trades.to_parquet(analysis_dir / f"{name}_trades.parquet", index=False)
        block_rows.append(blocks | {"profile": name, "candidates": int(len(cand))})
        summary_rows.append(
            {
                "profile": name,
                "horizons": ",".join(str(h) for h in sorted(hset)),
                "horizon_count": len(hset),
                "candidates": int(len(cand)),
                "candidate_scans": int(cand["anchor_time"].nunique()) if len(cand) else 0,
                "top_per_scan": args.top_per_scan,
                "max_open": args.max_open,
                "cooldown_min": args.cooldown_min,
                "stake_margin_usd": args.stake,
                "leverage": args.leverage,
                **account,
                "max_open_used": blocks.get("max_open_used", 0),
                "block_max_open": blocks.get("block_max_open", 0),
                "block_cooldown": blocks.get("block_cooldown", 0),
                "block_no_price": blocks.get("block_no_price", 0),
            }
        )

    candidates_all = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    if len(candidates_all):
        candidates_all.to_parquet(analysis_dir / "hc_offgrid_candidates_all.parquet", index=False)
    if len(trades_all):
        trades_all.to_parquet(analysis_dir / "hc_offgrid_trades_all.parquet", index=False)

    summary = pd.DataFrame(summary_rows).sort_values(["final_balance", "trades"], ascending=[False, False])
    smooth = smoothness_table(scored, all_horizons)
    cand_h = candidate_horizon_table(candidates_all)
    trade_h = trade_horizon_table(trades_all)
    if len(trades_all):
        top_trades = trades_all.sort_values("pnl_usd", ascending=False)[
            [
                "engine",
                "symbol",
                "side",
                "exit",
                "opened_at",
                "closed_at",
                "net_pnl_pct",
                "levered_pnl_pct",
                "pnl_usd",
                "balance_after",
            ]
        ].head(40)
    else:
        top_trades = pd.DataFrame()

    summary.to_csv(analysis_dir / "summary.csv", index=False)
    smooth.to_csv(analysis_dir / "smoothness.csv", index=False)
    cand_h.to_csv(analysis_dir / "candidates_by_horizon.csv", index=False)
    trade_h.to_csv(analysis_dir / "trades_by_horizon.csv", index=False)
    pd.DataFrame(block_rows).to_csv(analysis_dir / "blocks.csv", index=False)

    metadata = {
        "date_utc": args.date,
        "scheduled_entry_min": scheduled_entries.min(),
        "scheduled_entry_max": scheduled_entries.max(),
        "entry_min": entries.min(),
        "entry_max": entries.max(),
        "scan_stride_min": args.scan_stride_min,
        "fetch_delay_min_sec": args.fetch_delay_min_sec,
        "fetch_delay_max_sec": args.fetch_delay_max_sec,
        "fetch_delay_seed": args.fetch_delay_seed,
        "fetch_delay_realized_min_sec": int(fetch_delays.min()) if len(fetch_delays) else 0,
        "fetch_delay_realized_max_sec": int(fetch_delays.max()) if len(fetch_delays) else 0,
        "symbols": len(symbols),
        "rows": int(len(scored)),
        "grid5_horizons": ",".join(str(h) for h in grid5),
        "offgrid_horizons": ",".join(str(h) for h in offgrid),
        "high": args.high,
        "opp_cap": args.opp_cap,
        "model_dir": str(args.model_dir),
        "scored_path": str(scored_path),
        "analysis_dir": str(analysis_dir),
    }
    write_report(
        path=report_md,
        metadata=metadata,
        summary=summary,
        smoothness=smooth,
        candidate_by_horizon=cand_h,
        trade_by_horizon=trade_h,
        top_trades=top_trades,
    )

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    if args.print_smoothness:
        print("\nSMOOTHNESS")
        print(smooth.to_string(index=False))
    print(f"\nreport -> {report_md}")
    print(f"analysis -> {analysis_dir}")


if __name__ == "__main__":
    main()
