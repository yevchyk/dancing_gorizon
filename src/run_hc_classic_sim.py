"""Run HC signals through the old live-like candle-replay simulator.

This is intentionally a sidecar: it does not change the legacy simulator.  It
adapts leak-free HC probabilities into the candidate table expected by
run_test_engine_harvest_sim.simulate_engine.

Default run:
    python -m src.run_hc_classic_sim --date 2026-06-04 --fresh
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from . import config as C
from .hc import config as HC
from .hc.data import build_dataset_shards, load_dataset, to_ns
from .markets import get
from .run_test_engine_harvest_sim import simulate_engine
from .run_test_engines_compare import EXIT_MIN


OUT_DIR = C.OUTPUTS_DIR / "analysis" / "hc_classic"
RESULT_MD = C.ROOT / "docs" / "HC_CLASSIC_20260604_1H.md"
SUMMARY_CSV = C.ROOT / "docs" / "HC_CLASSIC_20260604_1H_SUMMARY.csv"
DENSE_HORIZON_DEFAULT = "10,15,20,25,30,35,40,45,50,60,75,90,120"


@dataclass(frozen=True)
class ClassicConfig:
    name: str
    high: float
    opp_cap: float
    horizon_min: int
    horizon_max: int
    entry_delay_min: int
    wait_confirm: bool = False
    top_per_scan: int = 3
    cooldown_min: int = 30


@dataclass
class PriceSeries:
    ts_ns: np.ndarray
    close: np.ndarray

    def at(self, t: pd.Timestamp) -> float | None:
        t_ns = int(pd.Timestamp(t).value)
        idx = int(np.searchsorted(self.ts_ns, t_ns, side="right")) - 1
        if idx < 0:
            return None
        px = float(self.close[idx])
        return px if np.isfinite(px) and px > 0 else None


class ProductionPriceBook:
    """PriceBook compatible with the legacy simulator, backed by data/candles."""

    def __init__(self) -> None:
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
                )
        series = self._cache[symbol]
        return None if series is None else series.at(t)


def _model_folds(model_dir: Path) -> list[str]:
    snapshot = model_dir / "config_snapshot.json"
    if snapshot.exists():
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        names = [f["name"] for f in data.get("folds", [])]
        if names:
            return names
    return sorted(p.name for p in model_dir.iterdir() if (p / "up.cbm").exists())


def _load_model(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(path)
    return model


def score_ensemble(df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    folds = _model_folds(model_dir)
    if not folds:
        raise FileNotFoundError(f"No HC fold models found under {model_dir}")
    x = df[HC.FEATURE_COLUMNS]
    up_preds: list[np.ndarray] = []
    down_preds: list[np.ndarray] = []
    for fold in folds:
        fold_dir = model_dir / fold
        print(f"  score ensemble fold {fold}", flush=True)
        up = _load_model(fold_dir / "up.cbm")
        down = _load_model(fold_dir / "down.cbm")
        up_preds.append(up.predict_proba(x)[:, 1].astype("float32"))
        down_preds.append(down.predict_proba(x)[:, 1].astype("float32"))
    out = df.copy()
    out["up_prob"] = np.vstack(up_preds).mean(axis=0).astype("float32")
    out["down_prob"] = np.vstack(down_preds).mean(axis=0).astype("float32")
    out["model_vote_count"] = len(folds)
    return out


def _confirmation_mask(df: pd.DataFrame, side: str, book: ProductionPriceBook) -> np.ndarray:
    ok = np.zeros(len(df), dtype=bool)
    for i, row in enumerate(df[["symbol", "base_time"]].itertuples(index=False)):
        base = pd.Timestamp(row.base_time)
        p5 = book.at(str(row.symbol), base + pd.Timedelta(minutes=5))
        p10 = book.at(str(row.symbol), base + pd.Timedelta(minutes=10))
        if p5 is None or p10 is None:
            continue
        move = p10 / p5 - 1.0
        ok[i] = move > 0.0 if side == "long" else move < 0.0
    return ok


def make_candidates(
    scored: pd.DataFrame,
    cfg: ClassicConfig,
    *,
    book: ProductionPriceBook,
    leverage: float,
) -> pd.DataFrame:
    parts = []
    base_mask = scored["horizon_minutes"].between(cfg.horizon_min, cfg.horizon_max)
    for side_name, side_int, prob_col, opp_col in (
        ("long", 1, "up_prob", "down_prob"),
        ("short", -1, "down_prob", "up_prob"),
    ):
        mask = (
            base_mask
            & scored[prob_col].ge(cfg.high)
            & scored[opp_col].le(cfg.opp_cap)
        )
        d = scored.loc[mask, ["symbol", "base_time", "horizon_minutes", prob_col, opp_col]].copy()
        if d.empty:
            continue
        if cfg.wait_confirm:
            conf = _confirmation_mask(d, side_name, book)
            d = d.loc[conf].copy()
            if d.empty:
                continue
        d["side"] = side_int
        d["exit"] = d["horizon_minutes"].astype(int).astype(str) + "m"
        d["score"] = d[prob_col].astype(float) - d[opp_col].astype(float)
        d["p_dir"] = d[prob_col].astype(float)
        d["p_opp"] = d[opp_col].astype(float)
        d["signal_model"] = cfg.name + "_" + d["exit"].astype(str)
        d["entry_delay_min"] = cfg.entry_delay_min
        d["anchor_time"] = pd.to_datetime(d["base_time"], utc=True) + pd.to_timedelta(
            cfg.entry_delay_min, unit="min"
        )
        parts.append(d)
    if not parts:
        return pd.DataFrame()

    cand = pd.concat(parts, ignore_index=True)
    cand = cand.sort_values(["base_time", "symbol", "score"], ascending=[True, True, False])
    cand = cand.drop_duplicates(["base_time", "symbol"], keep="first")
    cand = cand.sort_values(["anchor_time", "score"], ascending=[True, False])
    cand["engine"] = cfg.name
    cand["family"] = "hc_classic"
    cand["source"] = "hc"
    cand["threshold"] = cfg.high
    cand["leverage"] = float(leverage)
    cand["day"] = pd.to_datetime(cand["anchor_time"], utc=True).dt.strftime("%m-%d")
    return cand[
        [
            "engine",
            "family",
            "source",
            "signal_model",
            "symbol",
            "anchor_time",
            "day",
            "side",
            "exit",
            "threshold",
            "leverage",
            "score",
            "base_time",
            "p_dir",
            "p_opp",
            "entry_delay_min",
        ]
    ]


def account_summary(
    trades: pd.DataFrame,
    *,
    initial_balance: float,
    stake_usd: float,
) -> tuple[pd.DataFrame, dict]:
    if trades.empty:
        return trades.copy(), {
            "trades": 0,
            "win": np.nan,
            "avg_net_pct": np.nan,
            "avg_levered_pct": np.nan,
            "pnl_usd": 0.0,
            "final_balance": initial_balance,
            "roi_pct": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
        }
    out = trades.sort_values("closed_at").copy()
    out["pnl_usd"] = stake_usd * out["levered_pnl_pct"].astype(float) / 100.0
    out["balance_after"] = initial_balance + out["pnl_usd"].cumsum()
    peaks = pd.concat(
        [pd.Series([initial_balance]), out["balance_after"].reset_index(drop=True)],
        ignore_index=True,
    ).cummax().iloc[1:].to_numpy()
    dd = out["balance_after"].to_numpy("float64") - peaks
    max_dd = float(dd.min()) if len(dd) else 0.0
    summary = {
        "trades": int(len(out)),
        "win": float(out["won"].mean()),
        "avg_net_pct": float(out["net_pnl_pct"].mean()),
        "avg_levered_pct": float(out["levered_pnl_pct"].mean()),
        "pnl_usd": float(out["pnl_usd"].sum()),
        "final_balance": float(initial_balance + out["pnl_usd"].sum()),
        "roi_pct": float(out["pnl_usd"].sum() / initial_balance * 100.0),
        "max_drawdown_usd": max_dd,
        "max_drawdown_pct": float(max_dd / initial_balance * 100.0),
    }
    return out, summary


def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            show[col] = show[col].astype(str)
    widths = [len(c) for c in show.columns]
    rows = show.values.tolist()
    for row in rows:
        widths = [max(w, len(str(v))) for w, v in zip(widths, row)]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(show.columns, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _parse_horizons(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return tuple(int(h) for h in HC.HORIZON_ANCHORS)
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    horizons = tuple(sorted(dict.fromkeys(int(p) for p in parts)))
    if not horizons:
        raise ValueError("horizons must not be empty")
    bad = [h for h in horizons if h <= 0 or h % 5 != 0]
    if bad:
        raise ValueError(f"horizons must be positive 5m-grid values, got {bad}")
    return horizons


def _pct_label(value: float) -> str:
    return f"{int(round(float(value) * 100)):02d}"


def _default_output_paths(
    *,
    tag: str,
    horizons: tuple[int, ...],
    tag_suffix: str,
) -> tuple[str, Path, Path, Path]:
    default_horizons = tuple(int(h) for h in HC.HORIZON_ANCHORS)
    suffix = tag_suffix.strip("_ ")
    if not suffix and horizons != default_horizons:
        suffix = f"dense_h{min(horizons)}_{max(horizons)}_x{len(horizons)}"
    run_label = f"{tag}_1h" + (f"_{suffix}" if suffix else "")
    if suffix:
        safe = suffix.upper().replace("-", "_")
        report_md = C.ROOT / "docs" / f"HC_CLASSIC_{tag}_1H_{safe}.md"
        summary_csv = C.ROOT / "docs" / f"HC_CLASSIC_{tag}_1H_{safe}_SUMMARY.csv"
    else:
        report_md = RESULT_MD
        summary_csv = SUMMARY_CSV
    analysis_dir = OUT_DIR / run_label
    return run_label, analysis_dir, report_md, summary_csv


def write_report(
    *,
    report_md: Path,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    side: pd.DataFrame,
    top_trades: pd.DataFrame,
    metadata: dict,
) -> None:
    lines = [
        "# HC Classic Simulation",
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
        "## Daily",
        "",
        _md_table(daily),
        "",
        "## Side Split",
        "",
        _md_table(side),
        "",
        "## Top Trades",
        "",
        _md_table(top_trades),
        "",
        "## Leakage Check",
        "",
        "- Feature rows are built at `base_time`; HC feature lookup uses bars with timestamp <= `base_time`.",
        "- The executable entry is delayed to `base_time + entry_delay_min`, so the shared close[t] target leak is not used.",
        "- Candidate selection uses only `up_prob`, `down_prob`, `horizon_minutes`, `symbol`, and `base_time`.",
        "- Realized returns/labels from the HC dataset are not used for candidate selection; PnL is replayed from candles by the legacy simulator.",
        "- `wait_confirm` configs use the move from `base_time+5m` to `base_time+10m`, but their entry time is also `base_time+10m`; they must be read as delayed-entry configs, not decisions at `base_time`.",
        "",
    ]
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-06-04")
    ap.add_argument("--stride-min", type=int, default=60)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=Path("models/hc_exec_stride120_nonoverlap"))
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--analysis-dir", type=Path, default=None)
    ap.add_argument("--balance", type=float, default=100.0)
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--leverage", type=float, default=8.0)
    ap.add_argument("--max-open", type=int, default=0)
    ap.add_argument("--top-per-scan", type=int, default=3)
    ap.add_argument("--cooldown-min", type=int, default=30)
    ap.add_argument("--high", type=float, default=0.90)
    ap.add_argument("--opp-cap", type=float, default=0.20)
    ap.add_argument(
        "--horizons",
        default="",
        help=f"comma-separated HC horizons; e.g. {DENSE_HORIZON_DEFAULT}",
    )
    ap.add_argument("--tag-suffix", default="", help="suffix for output/report paths")
    ap.add_argument("--report-md", type=Path, default=None)
    ap.add_argument("--summary-csv", type=Path, default=None)
    args = ap.parse_args()

    date = pd.Timestamp(args.date, tz="UTC")
    start = date
    end = date + pd.Timedelta(days=1)
    tag = date.strftime("%Y%m%d")
    horizons = _parse_horizons(args.horizons)
    run_label, default_analysis_dir, default_report_md, default_summary_csv = _default_output_paths(
        tag=tag,
        horizons=horizons,
        tag_suffix=args.tag_suffix,
    )
    HC.HORIZON_ANCHORS = horizons
    analysis_dir = args.analysis_dir or default_analysis_dir
    dataset_dir = args.dataset_dir or (C.DATA_DIR / f"hc_exec_classic_{run_label}" / "dataset")
    report_md = args.report_md or default_report_md
    summary_csv = args.summary_csv or default_summary_csv
    max_open = args.max_open or max(1, int(args.balance // args.stake))

    for h in horizons:
        EXIT_MIN[f"{int(h)}m"] = int(h)

    print(f"horizons -> {','.join(str(h) for h in horizons)}", flush=True)
    print(f"build HC dataset -> {dataset_dir}", flush=True)
    build_dataset_shards(
        out_dir=dataset_dir,
        stride_min=args.stride_min,
        days=args.days,
        anchors_only=True,
        random_count=0,
        entry_delay_min=HC.EXEC_ENTRY_DELAY_MIN,
        fresh=args.fresh,
    )

    df = load_dataset(dataset_dir)
    df = df[(df["base_time"] >= start) & (df["base_time"] < end)].copy()
    if df.empty:
        raise RuntimeError(f"No HC rows for {start} -> {end} in {dataset_dir}")
    print(
        f"date rows={len(df)} symbols={df['symbol'].nunique()} "
        f"base={df['base_time'].min()} -> {df['base_time'].max()}",
        flush=True,
    )

    scored = score_ensemble(df, args.model_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    scored_path = analysis_dir / "hc_classic_scored.parquet"
    scored.to_parquet(scored_path, index=False)

    prob_label = _pct_label(args.high)
    opp_label = _pct_label(args.opp_cap)
    configs = [
        ClassicConfig(
            name=f"hc_plain_mid_p{prob_label}_opp{opp_label}",
            high=args.high,
            opp_cap=args.opp_cap,
            horizon_min=30,
            horizon_max=90,
            entry_delay_min=5,
            top_per_scan=args.top_per_scan,
            cooldown_min=args.cooldown_min,
        ),
        ClassicConfig(
            name=f"hc_plain_all_p{prob_label}_opp{opp_label}",
            high=args.high,
            opp_cap=args.opp_cap,
            horizon_min=min(horizons),
            horizon_max=max(horizons),
            entry_delay_min=5,
            top_per_scan=args.top_per_scan,
            cooldown_min=args.cooldown_min,
        ),
        ClassicConfig(
            name=f"hc_wait_confirm_mid_p{prob_label}_opp{opp_label}",
            high=args.high,
            opp_cap=args.opp_cap,
            horizon_min=30,
            horizon_max=90,
            entry_delay_min=10,
            wait_confirm=True,
            top_per_scan=args.top_per_scan,
            cooldown_min=args.cooldown_min,
        ),
        ClassicConfig(
            name=f"hc_wait_confirm_all_p{prob_label}_opp{opp_label}",
            high=args.high,
            opp_cap=args.opp_cap,
            horizon_min=min(horizons),
            horizon_max=max(horizons),
            entry_delay_min=10,
            wait_confirm=True,
            top_per_scan=args.top_per_scan,
            cooldown_min=args.cooldown_min,
        ),
    ]

    book = ProductionPriceBook()
    scan_outputs: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    block_rows: list[dict] = []
    for cfg in configs:
        print(f"simulate {cfg.name}", flush=True)
        cand = make_candidates(scored, cfg, book=book, leverage=args.leverage)
        cand_path = analysis_dir / f"{cfg.name}_candidates.parquet"
        cand.to_parquet(cand_path, index=False)
        scan_times = sorted(pd.Timestamp(t) for t in cand["anchor_time"].drop_duplicates()) if len(cand) else []
        trades, blocks = simulate_engine(
            cfg.name,
            cand,
            scan_times,
            book,
            harvest=False,
            top_per_scan=cfg.top_per_scan,
            max_open=max_open,
            cooldown_min=cfg.cooldown_min,
        )
        trades, acct = account_summary(trades, initial_balance=args.balance, stake_usd=args.stake)
        if len(trades):
            trades["profile"] = cfg.name
            scan_outputs.append(trades)
        block_rows.append(blocks | {"candidates": int(len(cand)), "scan_times": int(len(scan_times))})
        summary_rows.append(
            {
                "profile": cfg.name,
                "candidates": int(len(cand)),
                "scan_times": int(len(scan_times)),
                "top_per_scan": cfg.top_per_scan,
                "max_open": max_open,
                "cooldown_min": cfg.cooldown_min,
                "entry_delay_min": cfg.entry_delay_min,
                "wait_confirm": cfg.wait_confirm,
                **acct,
                "max_open_used": blocks.get("max_open_used", 0),
                "block_max_open": blocks.get("block_max_open", 0),
                "block_cooldown": blocks.get("block_cooldown", 0),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["final_balance", "trades"], ascending=[False, False])
    trades_all = pd.concat(scan_outputs, ignore_index=True) if scan_outputs else pd.DataFrame()
    blocks = pd.DataFrame(block_rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    blocks.to_csv(analysis_dir / "hc_classic_blocks.csv", index=False)
    if len(trades_all):
        trades_path = analysis_dir / "hc_classic_trades.parquet"
        trades_all.to_parquet(trades_path, index=False)
        daily = (
            trades_all.assign(day=pd.to_datetime(trades_all["closed_at"], utc=True).dt.strftime("%Y-%m-%d"))
            .groupby(["profile", "day"])
            .agg(
                trades=("pnl_usd", "size"),
                win=("won", "mean"),
                pnl_usd=("pnl_usd", "sum"),
                avg_net_pct=("net_pnl_pct", "mean"),
                avg_lev_pct=("levered_pnl_pct", "mean"),
            )
            .reset_index()
        )
        side = (
            trades_all.groupby(["profile", "side"])
            .agg(
                trades=("pnl_usd", "size"),
                win=("won", "mean"),
                pnl_usd=("pnl_usd", "sum"),
                avg_net_pct=("net_pnl_pct", "mean"),
                avg_lev_pct=("levered_pnl_pct", "mean"),
            )
            .reset_index()
        )
        top_trades = trades_all.sort_values("pnl_usd", ascending=False)[
            [
                "profile",
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
        ].head(30)
    else:
        trades_path = analysis_dir / "hc_classic_trades.parquet"
        daily = pd.DataFrame()
        side = pd.DataFrame()
        top_trades = pd.DataFrame()

    daily.to_csv(analysis_dir / "hc_classic_daily.csv", index=False)
    side.to_csv(analysis_dir / "hc_classic_side.csv", index=False)
    metadata = {
        "date_utc": args.date,
        "base_time_min": scored["base_time"].min(),
        "base_time_max": scored["base_time"].max(),
        "symbols": int(scored["symbol"].nunique()),
        "rows": int(len(scored)),
        "scan_cadence": f"{args.stride_min}min",
        "horizons": ",".join(str(h) for h in horizons),
        "horizon_count": len(horizons),
        "balance_usd": args.balance,
        "stake_margin_usd": args.stake,
        "leverage": args.leverage,
        "notional_per_trade_usd": args.stake * args.leverage,
        "model_dir": str(args.model_dir),
        "scored_path": str(scored_path),
        "trades_path": str(trades_path),
    }
    write_report(
        report_md=report_md,
        summary=summary,
        daily=daily,
        side=side,
        top_trades=top_trades,
        metadata=metadata,
    )
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nreport -> {report_md}")
    print(f"summary -> {summary_csv}")
    print(f"analysis -> {analysis_dir}")


if __name__ == "__main__":
    main()
