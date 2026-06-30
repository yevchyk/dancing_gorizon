"""Sandbox strategy simulations over HC probabilities.

This does not retrain models. It stress-tests already scored HC probabilities
with executable returns:

    signal observed at t
    earliest honest entry = close[t + 5m] on the historical 5m grid
    exit = entry + horizon

The script intentionally lives outside the production/live engine. It is a
research playground for ranking probability/curve filters before we rebuild the
leak-free dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .hc import config as HC
from .hc.data import _load_raw, prepare_5m, to_ns


SCORED = HC.ANALYSIS_DIR / "hc_scored.parquet"
EXEC_SCORED = HC.ANALYSIS_DIR / "hc_scored_exec.parquet"
OUT_CSV = HC.RESULTS_MD.parent / "HC_SANDBOX_SIM_RESULTS.csv"
OUT_MD = HC.RESULTS_MD.parent / "HC_SANDBOX_SIM_RESULTS.md"
TRADES_OUT = HC.ANALYSIS_DIR / "hc_sandbox_top_trades.parquet"


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    high: float
    opp_cap: float
    horizon_min: int
    horizon_max: int
    score_kind: str
    curve_min_count: int = 1
    opp_curve_cap: float | None = None
    lastbar_filter: str = "none"
    lastbar_abs_cap_pct: float | None = None
    wait_filter: str = "none"
    entry_delay_min: int = 5
    top_per_scan: int = 20
    max_open: int = 80
    cooldown_min: int = 0
    extra_slip_pct: float = 0.0


def _price_at(index_ns: np.ndarray, close: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    if len(index_ns) == 0:
        return np.full(len(query_ns), np.nan, dtype="float64")
    pos = np.searchsorted(index_ns, query_ns)
    valid = (pos < len(index_ns)) & (index_ns[np.minimum(pos, len(index_ns) - 1)] == query_ns)
    out = np.full(len(query_ns), np.nan, dtype="float64")
    out[valid] = close[pos[valid]]
    return out


def build_exec_scored(scored_path: Path = SCORED, out_path: Path = EXEC_SCORED, fresh: bool = False) -> pd.DataFrame:
    if out_path.exists() and not fresh:
        print(f"load cached executable scored -> {out_path}")
        return pd.read_parquet(out_path)

    cols = [
        "fold",
        "symbol",
        "base_time",
        "horizon_minutes",
        "ret",
        "thr_pct",
        "up_prob",
        "down_prob",
        "c5m_rel_0",
        "c5m_rel_1",
        "c15m_rel_0",
        "c1h_rel_0",
        "c4h_rel_0",
    ]
    df = pd.read_parquet(scored_path, columns=cols)
    df["base_time"] = pd.to_datetime(df["base_time"], utc=True)
    df["last5_ret"] = df["c5m_rel_0"].astype("float64") - 1.0
    df["prev5_ret"] = df["c5m_rel_1"].astype("float64") - 1.0

    parts = []
    n_symbols = df["symbol"].nunique()
    for i, (symbol, g) in enumerate(df.groupby("symbol", sort=False), 1):
        base = prepare_5m(_load_raw(symbol))
        index_ns = to_ns(base.index)
        close = base["close"].to_numpy("float64")
        t_ns = g["base_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        h_min = g["horizon_minutes"].to_numpy("int64")

        entry_5 = _price_at(index_ns, close, t_ns + 5 * HC.NS_PER_MIN)
        exit_5 = _price_at(index_ns, close, t_ns + (h_min + 5) * HC.NS_PER_MIN)
        entry_10 = _price_at(index_ns, close, t_ns + 10 * HC.NS_PER_MIN)
        exit_10 = _price_at(index_ns, close, t_ns + (h_min + 10) * HC.NS_PER_MIN)

        out = g.copy()
        out["ret_exec_5"] = exit_5 / entry_5 - 1.0
        out["ret_exec_10"] = exit_10 / entry_10 - 1.0
        out["pre_entry_5m"] = entry_10 / entry_5 - 1.0
        parts.append(out)
        if i % 50 == 0 or i == n_symbols:
            print(f"  executable returns {i}/{n_symbols}", flush=True)

    full = pd.concat(parts, ignore_index=True)

    group_cols = ["fold", "symbol", "base_time"]
    full["up_count_070"] = full["up_prob"].ge(0.70).groupby([full[c] for c in group_cols]).transform("sum").astype("int16")
    full["down_count_070"] = full["down_prob"].ge(0.70).groupby([full[c] for c in group_cols]).transform("sum").astype("int16")
    full["up_curve_max"] = full["up_prob"].groupby([full[c] for c in group_cols]).transform("max").astype("float32")
    full["down_curve_max"] = full["down_prob"].groupby([full[c] for c in group_cols]).transform("max").astype("float32")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    print(f"exec scored -> {out_path} rows={len(full)}")
    return full


def _side_candidates(df: pd.DataFrame, cfg: StrategyConfig, side: str) -> pd.DataFrame:
    if side == "LONG":
        p_dir = df["up_prob"]
        p_opp = df["down_prob"]
        curve_count = df["up_count_070"]
        opp_curve = df["down_curve_max"]
        signed_ret = df[f"ret_exec_{cfg.entry_delay_min}"]
        last = df["last5_ret"]
        pre = df["pre_entry_5m"]
    else:
        p_dir = df["down_prob"]
        p_opp = df["up_prob"]
        curve_count = df["down_count_070"]
        opp_curve = df["up_curve_max"]
        signed_ret = -df[f"ret_exec_{cfg.entry_delay_min}"]
        last = -df["last5_ret"]
        pre = -df["pre_entry_5m"]

    mask = (
        df["horizon_minutes"].between(cfg.horizon_min, cfg.horizon_max)
        & p_dir.ge(cfg.high)
        & p_opp.le(cfg.opp_cap)
        & signed_ret.notna()
    )
    if cfg.curve_min_count > 1:
        mask &= curve_count.ge(cfg.curve_min_count)
    if cfg.opp_curve_cap is not None:
        mask &= opp_curve.le(cfg.opp_curve_cap)

    if cfg.lastbar_filter == "no_spike" and cfg.lastbar_abs_cap_pct is not None:
        mask &= df["last5_ret"].abs().le(cfg.lastbar_abs_cap_pct / 100.0)
    elif cfg.lastbar_filter == "no_mirror" and cfg.lastbar_abs_cap_pct is not None:
        mask &= last.ge(-(cfg.lastbar_abs_cap_pct / 100.0))
    elif cfg.lastbar_filter == "momentum":
        mask &= last.gt(0.0)
    elif cfg.lastbar_filter == "reversal" and cfg.lastbar_abs_cap_pct is not None:
        mask &= last.lt(-(cfg.lastbar_abs_cap_pct / 100.0))

    if cfg.wait_filter == "confirm":
        mask &= pre.gt(0.0)
    elif cfg.wait_filter == "pullback":
        mask &= pre.lt(0.0)

    d = df.loc[mask, ["fold", "symbol", "base_time", "horizon_minutes"]].copy()
    if d.empty:
        return d
    d["side"] = side
    d["p_dir"] = p_dir.loc[mask].to_numpy("float64")
    d["p_opp"] = p_opp.loc[mask].to_numpy("float64")
    d["spread"] = d["p_dir"] - d["p_opp"]
    d["signed_ret"] = signed_ret.loc[mask].to_numpy("float64")
    d["net_ret"] = d["signed_ret"] - (HC.ROUND_TRIP_FEE_PCT + cfg.extra_slip_pct) / 100.0
    if cfg.score_kind == "prob":
        d["score"] = d["p_dir"]
    elif cfg.score_kind == "spread_per_min":
        d["score"] = d["spread"] / d["horizon_minutes"].clip(lower=1)
    elif cfg.score_kind == "spread_per_sqrt_min":
        d["score"] = d["spread"] / np.sqrt(d["horizon_minutes"].clip(lower=1))
    else:
        d["score"] = d["spread"]
    d["entry_time"] = d["base_time"] + pd.to_timedelta(cfg.entry_delay_min, unit="min")
    d["exit_time"] = d["entry_time"] + pd.to_timedelta(d["horizon_minutes"], unit="min")
    return d


def make_candidates(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    cand = pd.concat(
        [_side_candidates(df, cfg, "LONG"), _side_candidates(df, cfg, "SHORT")],
        ignore_index=True,
    )
    if cand.empty:
        return cand
    cand = cand.sort_values(["fold", "base_time", "symbol", "score"], ascending=[True, True, True, False])
    cand = cand.drop_duplicates(["fold", "base_time", "symbol"], keep="first")
    cand = cand.sort_values(["fold", "base_time", "score"], ascending=[True, True, False])
    cand["scan_rank"] = cand.groupby(["fold", "base_time"])["score"].rank(ascending=False, method="first")
    return cand[cand["scan_rank"] <= cfg.top_per_scan].copy()


def simulate_config(df: pd.DataFrame, cfg: StrategyConfig, keep_trades: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cand = make_candidates(df, cfg)
    if cand.empty:
        empty = pd.DataFrame()
        return empty, empty

    trades: list[dict] = []
    for fold, fdf in cand.groupby("fold", sort=False):
        open_positions: list[tuple[pd.Timestamp, str]] = []
        last_trade_at: dict[str, pd.Timestamp] = {}
        for scan_time, sdf in fdf.groupby("base_time", sort=True):
            scan_time = pd.Timestamp(scan_time)
            open_positions = [(et, sym) for et, sym in open_positions if et > scan_time]
            open_symbols = {sym for _, sym in open_positions}
            slots = max(0, cfg.max_open - len(open_positions))
            if slots == 0:
                continue
            for row in sdf.sort_values("score", ascending=False).itertuples(index=False):
                if slots <= 0:
                    break
                if row.symbol in open_symbols:
                    continue
                prev = last_trade_at.get(row.symbol)
                if prev is not None and (pd.Timestamp(row.entry_time) - prev).total_seconds() < cfg.cooldown_min * 60:
                    continue
                rec = row._asdict()
                rec["strategy"] = cfg.name
                rec["win"] = bool(rec["net_ret"] > 0)
                trades.append(rec)
                open_positions.append((pd.Timestamp(row.exit_time), row.symbol))
                open_symbols.add(row.symbol)
                last_trade_at[row.symbol] = pd.Timestamp(row.entry_time)
                slots -= 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return pd.DataFrame(), trades_df

    rows = []
    for fold, f in trades_df.groupby("fold", sort=False):
        rows.append(_summary_row(cfg, f, fold))
    rows.append(_summary_row(cfg, trades_df, "ALL"))
    summary = pd.DataFrame(rows)
    if not keep_trades:
        trades_df = pd.DataFrame()
    return summary, trades_df


def _summary_row(cfg: StrategyConfig, trades: pd.DataFrame, fold: str) -> dict:
    net = trades["net_ret"].to_numpy("float64")
    days = max(1.0, trades["base_time"].dt.floor("D").nunique())
    return {
        **asdict(cfg),
        "fold": fold,
        "trades": int(len(trades)),
        "win_rate": float((net > 0).mean()),
        "avg_net_pct": float(net.mean() * 100.0),
        "median_net_pct": float(np.median(net) * 100.0),
        "total_net_pct": float(net.sum() * 100.0),
        "trades_per_day": float(len(trades) / days),
        "long_share": float((trades["side"] == "LONG").mean()),
        "avg_horizon": float(trades["horizon_minutes"].mean()),
        "avg_score": float(trades["score"].mean()),
    }


def strategy_grid() -> list[StrategyConfig]:
    configs: list[StrategyConfig] = []
    horizons = {
        "all": (5, 180),
        "no5": (10, 180),
        "short": (5, 30),
        "mid": (30, 90),
        "long": (60, 180),
    }
    base_filters = [
        ("plain", "none", None, "none", 1, None, 5),
        ("curve2", "none", None, "none", 2, None, 5),
        ("curve3", "none", None, "none", 3, None, 5),
        ("oppcurve30", "none", None, "none", 1, 0.30, 5),
        ("nospike1", "no_spike", 1.0, "none", 1, None, 5),
        ("nospike2", "no_spike", 2.0, "none", 1, None, 5),
        ("nomirror03", "no_mirror", 0.3, "none", 1, None, 5),
        ("momentum", "momentum", None, "none", 1, None, 5),
        ("reversal1", "reversal", 1.0, "none", 1, None, 5),
        ("wait_confirm", "none", None, "confirm", 1, None, 10),
        ("wait_pullback", "none", None, "pullback", 1, None, 10),
    ]
    for hname, (hmin, hmax) in horizons.items():
        for high, opp in [(0.70, 0.30), (0.80, 0.25), (0.90, 0.20)]:
            for score in ["spread", "spread_per_sqrt_min", "prob"]:
                for fname, last_filter, cap, wait, curve_n, opp_curve, delay in base_filters:
                    name = f"{hname}_p{int(high*100)}_opp{int(opp*100)}_{score}_{fname}"
                    configs.append(
                        StrategyConfig(
                            name=name,
                            high=high,
                            opp_cap=opp,
                            horizon_min=hmin,
                            horizon_max=hmax,
                            score_kind=score,
                            curve_min_count=curve_n,
                            opp_curve_cap=opp_curve,
                            lastbar_filter=last_filter,
                            lastbar_abs_cap_pct=cap,
                            wait_filter=wait,
                            entry_delay_min=delay,
                        )
                    )
    return configs


def _rank(summary: pd.DataFrame) -> pd.DataFrame:
    all_rows = summary[summary["fold"] == "ALL"].copy()
    fold_rows = summary[summary["fold"] != "ALL"].copy()
    mins = fold_rows.groupby("name").agg(
        min_fold_avg_net_pct=("avg_net_pct", "min"),
        min_fold_win_rate=("win_rate", "min"),
        min_fold_trades=("trades", "min"),
    )
    ranked = all_rows.merge(mins, left_on="name", right_index=True, how="left")
    ranked = ranked.sort_values(
        ["min_fold_avg_net_pct", "avg_net_pct", "min_fold_win_rate", "trades"],
        ascending=[False, False, False, False],
    )
    return ranked


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    cols = [
        "name",
        "fold",
        "trades",
        "win_rate",
        "avg_net_pct",
        "total_net_pct",
        "min_fold_avg_net_pct",
        "min_fold_win_rate",
        "avg_horizon",
        "long_share",
    ]
    show = df[[c for c in cols if c in df.columns]].head(max_rows).copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    widths = [len(c) for c in show.columns]
    rows = [[str(v) for v in row] for row in show.itertuples(index=False, name=None)]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(show.columns, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_report(
    summary: pd.DataFrame,
    ranked: pd.DataFrame,
    configs: list[StrategyConfig],
    *,
    out_csv: Path = OUT_CSV,
    out_md: Path = OUT_MD,
    scored_path: Path = SCORED,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    top = ranked.head(30)
    positive = ranked[(ranked["min_fold_avg_net_pct"] > 0) & (ranked["min_fold_win_rate"] > 0.5)]
    lines = [
        "# HC Sandbox Simulation Results",
        "",
        f"Generated: {pd.Timestamp.utcnow().isoformat()}",
        "",
        "## Important",
        "",
        f"Scored probabilities: `{scored_path}`",
        "These simulations evaluate trades with executable 5m-grid entry.",
        "They are a strategy-shape diagnostic, not a green light for live trading.",
        "",
        f"Configs tested: {len(configs)}",
        f"Round-trip fee: {HC.ROUND_TRIP_FEE_PCT:.2f}%",
        "Entry convention: signal at t, earliest entry close[t+5m]; wait strategies enter close[t+10m].",
        "",
        "Historical fold data is 5m-grid here. Exact 2m/4m around-signal checks require a new 1m/4m scored dataset.",
        "",
        "## Robust Positive Strategies",
        "",
        _markdown_table(positive.head(30)),
        "",
        "## Top Ranked By Worst Fold Avg Net",
        "",
        _markdown_table(top),
        "",
        "## CSV",
        "",
        f"`{out_csv}`",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary csv -> {out_csv}")
    print(f"report -> {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, default=SCORED)
    ap.add_argument("--exec-scored", type=Path, default=EXEC_SCORED)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--trades-out", type=Path, default=TRADES_OUT)
    ap.add_argument("--fresh-exec", action="store_true")
    ap.add_argument("--limit-configs", type=int, default=0)
    ap.add_argument("--keep-top-trades", type=int, default=10)
    args = ap.parse_args()

    df = build_exec_scored(args.scored, args.exec_scored, fresh=args.fresh_exec)
    configs = strategy_grid()
    if args.limit_configs:
        configs = configs[: args.limit_configs]
    print(f"simulate configs={len(configs)} rows={len(df)}")

    summaries = []
    trades_to_save = []
    for i, cfg in enumerate(configs, 1):
        summary, _ = simulate_config(df, cfg, keep_trades=False)
        if not summary.empty:
            summaries.append(summary)
        if i % 50 == 0 or i == len(configs):
            print(f"  simulated {i}/{len(configs)}", flush=True)
    full_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    ranked = _rank(full_summary)

    if args.keep_top_trades and not ranked.empty:
        for name in ranked.head(args.keep_top_trades)["name"]:
            cfg = next(c for c in configs if c.name == name)
            _, trades = simulate_config(df, cfg, keep_trades=True)
            if not trades.empty:
                trades_to_save.append(trades)
        if trades_to_save:
            trades_df = pd.concat(trades_to_save, ignore_index=True)
            args.trades_out.parent.mkdir(parents=True, exist_ok=True)
            trades_df.to_parquet(args.trades_out, index=False)
            print(f"top trades -> {args.trades_out}")

    write_report(full_summary, ranked, configs, out_csv=args.out_csv, out_md=args.out_md, scored_path=args.scored)
    print("\nTop 20:")
    cols = ["name", "trades", "win_rate", "avg_net_pct", "min_fold_avg_net_pct", "min_fold_win_rate"]
    print(ranked[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
