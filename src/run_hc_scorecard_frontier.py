"""Prototype HC scorecard frontier.

Consumes the frozen leg table produced by ``run_hc_scorecard_analysis`` and
turns the discovered generators into a transparent scorecard:

* broad pool generation: RAW90 / SPREAD80 / HMEAN85 / SMEAN70 / temporal impulse;
* diminishing generator bonus, soft penalties and a regime multiplier;
* operating-point frontier with one best leg per symbol, top-N per scan, max-open
  and cooldown constraints.

This is a research sidecar, not live trading code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dh import config as cfg


DEFAULT_LEGS = (
    cfg.ROOT
    / "outputs"
    / "analysis"
    / "hc_scorecard"
    / "old_2026-06-01_4d_h30-90_p50_slip0p6"
    / "frozen_legs.parquet"
)
OUT_DIR = cfg.ROOT / "outputs" / "analysis" / "hc_scorecard_frontier"


def _clip01(x) -> np.ndarray:
    return np.clip(np.asarray(x, dtype="float64"), 0.0, 1.0)


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            show[col] = show[col].astype(str)
    widths = [len(str(c)) for c in show.columns]
    rows = show.values.tolist()
    for row in rows:
        widths = [max(w, len(str(v))) for w, v in zip(widths, row)]
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(show.columns, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


@dataclass(frozen=True)
class SimConfig:
    threshold: float
    top_per_scan: int
    max_open: int
    cooldown_min: int


def add_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Generators: broad candidate rights.
    d["gen_raw90"] = d["p_dir"].ge(0.90)
    d["gen_spread80"] = d["spread"].ge(0.80)
    d["gen_hmean85"] = d["horizon_mean_prob"].ge(0.85)
    d["gen_smean70"] = d["horizon_mean_spread"].ge(0.70)
    d["gen_tprob_slope50"] = d["temporal_prob_slope_30m"].ge(0.505)
    gen_cols = ["gen_raw90", "gen_spread80", "gen_hmean85", "gen_smean70", "gen_tprob_slope50"]
    d["pool"] = d[gen_cols].any(axis=1)
    d["generator_count"] = d[gen_cols].sum(axis=1).astype("int8")

    # Generator evidence uses max + small combo bonus to avoid double-counting
    # highly correlated RAW/SPREAD/HMEAN signals.
    raw_e = 1.25 * _clip01((d["p_dir"] - 0.86) / 0.12)
    spread_e = 1.10 * _clip01((d["spread"] - 0.72) / 0.16)
    hmean_e = 0.90 * _clip01((d["horizon_mean_prob"] - 0.78) / 0.14)
    smean_e = 0.70 * _clip01((d["horizon_mean_spread"] - 0.62) / 0.18)
    impulse_e = 0.95 * _clip01((d["temporal_prob_slope_30m"] - 0.20) / 0.45)
    gen_stack = np.vstack([raw_e, spread_e, hmean_e, smean_e, impulse_e])
    d["score_generator"] = gen_stack.max(axis=0)
    d["score_combo"] = np.minimum(np.maximum(d["generator_count"].astype(float) - 1.0, 0.0), 3.0) * 0.08

    # Soft evidence/penalties.  Opposite probability is not a hard gate.
    d["score_horizon"] = (
        0.18 * _clip01(d["horizon_prob_count_085"] / 4.0)
        + 0.10 * _clip01(d["horizon_prob_count_070"] / 4.0)
        + 0.10 * _clip01((d["horizon_max_spread"] - 0.70) / 0.20)
    )
    d["score_low_opp"] = 0.10 * _clip01((0.06 - d["p_opp"]) / 0.06)
    d["penalty_opp"] = 0.45 * _clip01((d["p_opp"] - 0.08) / 0.22)
    d["penalty_conflict"] = 0.20 * _clip01((d["p_opp"] - 0.18) / 0.20)

    pre = (
        d["score_generator"].astype(float)
        + d["score_combo"].astype(float)
        + d["score_horizon"].astype(float)
        + d["score_low_opp"].astype(float)
        - d["penalty_opp"].astype(float)
        - d["penalty_conflict"].astype(float)
    )
    d["score_pre_regime"] = pre.astype("float32")

    # Regime boosts active broad markets but does not create candidates by itself.
    regime_p = _clip01((d["regime_avg_p"] - 0.58) / 0.13)
    regime_sp = _clip01((d["regime_avg_spread"] - 0.53) / 0.14)
    d["regime_mult"] = (0.78 + 0.26 * regime_p + 0.16 * regime_sp).clip(0.72, 1.20).astype("float32")
    d["score"] = (d["score_pre_regime"] * d["regime_mult"]).astype("float32")
    d["score_rank"] = d["score"] + 0.08 * d["spread"].astype(float) + 0.04 * d["p_dir"].astype(float)
    return d


def generator_stats(d: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "RAW90": d["gen_raw90"],
        "SPREAD80": d["gen_spread80"],
        "HMEAN85": d["gen_hmean85"],
        "SMEAN70": d["gen_smean70"],
        "TPROB_SLOPE50": d["gen_tprob_slope50"],
        "POOL_ANY": d["pool"],
    }
    rows = []
    for name, mask in masks.items():
        x = d[mask]
        rows.append(_summary_row(name, x))
    return pd.DataFrame(rows).sort_values(["avg_net_pct", "n"], ascending=[False, False])


def _summary_row(name: str, x: pd.DataFrame) -> dict:
    if x.empty:
        return {
            "name": name,
            "n": 0,
            "scans": 0,
            "symbols": 0,
            "win": np.nan,
            "avg_net_pct": np.nan,
            "total_net_pct": 0.0,
            "median_net_pct": np.nan,
            "avg_score": np.nan,
        }
    return {
        "name": name,
        "n": int(len(x)),
        "scans": int(x["base_time"].nunique()),
        "symbols": int(x["symbol"].nunique()),
        "win": float(x["won"].mean()),
        "avg_net_pct": float(x["net_pnl_pct"].mean()),
        "total_net_pct": float(x["net_pnl_pct"].sum()),
        "median_net_pct": float(x["net_pnl_pct"].median()),
        "avg_score": float(x["score"].mean()) if "score" in x.columns else np.nan,
    }


def threshold_grid(d: pd.DataFrame) -> list[float]:
    pool_scores = d.loc[d["pool"], "score"].dropna()
    if pool_scores.empty:
        return []
    qs = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]
    vals = [float(pool_scores.quantile(q)) for q in qs]
    vals.extend(np.linspace(float(pool_scores.min()), float(pool_scores.max()), 16).tolist())
    vals = sorted({round(v, 4) for v in vals if np.isfinite(v)})
    return vals


def preselect(d: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    x = d[d["pool"] & d["score"].ge(float(cfg.threshold))].copy()
    if x.empty:
        return x
    x = x.sort_values(["base_time", "symbol", "score_rank"], ascending=[True, True, False])
    x = x.drop_duplicates(["base_time", "symbol"], keep="first")
    x = x.sort_values(["base_time", "score_rank"], ascending=[True, False])
    if cfg.top_per_scan > 0:
        x = x.groupby("base_time", sort=False, group_keys=False).head(int(cfg.top_per_scan))
    return x.reset_index(drop=True)


def apply_portfolio_constraints(cand: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    if cand.empty:
        return cand.copy()
    open_until: dict[str, pd.Timestamp] = {}
    last_open: dict[str, pd.Timestamp] = {}
    selected: list[pd.Series] = []

    cand = cand.sort_values(["base_time", "score_rank"], ascending=[True, False]).copy()
    for row in cand.itertuples(index=False):
        now = pd.Timestamp(row.entry_time)
        open_until = {sym: until for sym, until in open_until.items() if until > now}
        sym = str(row.symbol)
        if sym in open_until:
            continue
        if cfg.max_open > 0 and len(open_until) >= int(cfg.max_open):
            continue
        prev = last_open.get(sym)
        if prev is not None and now < prev + pd.Timedelta(minutes=int(cfg.cooldown_min)):
            continue
        selected.append(pd.Series(row._asdict()))
        open_until[sym] = pd.Timestamp(row.exit_time)
        last_open[sym] = now
    return pd.DataFrame(selected)


def account_metrics(trades: pd.DataFrame, *, initial_balance: float, stake: float, leverage: float) -> dict:
    if trades.empty:
        return {
            "pnl_usd": 0.0,
            "final_balance": initial_balance,
            "roi_pct": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
        }
    t = trades.sort_values("exit_time").copy()
    pnl = float(stake) * float(leverage) * t["net_pnl_pct"].astype(float) / 100.0
    balance = float(initial_balance) + pnl.cumsum()
    peaks = pd.concat([pd.Series([float(initial_balance)]), balance.reset_index(drop=True)], ignore_index=True).cummax().iloc[1:]
    dd = balance.to_numpy("float64") - peaks.to_numpy("float64")
    max_dd = float(dd.min()) if len(dd) else 0.0
    pnl_sum = float(pnl.sum())
    return {
        "pnl_usd": pnl_sum,
        "final_balance": float(initial_balance) + pnl_sum,
        "roi_pct": pnl_sum / float(initial_balance) * 100.0 if initial_balance else np.nan,
        "max_drawdown_usd": max_dd,
        "max_drawdown_pct": max_dd / float(initial_balance) * 100.0 if initial_balance else np.nan,
    }


def data_window_hours(d: pd.DataFrame) -> float:
    if d.empty:
        return 0.0
    start = pd.Timestamp(d["entry_time"].min())
    end = pd.Timestamp(d["entry_time"].max())
    hours = (end - start).total_seconds() / 3600.0
    return max(0.0, float(hours))


def summarize_trades(
    name: str,
    trades: pd.DataFrame,
    cfg: SimConfig,
    *,
    initial_balance: float,
    stake: float,
    leverage: float,
    window_hours: float,
) -> dict:
    row = _summary_row(name, trades)
    row.update(
        {
            "threshold": cfg.threshold,
            "top_per_scan": cfg.top_per_scan,
            "max_open": cfg.max_open,
            "cooldown_min": cfg.cooldown_min,
        }
    )
    row.update(account_metrics(trades, initial_balance=initial_balance, stake=stake, leverage=leverage))
    if not trades.empty:
        row["trades_per_24h"] = float(len(trades) / window_hours * 24.0) if window_hours > 0 else np.nan
        row["avg_hold_min"] = float((pd.to_datetime(trades["exit_time"], utc=True) - pd.to_datetime(trades["entry_time"], utc=True)).dt.total_seconds().mean() / 60.0)
    else:
        row["trades_per_24h"] = 0.0
        row["avg_hold_min"] = np.nan
    return row


def build_frontier(
    d: pd.DataFrame,
    *,
    top_per_scan_values: list[int],
    max_open_values: list[int],
    cooldown_min: int,
    initial_balance: float,
    stake: float,
    leverage: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict] = []
    saved: dict[str, pd.DataFrame] = {}
    thresholds = threshold_grid(d)
    window_hours = data_window_hours(d)
    for top in top_per_scan_values:
        for max_open in max_open_values:
            for thr in thresholds:
                cfg = SimConfig(threshold=thr, top_per_scan=top, max_open=max_open, cooldown_min=cooldown_min)
                cand = preselect(d, cfg)
                trades = apply_portfolio_constraints(cand, cfg)
                label = f"thr{thr:.4f}_top{top}_cap{max_open}_cd{cooldown_min}"
                rows.append(
                    summarize_trades(
                        label,
                        trades,
                        cfg,
                        initial_balance=initial_balance,
                        stake=stake,
                        leverage=leverage,
                        window_hours=window_hours,
                    )
                )
                if len(trades) >= 20:
                    saved[label] = trades
    frontier = pd.DataFrame(rows)
    if not frontier.empty:
        frontier = frontier.sort_values(["pnl_usd", "avg_net_pct", "n"], ascending=[False, False, False])
    return frontier, saved


def component_lift(d: pd.DataFrame) -> pd.DataFrame:
    pool = d[d["pool"]].copy()
    if pool.empty:
        return pd.DataFrame()
    qs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    rows = []
    for col in ["score", "score_generator", "score_horizon", "penalty_opp", "regime_mult"]:
        bins = pd.qcut(pool[col], q=qs, duplicates="drop")
        tmp = pool.assign(_bin=bins.astype(str))
        for b, g in tmp.groupby("_bin", sort=False):
            rows.append(
                {
                    "component": col,
                    "bin": b,
                    "n": int(len(g)),
                    "win": float(g["won"].mean()),
                    "avg_net_pct": float(g["net_pnl_pct"].mean()),
                    "total_net_pct": float(g["net_pnl_pct"].sum()),
                    "avg_score": float(g["score"].mean()),
                }
            )
    return pd.DataFrame(rows)


def write_report(path: Path, *, metadata: dict, generators: pd.DataFrame, frontier: pd.DataFrame, components: pd.DataFrame) -> None:
    viable = frontier[(frontier["n"] >= 20) & (frontier["avg_net_pct"] > 0)].copy()
    best_pnl = viable.sort_values(["pnl_usd", "avg_net_pct"], ascending=[False, False]).head(20)
    best_avg = viable.sort_values(["avg_net_pct", "n"], ascending=[False, False]).head(20)
    balanced = viable[(viable["n"] >= 200)].sort_values(["pnl_usd", "avg_net_pct"], ascending=[False, False]).head(20)
    lines = [
        "# HC Prototype Scorecard Frontier",
        "",
        f"Generated: {pd.Timestamp.now('UTC').isoformat()}",
        "",
        "## Setup",
        "",
        _md_table(pd.DataFrame([metadata]), max_rows=5),
        "",
        "## Generator Pool",
        "",
        _md_table(generators, max_rows=20),
        "",
        "## Best PnL Frontier",
        "",
        _md_table(best_pnl, max_rows=20),
        "",
        "## Balanced Frontier (n>=200)",
        "",
        _md_table(balanced, max_rows=20),
        "",
        "## Best Avg Net",
        "",
        _md_table(best_avg, max_rows=20),
        "",
        "## Component Lift",
        "",
        _md_table(components, max_rows=50),
        "",
        "## Notes",
        "",
        "- Pool generators are OR'ed; scorecard ranks inside that pool.",
        "- Generator evidence uses max + a small combo bonus, so RAW/SPREAD/HMEAN do not double-count heavily.",
        "- Portfolio simulation keeps one active position per symbol, caps max open positions, and applies cooldown after entry.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", type=Path, default=DEFAULT_LEGS)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--top-per-scan", default="3,6,10,20,50")
    ap.add_argument("--max-open", default="6,10,20")
    ap.add_argument("--cooldown-min", type=int, default=30)
    ap.add_argument("--balance", type=float, default=100.0)
    ap.add_argument("--stake", type=float, default=8.0)
    ap.add_argument("--leverage", type=float, default=5.0)
    args = ap.parse_args()

    legs_path = Path(args.legs)
    if not legs_path.exists():
        raise FileNotFoundError(f"Frozen legs not found: {legs_path}")
    out_dir = args.out_dir or (OUT_DIR / legs_path.parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"load legs -> {legs_path}", flush=True)
    df = pd.read_parquet(legs_path)
    for col in ["base_time", "entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    scored = add_scorecard(df)
    scored.to_parquet(out_dir / "scorecard_legs.parquet", index=False)

    generators = generator_stats(scored)
    components = component_lift(scored)
    frontier, saved = build_frontier(
        scored,
        top_per_scan_values=_parse_ints(args.top_per_scan),
        max_open_values=_parse_ints(args.max_open),
        cooldown_min=args.cooldown_min,
        initial_balance=args.balance,
        stake=args.stake,
        leverage=args.leverage,
    )

    generators.to_csv(out_dir / "generator_stats.csv", index=False)
    components.to_csv(out_dir / "component_lift.csv", index=False)
    frontier.to_csv(out_dir / "frontier.csv", index=False)

    viable = frontier[(frontier["n"] >= 20) & (frontier["avg_net_pct"] > 0)].copy()
    if not viable.empty:
        best_label = str(viable.sort_values(["pnl_usd", "avg_net_pct"], ascending=[False, False]).iloc[0]["name"])
        trades = saved.get(best_label)
        if trades is not None:
            trades.to_parquet(out_dir / "best_trades.parquet", index=False)

    metadata = {
        "legs": str(legs_path),
        "rows": int(len(scored)),
        "data_start": str(scored["entry_time"].min()),
        "data_end": str(scored["entry_time"].max()),
        "data_window_hours": data_window_hours(scored),
        "pool_rows": int(scored["pool"].sum()),
        "pool_win": float(scored.loc[scored["pool"], "won"].mean()),
        "pool_avg_net_pct": float(scored.loc[scored["pool"], "net_pnl_pct"].mean()),
        "top_per_scan": args.top_per_scan,
        "max_open": args.max_open,
        "cooldown_min": args.cooldown_min,
        "balance": args.balance,
        "stake_margin": args.stake,
        "leverage": args.leverage,
        "out_dir": str(out_dir),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    report = out_dir / "HC_SCORECARD_FRONTIER.md"
    write_report(report, metadata=metadata, generators=generators, frontier=frontier, components=components)

    print("\nGENERATOR STATS")
    print(generators.to_string(index=False))
    print("\nFRONTIER TOP (viable n>=20 avg_net>0)")
    show_cols = [
        "name",
        "n",
        "trades_per_24h",
        "win",
        "avg_net_pct",
        "total_net_pct",
        "pnl_usd",
        "max_drawdown_usd",
        "threshold",
        "top_per_scan",
        "max_open",
    ]
    viable_print = frontier[(frontier["n"] >= 20) & (frontier["avg_net_pct"] > 0)].copy()
    if viable_print.empty:
        print("No viable rows.")
    else:
        print(viable_print[show_cols].head(30).to_string(index=False))
    print(f"\nreport -> {report}")
    print(f"out -> {out_dir}")


if __name__ == "__main__":
    main()
