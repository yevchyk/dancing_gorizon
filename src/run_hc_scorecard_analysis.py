"""HC scorecard discovery on a clean OLD-model OOS window.

Builds a frozen leg-level table, then emits Stage 1/2 diagnostics:

* univariate lift tables by feature bin;
* RAW-adjusted incremental lift;
* generator/feature overlap and stats.

This is intentionally a research sidecar.  It does not change live trading.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dh import config as cfg
from dh import data, models, sim
from src import config as C
from src.fast import config as FC
from src.hc import config as HC
from src.hc.data import read_json_symbols
from src.run_hc_horizon_threshold_optimizer import attach_exact_outcomes


OUT_DIR = cfg.ROOT / "outputs" / "analysis" / "hc_scorecard"
KEYS = ["symbol", "base_time", "horizon_minutes"]
TEMPORAL_LAGS_MIN = (10, 20, 30)


def _label_pct(value: float) -> str:
    return f"{int(round(float(value) * 100)):02d}"


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
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


def _symbols(include_hc_blacklist: bool, max_symbols: int | None) -> list[str]:
    if include_hc_blacklist:
        syms = [s for s in read_json_symbols() if s not in set(C.BLACKLIST_SYMBOLS)]
    else:
        syms = data.universe(drop_blacklist=True)
    syms = sorted(syms)
    return syms[: int(max_symbols)] if max_symbols else syms


def build_scored_rows(
    *,
    date: str,
    days: float,
    horizons: tuple[int, ...],
    symbols: list[str],
    model_name: str,
) -> pd.DataFrame:
    entries = sim.date_grid(date, days)
    print(
        f"build features: symbols={len(symbols)} scans={len(entries)} "
        f"horizons={','.join(str(h) for h in horizons)}",
        flush=True,
    )
    feats = data.build_features(symbols, entries, horizons, cfg.ENTRY_DELAY_MIN)
    if feats.empty:
        raise RuntimeError("No feature rows built")
    print(
        f"features rows={len(feats)} symbols={feats['symbol'].nunique()} "
        f"base={feats['base_time'].min()} -> {feats['base_time'].max()}",
        flush=True,
    )
    scored = models.score(feats, model_name)
    keep = [*KEYS, "up_prob", "down_prob"]
    return scored[keep].copy()


def make_leg_table(
    scored: pd.DataFrame,
    *,
    cost_pct: float,
    candidate_floor: float,
    max_stale_min: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for side_name, side_int, p_col, opp_col in (
        ("long", 1, "up_prob", "down_prob"),
        ("short", -1, "down_prob", "up_prob"),
    ):
        d = scored[[*KEYS, "up_prob", "down_prob"]].copy()
        d["side_name"] = side_name
        d["side"] = side_int
        d["p_dir"] = d[p_col].astype("float32")
        d["p_opp"] = d[opp_col].astype("float32")
        d["spread"] = (d["p_dir"] - d["p_opp"]).astype("float32")
        d = d[d["p_dir"].ge(float(candidate_floor))].copy()
        parts.append(d)
    legs = pd.concat(parts, ignore_index=True)
    print(f"legs before outcomes={len(legs)} candidate_floor={candidate_floor}", flush=True)
    legs = attach_exact_outcomes(legs, max_stale_min=max_stale_min)
    if legs.empty:
        raise RuntimeError("No legs with exact outcomes")

    gross_side_pct = legs["net_pnl_pct"].astype("float64") + float(FC.EVAL_COST) * 100.0
    legs["gross_pnl_pct"] = gross_side_pct.astype("float32")
    legs["net_pnl_pct"] = (gross_side_pct - float(cost_pct)).astype("float32")
    legs["won"] = legs["net_pnl_pct"].gt(0).astype("int8")
    print(
        f"legs with outcomes={len(legs)} win={legs['won'].mean():.3f} "
        f"avg_net={legs['net_pnl_pct'].mean():+.4f} total_net={legs['net_pnl_pct'].sum():+.1f}",
        flush=True,
    )
    return legs


def make_signal_table(scored: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for side_name, side_int, p_col, opp_col in (
        ("long", 1, "up_prob", "down_prob"),
        ("short", -1, "down_prob", "up_prob"),
    ):
        d = scored[[*KEYS]].copy()
        d["side_name"] = side_name
        d["side"] = side_int
        d["p_dir"] = scored[p_col].astype("float32")
        d["p_opp"] = scored[opp_col].astype("float32")
        d["spread"] = (d["p_dir"] - d["p_opp"]).astype("float32")
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def add_horizon_features(legs: pd.DataFrame) -> pd.DataFrame:
    d = legs.copy()
    g = d.groupby(["symbol", "base_time", "side"], sort=False)
    d["horizon_mean_prob"] = g["p_dir"].transform("mean").astype("float32")
    d["horizon_max_prob"] = g["p_dir"].transform("max").astype("float32")
    d["horizon_mean_spread"] = g["spread"].transform("mean").astype("float32")
    d["horizon_max_spread"] = g["spread"].transform("max").astype("float32")
    d["horizon_prob_std"] = g["p_dir"].transform("std").fillna(0.0).astype("float32")
    d["horizon_spread_std"] = g["spread"].transform("std").fillna(0.0).astype("float32")
    d["horizon_prob_count_070"] = g["p_dir"].transform(lambda s: int((s >= 0.70).sum())).astype("int8")
    d["horizon_prob_count_085"] = g["p_dir"].transform(lambda s: int((s >= 0.85).sum())).astype("int8")
    return d


def add_temporal_features(legs: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    d = legs.copy()
    merge_keys = ["symbol", "base_time", "horizon_minutes", "side"]
    base = signals[[*merge_keys, "p_dir", "spread"]].copy()
    for lag in TEMPORAL_LAGS_MIN:
        lagged = base.rename(columns={"p_dir": f"p_dir_lag{lag}", "spread": f"spread_lag{lag}"})
        lagged["base_time"] = lagged["base_time"] + pd.Timedelta(minutes=lag)
        d = d.merge(lagged, on=merge_keys, how="left")

    prob_cols = ["p_dir", *[f"p_dir_lag{x}" for x in TEMPORAL_LAGS_MIN]]
    spread_cols = ["spread", *[f"spread_lag{x}" for x in TEMPORAL_LAGS_MIN]]
    lag_cols = [f"p_dir_lag{x}" for x in TEMPORAL_LAGS_MIN]
    complete = d[lag_cols].notna().all(axis=1)
    d["temporal_lag_count"] = d[lag_cols].notna().sum(axis=1).astype("int8")
    d["temporal_prob_count_070"] = sum(d[c].ge(0.70).fillna(False).astype("int8") for c in prob_cols)
    d["temporal_prob_count_085"] = sum(d[c].ge(0.85).fillna(False).astype("int8") for c in prob_cols)
    d["temporal_spread_count_050"] = sum(d[c].ge(0.50).fillna(False).astype("int8") for c in spread_cols)
    d["temporal_spread_count_070"] = sum(d[c].ge(0.70).fillna(False).astype("int8") for c in spread_cols)
    d["temporal_prob_min"] = np.where(complete, d[prob_cols].min(axis=1, skipna=False), np.nan).astype("float32")
    d["temporal_spread_min"] = np.where(complete, d[spread_cols].min(axis=1, skipna=False), np.nan).astype("float32")
    d["temporal_prob_slope_30m"] = (d["p_dir"] - d["p_dir_lag30"]).astype("float32")
    d["temporal_spread_slope_30m"] = (d["spread"] - d["spread_lag30"]).astype("float32")
    return d


def add_regime_features(legs: pd.DataFrame) -> pd.DataFrame:
    d = legs.copy()
    by_scan = (
        d.groupby("base_time", sort=False)
        .agg(
            regime_legs=("symbol", "size"),
            regime_symbols=("symbol", "nunique"),
            regime_tail085=("p_dir", lambda s: int((s >= 0.85).sum())),
            regime_tail090=("p_dir", lambda s: int((s >= 0.90).sum())),
            regime_spread070=("spread", lambda s: int((s >= 0.70).sum())),
            regime_avg_p=("p_dir", "mean"),
            regime_avg_spread=("spread", "mean"),
        )
        .reset_index()
    )
    for col in ["regime_avg_p", "regime_avg_spread"]:
        by_scan[col] = by_scan[col].astype("float32")
    return d.merge(by_scan, on="base_time", how="left")


def add_derived_features(legs: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    print("derive horizon features", flush=True)
    out = add_horizon_features(legs)
    print("derive temporal features", flush=True)
    out = add_temporal_features(out, signals)
    print("derive regime features", flush=True)
    out = add_regime_features(out)
    return out


def _feature_bins(s: pd.Series, *, bins: int) -> pd.DataFrame:
    x = pd.to_numeric(s, errors="coerce")
    ok = x.notna()
    out = pd.DataFrame({"bin_id": np.full(len(x), -1, dtype="int16"), "bin": np.full(len(x), "NA", dtype=object)})
    vals = x[ok]
    nunique = vals.nunique(dropna=True)
    if nunique <= max(12, bins):
        cats = vals.astype("float64")
        uniq = sorted(cats.dropna().unique())
        id_map = {v: i for i, v in enumerate(uniq)}
        out.loc[ok, "bin_id"] = cats.map(id_map).astype("int16")
        out.loc[ok, "bin"] = cats.map(lambda v: f"{v:g}")
        return out

    q = pd.qcut(vals, q=int(bins), duplicates="drop")
    labels = {cat: f"{cat.left:.4g}..{cat.right:.4g}" for cat in q.cat.categories}
    out.loc[ok, "bin_id"] = q.cat.codes.astype("int16")
    out.loc[ok, "bin"] = q.map(labels).astype(str)
    return out


def _iv_for_bins(grouped: pd.DataFrame, *, total_good: float, total_bad: float) -> float:
    iv = 0.0
    for r in grouped.itertuples(index=False):
        good = float(r.wins) + 0.5
        bad = float(r.losses) + 0.5
        dist_good = good / (total_good + 0.5 * len(grouped))
        dist_bad = bad / (total_bad + 0.5 * len(grouped))
        woe = math.log(max(dist_good, 1e-12) / max(dist_bad, 1e-12))
        iv += (dist_good - dist_bad) * woe
    return float(iv)


def lift_tables(df: pd.DataFrame, features: list[str], *, bins: int, min_bin_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_net = float(df["net_pnl_pct"].mean())
    total_good = float(df["won"].sum())
    total_bad = float(len(df) - total_good)
    lift_rows: list[dict] = []
    summary_rows: list[dict] = []

    for feature in features:
        b = _feature_bins(df[feature], bins=bins)
        tmp = df[["won", "net_pnl_pct", "p_dir", "spread"]].copy()
        tmp["_bin_id"] = b["bin_id"].to_numpy()
        tmp["_bin"] = b["bin"].to_numpy()
        grouped = (
            tmp[tmp["_bin_id"] >= 0]
            .groupby(["_bin_id", "_bin"], sort=True)
            .agg(
                n=("won", "size"),
                wins=("won", "sum"),
                avg_net_pct=("net_pnl_pct", "mean"),
                total_net_pct=("net_pnl_pct", "sum"),
                avg_p_dir=("p_dir", "mean"),
                avg_spread=("spread", "mean"),
            )
            .reset_index()
        )
        grouped = grouped[grouped["n"] >= int(min_bin_n)].copy()
        if grouped.empty:
            continue
        grouped["losses"] = grouped["n"] - grouped["wins"]
        grouped["win_rate"] = grouped["wins"] / grouped["n"]
        ci = grouped.apply(lambda r: _wilson(int(r["wins"]), int(r["n"])), axis=1)
        grouped["win_ci_low"] = [x[0] for x in ci]
        grouped["win_ci_high"] = [x[1] for x in ci]
        grouped["lift_vs_global_pct"] = grouped["avg_net_pct"] - global_net
        iv = _iv_for_bins(grouped, total_good=total_good, total_bad=total_bad)
        grouped["feature"] = feature
        lift_rows.extend(
            grouped.rename(columns={"_bin_id": "bin_id", "_bin": "bin"})[
                [
                    "feature",
                    "bin_id",
                    "bin",
                    "n",
                    "win_rate",
                    "win_ci_low",
                    "win_ci_high",
                    "avg_net_pct",
                    "total_net_pct",
                    "lift_vs_global_pct",
                    "avg_p_dir",
                    "avg_spread",
                ]
            ].to_dict("records")
        )
        corr = df[[feature, "net_pnl_pct"]].dropna().corr(method="spearman").iloc[0, 1]
        best = grouped.sort_values("avg_net_pct", ascending=False).iloc[0]
        summary_rows.append(
            {
                "feature": feature,
                "bins": int(len(grouped)),
                "iv_win": iv,
                "spearman_net": float(corr) if np.isfinite(corr) else np.nan,
                "best_bin": str(best["_bin"]),
                "best_bin_n": int(best["n"]),
                "best_bin_win": float(best["win_rate"]),
                "best_bin_avg_net_pct": float(best["avg_net_pct"]),
                "best_bin_lift_pct": float(best["lift_vs_global_pct"]),
            }
        )

    lift = pd.DataFrame(lift_rows)
    summary_cols = [
        "feature",
        "bins",
        "iv_win",
        "spearman_net",
        "best_bin",
        "best_bin_n",
        "best_bin_win",
        "best_bin_avg_net_pct",
        "best_bin_lift_pct",
    ]
    summary = pd.DataFrame(summary_rows, columns=summary_cols)
    if not summary.empty:
        summary = summary.sort_values(["best_bin_lift_pct", "iv_win"], ascending=[False, False])
    return lift, summary


def incremental_lift(df: pd.DataFrame, features: list[str], *, bins: int) -> pd.DataFrame:
    raw_bins = _feature_bins(df["p_dir"], bins=bins)
    tmp = df[["net_pnl_pct"]].copy()
    tmp["_raw_bin"] = raw_bins["bin_id"].to_numpy()
    raw_mean = tmp.groupby("_raw_bin")["net_pnl_pct"].transform("mean")
    residual = tmp["net_pnl_pct"] - raw_mean

    rows: list[dict] = []
    for feature in features:
        if feature == "p_dir":
            continue
        x = pd.to_numeric(df[feature], errors="coerce")
        valid = x.notna()
        if valid.sum() < 100:
            continue
        corr = df[[feature, "net_pnl_pct"]].dropna().corr(method="spearman").iloc[0, 1]
        high_is_good = bool(corr >= 0) if np.isfinite(corr) else True
        q = 0.80 if high_is_good else 0.20
        cut = float(x[valid].quantile(q))
        mask = x.ge(cut) if high_is_good else x.le(cut)
        mask &= valid
        rest = valid & ~mask
        if mask.sum() < 30 or rest.sum() < 30:
            continue
        rows.append(
            {
                "feature": feature,
                "direction": "high" if high_is_good else "low",
                "cut": cut,
                "selected_n": int(mask.sum()),
                "selected_win": float(df.loc[mask, "won"].mean()),
                "selected_avg_net_pct": float(df.loc[mask, "net_pnl_pct"].mean()),
                "rest_avg_net_pct": float(df.loc[rest, "net_pnl_pct"].mean()),
                "raw_adjusted_selected_resid_pct": float(residual[mask].mean()),
                "raw_adjusted_rest_resid_pct": float(residual[rest].mean()),
                "incremental_lift_pct": float(residual[mask].mean() - residual[rest].mean()),
                "spearman_net": float(corr) if np.isfinite(corr) else np.nan,
            }
        )
    cols = [
        "feature",
        "direction",
        "cut",
        "selected_n",
        "selected_win",
        "selected_avg_net_pct",
        "rest_avg_net_pct",
        "raw_adjusted_selected_resid_pct",
        "raw_adjusted_rest_resid_pct",
        "incremental_lift_pct",
        "spearman_net",
    ]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("incremental_lift_pct", ascending=False) if not out.empty else out


def mask_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = {
        "RAW85": df["p_dir"].ge(0.85),
        "RAW90": df["p_dir"].ge(0.90),
        "SPREAD60": df["spread"].ge(0.60),
        "SPREAD70": df["spread"].ge(0.70),
        "SPREAD80": df["spread"].ge(0.80),
        "HMEAN85": df["horizon_mean_prob"].ge(0.85),
        "HMEAN90": df["horizon_mean_prob"].ge(0.90),
        "SMEAN60": df["horizon_mean_spread"].ge(0.60),
        "SMEAN70": df["horizon_mean_spread"].ge(0.70),
        "TEMP_PROB3_070": df["temporal_prob_count_070"].ge(3),
        "TEMP_PROB4_070": df["temporal_prob_count_070"].ge(4),
        "TEMP_SPREAD3_050": df["temporal_spread_count_050"].ge(3),
        "TEMP_SPREAD4_050": df["temporal_spread_count_050"].ge(4),
    }
    summary = []
    for name, mask in masks.items():
        d = df[mask]
        summary.append(
            {
                "mask": name,
                "n": int(len(d)),
                "pct_of_legs": float(len(d) / max(1, len(df))),
                "win": float(d["won"].mean()) if len(d) else np.nan,
                "avg_net_pct": float(d["net_pnl_pct"].mean()) if len(d) else np.nan,
                "total_net_pct": float(d["net_pnl_pct"].sum()) if len(d) else 0.0,
                "avg_p_dir": float(d["p_dir"].mean()) if len(d) else np.nan,
                "avg_spread": float(d["spread"].mean()) if len(d) else np.nan,
            }
        )
    names = list(masks)
    overlap = pd.DataFrame(index=names, columns=names, dtype="float64")
    for a in names:
        ma = masks[a]
        for b in names:
            mb = masks[b]
            union = int((ma | mb).sum())
            overlap.loc[a, b] = float((ma & mb).sum() / union) if union else np.nan
    return pd.DataFrame(summary).sort_values("avg_net_pct", ascending=False), overlap.reset_index(names="mask")


def write_report(
    *,
    path: Path,
    metadata: dict,
    feature_summary: pd.DataFrame,
    incremental: pd.DataFrame,
    mask_summary: pd.DataFrame,
    lift: pd.DataFrame,
) -> None:
    top_lift = (
        lift.sort_values(["lift_vs_global_pct", "n"], ascending=[False, False]).head(30)
        if not lift.empty else lift
    )
    lines = [
        "# HC Scorecard Discovery",
        "",
        f"Generated: {pd.Timestamp.now('UTC').isoformat()}",
        "",
        "## Setup",
        "",
        _md_table(pd.DataFrame([metadata]), max_rows=5),
        "",
        "## Feature Summary",
        "",
        _md_table(feature_summary, max_rows=40),
        "",
        "## RAW-Adjusted Incremental Lift",
        "",
        _md_table(incremental, max_rows=40),
        "",
        "## Generator Masks",
        "",
        _md_table(mask_summary, max_rows=40),
        "",
        "## Best Lift Bins",
        "",
        _md_table(top_lift, max_rows=30),
        "",
        "## Notes",
        "",
        "- Rows are leg-level: one `(symbol, base_time, horizon, side)` candidate.",
        "- Candidate floor keeps `p_dir >= floor`; spread generators are still preserved because high spread implies a high enough directional prob.",
        "- `net_pnl_pct` is recomputed as side gross minus configured fee+slippage cost.",
        "- Incremental lift is residualized by RAW probability bins, so it asks whether a feature adds evidence beyond raw `p_dir`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_horizons(raw: str) -> tuple[int, ...]:
    vals = [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    if not vals:
        raise ValueError("horizons must not be empty")
    return tuple(sorted(dict.fromkeys(vals)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-06-01")
    ap.add_argument("--days", type=float, default=4.0)
    ap.add_argument("--model", choices=sorted(cfg.MODELS), default="old")
    ap.add_argument("--horizons", default="30,45,60,90")
    ap.add_argument("--slip", type=float, default=cfg.SLIP_ALL)
    ap.add_argument("--candidate-floor", type=float, default=0.50)
    ap.add_argument("--max-stale-min", type=float, default=2.0)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--min-bin-n", type=int, default=50)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--symbols", default=None,
                    help="comma-separated symbols to use instead of the HC universe")
    ap.add_argument("--include-hc-blacklist", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    horizons = parse_horizons(args.horizons)
    symbols = _symbols(args.include_hc_blacklist, args.max_symbols or None)
    if args.symbols:
        symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    cost_pct = cfg.cost(args.slip)
    tag = (
        f"{args.model}_{args.date}_{args.days:g}d_h{min(horizons)}-{max(horizons)}"
        f"_p{_label_pct(args.candidate_floor)}_slip{args.slip:g}"
    ).replace(".", "p")
    out_dir = args.out_dir or (OUT_DIR / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = build_scored_rows(
        date=args.date,
        days=args.days,
        horizons=horizons,
        symbols=symbols,
        model_name=args.model,
    )
    scored.to_parquet(out_dir / "scored_rows.parquet", index=False)

    legs = make_leg_table(
        scored,
        cost_pct=cost_pct,
        candidate_floor=args.candidate_floor,
        max_stale_min=args.max_stale_min,
    )
    signals = make_signal_table(scored)
    legs = add_derived_features(legs, signals)
    legs.to_parquet(out_dir / "frozen_legs.parquet", index=False)

    features = [
        "p_dir",
        "p_opp",
        "spread",
        "horizon_mean_prob",
        "horizon_max_prob",
        "horizon_mean_spread",
        "horizon_max_spread",
        "horizon_prob_std",
        "horizon_spread_std",
        "horizon_prob_count_070",
        "horizon_prob_count_085",
        "temporal_prob_count_070",
        "temporal_prob_count_085",
        "temporal_spread_count_050",
        "temporal_spread_count_070",
        "temporal_lag_count",
        "temporal_prob_min",
        "temporal_spread_min",
        "temporal_prob_slope_30m",
        "temporal_spread_slope_30m",
        "regime_tail085",
        "regime_tail090",
        "regime_spread070",
        "regime_avg_p",
        "regime_avg_spread",
    ]
    print("lift tables", flush=True)
    lift, feature_summary = lift_tables(legs, features, bins=args.bins, min_bin_n=args.min_bin_n)
    incremental = incremental_lift(legs, features, bins=args.bins)
    mask_summary, overlap = mask_tables(legs)

    lift.to_csv(out_dir / "lift_tables.csv", index=False)
    feature_summary.to_csv(out_dir / "feature_summary.csv", index=False)
    incremental.to_csv(out_dir / "incremental_lift.csv", index=False)
    mask_summary.to_csv(out_dir / "mask_summary.csv", index=False)
    overlap.to_csv(out_dir / "mask_overlap_jaccard.csv", index=False)

    metadata = {
        "model": args.model,
        "model_cutoff": cfg.MODEL_CUTOFF[args.model],
        "date": args.date,
        "days": args.days,
        "horizons": ",".join(str(h) for h in horizons),
        "symbols": len(symbols),
        "include_hc_blacklist": args.include_hc_blacklist,
        "candidate_floor": args.candidate_floor,
        "cost_pct": cost_pct,
        "rows_scored": int(len(scored)),
        "frozen_legs": int(len(legs)),
        "global_win": float(legs["won"].mean()),
        "global_avg_net_pct": float(legs["net_pnl_pct"].mean()),
        "global_total_net_pct": float(legs["net_pnl_pct"].sum()),
        "out_dir": str(out_dir),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    report = out_dir / "HC_SCORECARD_DISCOVERY.md"
    write_report(
        path=report,
        metadata=metadata,
        feature_summary=feature_summary,
        incremental=incremental,
        mask_summary=mask_summary,
        lift=lift,
    )
    print("\nTOP FEATURE SUMMARY")
    print(feature_summary.head(20).to_string(index=False))
    print("\nTOP INCREMENTAL")
    print(incremental.head(20).to_string(index=False))
    print("\nMASK SUMMARY")
    print(mask_summary.head(20).to_string(index=False))
    print(f"\nreport -> {report}")
    print(f"out -> {out_dir}")


if __name__ == "__main__":
    main()
