"""Causal market-state gate analysis for HC scorecard legs.

This research sidecar asks a narrow question:

    can a no-trade regime gate separate an active OLD edge window from a fresh
    adverse window, using only candle-derived state available at base_time?

It intentionally does not change live trading.  Feed it scorecard legs produced
by ``run_hc_scorecard_frontier`` or frozen legs from ``run_hc_scorecard_analysis``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dh import config as cfg
from dh import data
from src.hc import config as HC
from src.markets import get
from src.run_hc_scorecard_frontier import (
    SimConfig,
    add_scorecard,
    apply_portfolio_constraints,
    preselect,
)


OUT_DIR = cfg.ROOT / "outputs" / "analysis" / "hc_regime_gate"
NS_PER_MIN = 60_000_000_000

DEFAULT_LEGS = [
    (
        "old_good_jun1_4",
        cfg.ROOT
        / "outputs"
        / "analysis"
        / "hc_scorecard_frontier"
        / "old_2026-06-01_4d_h30-90_p50_slip0p6"
        / "scorecard_legs.parquet",
    ),
    (
        "old_bad_jun5",
        cfg.ROOT
        / "outputs"
        / "analysis"
        / "hc_scorecard_frontier"
        / "old_2026-06-05_0p5d_h30-90_p50_slip0p6"
        / "scorecard_legs.parquet",
    ),
]


@dataclass(frozen=True)
class Gate:
    name: str
    mask: pd.Series


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


def _parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.parent.name, path
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"missing label in --legs {raw!r}")
    return label, Path(path.strip())


def load_legs(items: list[tuple[str, Path]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, path in items:
        if not path.exists():
            raise FileNotFoundError(path)
        d = pd.read_parquet(path)
        for col in ["base_time", "entry_time", "exit_time"]:
            if col in d.columns:
                d[col] = pd.to_datetime(d[col], utc=True)
        if "pool" not in d.columns or "score" not in d.columns:
            d = add_scorecard(d)
        d["window"] = label
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _price_at(
    ts_ns: np.ndarray,
    close: np.ndarray,
    anchors_ns: np.ndarray,
    offset_min: int,
    *,
    max_stale_min: float,
) -> np.ndarray:
    target = anchors_ns - np.int64(offset_min) * NS_PER_MIN
    idx = np.searchsorted(ts_ns, target, side="right") - 1
    out = np.full(len(anchors_ns), np.nan, dtype="float64")
    ok = idx >= 0
    if not bool(ok.any()):
        return out
    stale_min = (target[ok] - ts_ns[idx[ok]]) / NS_PER_MIN
    ok_idx = np.flatnonzero(ok)[stale_min <= float(max_stale_min)]
    out[ok_idx] = close[idx[ok_idx]]
    return out


def compute_market_state(base_times: pd.Series, *, max_stale_min: float) -> pd.DataFrame:
    anchors = pd.DatetimeIndex(pd.to_datetime(base_times.drop_duplicates().sort_values(), utc=True))
    anchors_ns = anchors.as_unit("ns").asi8
    store = get(HC.STORE_KEY)
    symbols = data.universe(drop_blacklist=True)

    returns: dict[int, list[np.ndarray]] = {15: [], 60: [], 240: []}
    btc: dict[str, np.ndarray] = {}
    loaded = 0

    for symbol in symbols:
        raw = store.load(symbol)
        if raw is None or raw.empty:
            continue
        raw = raw.sort_index()
        ts_ns = pd.DatetimeIndex(raw.index).as_unit("ns").asi8
        close = raw["close"].to_numpy("float64")
        p0 = _price_at(ts_ns, close, anchors_ns, 0, max_stale_min=max_stale_min)
        for offset in returns:
            p_prev = _price_at(ts_ns, close, anchors_ns, offset, max_stale_min=max_stale_min)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where((p0 > 0) & (p_prev > 0), p0 / p_prev - 1.0, np.nan)
            returns[offset].append(r)
        if symbol == "BTC_USDT_SWAP":
            btc["btc_r15"] = returns[15][-1] * 100.0
            btc["btc_r60"] = returns[60][-1] * 100.0
            btc["btc_r240"] = returns[240][-1] * 100.0
            logs = np.log(close)
            vol60: list[float] = []
            for anchor_ns in anchors_ns:
                lo = np.searchsorted(ts_ns, anchor_ns - np.int64(60) * NS_PER_MIN, side="right")
                hi = np.searchsorted(ts_ns, anchor_ns, side="right")
                seg = np.diff(logs[max(0, lo):hi])
                vol60.append(float(np.nanstd(seg) * 100.0) if len(seg) >= 10 else np.nan)
            btc["btc_vol60"] = np.asarray(vol60, dtype="float64")
        loaded += 1

    if not returns[60]:
        raise RuntimeError("No candle returns could be computed")

    out = pd.DataFrame({"base_time": anchors, "weather_symbols": loaded})
    for offset, chunks in returns.items():
        r = np.vstack(chunks)
        valid = np.isfinite(r)
        up = np.where(valid, r > 0.0, np.nan)
        out[f"breadth{offset}"] = np.nanmean(up, axis=0)
        out[f"together{offset}"] = np.maximum(out[f"breadth{offset}"], 1.0 - out[f"breadth{offset}"])
        out[f"mkt_ret{offset}"] = np.nanmedian(r, axis=0) * 100.0
        out[f"disp{offset}"] = np.nanstd(r, axis=0) * 100.0
    for col, values in btc.items():
        out[col] = values
    out["abs_mkt60"] = out["mkt_ret60"].abs()
    out["abs_btc60"] = out["btc_r60"].abs() if "btc_r60" in out else np.nan
    return out


def _summary(prefix: str, x: pd.DataFrame) -> dict:
    if x.empty:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_win": np.nan,
            f"{prefix}_avg_net_pct": np.nan,
            f"{prefix}_total_net_pct": 0.0,
        }
    return {
        f"{prefix}_n": int(len(x)),
        f"{prefix}_win": float(x["won"].mean()),
        f"{prefix}_avg_net_pct": float(x["net_pnl_pct"].mean()),
        f"{prefix}_total_net_pct": float(x["net_pnl_pct"].sum()),
    }


def day_summary(d: pd.DataFrame) -> pd.DataFrame:
    x = d[d["pool"]].copy()
    x["day"] = x["base_time"].dt.strftime("%Y-%m-%d")
    return (
        x.groupby(["window", "day"], sort=True)
        .agg(
            n=("symbol", "size"),
            win=("won", "mean"),
            avg_net_pct=("net_pnl_pct", "mean"),
            total_net_pct=("net_pnl_pct", "sum"),
            breadth60=("breadth60", "mean"),
            together60=("together60", "mean"),
            mkt_ret60=("mkt_ret60", "mean"),
            breadth240=("breadth240", "mean"),
            mkt_ret240=("mkt_ret240", "mean"),
            btc_r60=("btc_r60", "mean"),
            btc_r240=("btc_r240", "mean"),
            btc_vol60=("btc_vol60", "mean"),
        )
        .reset_index()
    )


def side_summary(d: pd.DataFrame) -> pd.DataFrame:
    x = d[d["pool"]].copy()
    x["day"] = x["base_time"].dt.strftime("%Y-%m-%d")
    return (
        x.groupby(["window", "day", "side_name"], sort=True)
        .agg(
            n=("symbol", "size"),
            win=("won", "mean"),
            avg_net_pct=("net_pnl_pct", "mean"),
            mkt_ret60=("mkt_ret60", "mean"),
            mkt_ret240=("mkt_ret240", "mean"),
            breadth240=("breadth240", "mean"),
        )
        .reset_index()
    )


def feature_gate_sweep(d: pd.DataFrame) -> pd.DataFrame:
    pool = d[d["pool"]].copy()
    features = [
        "breadth60",
        "together60",
        "mkt_ret60",
        "abs_mkt60",
        "disp60",
        "breadth240",
        "mkt_ret240",
        "abs_btc60",
        "btc_vol60",
    ]
    rows: list[dict] = []
    windows = sorted(pool["window"].unique())
    for feature in features:
        values = pool[feature].dropna()
        if values.empty:
            continue
        cuts = values.quantile([0.2, 0.4, 0.6, 0.8]).drop_duplicates()
        for cut in cuts:
            for op in (">=", "<="):
                keep = pool[feature].ge(cut) if op == ">=" else pool[feature].le(cut)
                row = {"gate": f"{feature}{op}{cut:.6g}", "feature": feature, "op": op, "cut": float(cut)}
                for window in windows:
                    w = pool["window"].eq(window)
                    row.update(_summary(f"{window}_keep", pool[keep & w]))
                    row.update(_summary(f"{window}_block", pool[(~keep) & w]))
                rows.append(row)
    return pd.DataFrame(rows)


def named_gates(d: pd.DataFrame) -> list[Gate]:
    return [
        Gate("none", pd.Series(True, index=d.index)),
        Gate("no_persistent_down_br240_020_m240_m150", ~(d["breadth240"].le(0.20) & d["mkt_ret240"].le(-1.50))),
        Gate("no_extreme_herd_together60_094", d["together60"].le(0.94)),
        Gate("no_btc60_abs_gt_070", d["abs_btc60"].le(0.70)),
        Gate("low_or_mid_breadth60_le_045", d["breadth60"].le(0.45)),
    ]


def profile_gate_table(d: pd.DataFrame, *, cooldown_min: int) -> pd.DataFrame:
    profiles = [
        ("quality", SimConfig(1.9518, 3, 6, cooldown_min)),
        ("balanced", SimConfig(1.1993, 6, 6, cooldown_min)),
        ("aggressive", SimConfig(0.8984, 10, 10, cooldown_min)),
    ]
    rows: list[dict] = []
    for window, wdf in d.groupby("window", sort=True):
        for profile, sim_cfg in profiles:
            candidates = preselect(wdf, sim_cfg)
            for gate in named_gates(candidates):
                if candidates.empty:
                    filtered = candidates
                else:
                    filtered = candidates[gate.mask.to_numpy()].copy()
                trades = apply_portfolio_constraints(filtered, sim_cfg)
                row = {
                    "window": window,
                    "profile": profile,
                    "gate": gate.name,
                    "threshold": sim_cfg.threshold,
                    "top_per_scan": sim_cfg.top_per_scan,
                    "max_open": sim_cfg.max_open,
                }
                row.update(_summary("trades", trades))
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["window", "profile", "trades_avg_net_pct", "trades_n"], ascending=[True, True, False, False])
    return out


def write_report(
    path: Path,
    *,
    metadata: dict,
    days: pd.DataFrame,
    sides: pd.DataFrame,
    sweeps: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    sweep_cols = [c for c in sweeps.columns if c.endswith("_avg_net_pct")]
    top_sweeps = sweeps.copy()
    if sweep_cols:
        top_sweeps["_score"] = top_sweeps[sweep_cols].mean(axis=1, skipna=True)
        top_sweeps = top_sweeps.sort_values("_score", ascending=False).drop(columns=["_score"])
    lines = [
        "# HC Regime Gate Analysis",
        "",
        f"Generated: {pd.Timestamp.now('UTC').isoformat()}",
        "",
        "## Setup",
        "",
        _md_table(pd.DataFrame([metadata]), max_rows=3),
        "",
        "## Pool By Day",
        "",
        _md_table(days, max_rows=30),
        "",
        "## Pool By Day And Side",
        "",
        _md_table(sides, max_rows=40),
        "",
        "## Named Gate Profiles",
        "",
        _md_table(profiles, max_rows=80),
        "",
        "## Feature Gate Sweep",
        "",
        _md_table(top_sweeps, max_rows=80),
        "",
        "## Notes",
        "",
        "- Market-state is computed at `base_time` only; no future candles are used.",
        "- Gates are diagnostics. A gate is only interesting if it blocks bad windows without destroying good OOS edge.",
        "- If every gate keeps a negative bad-window average, the correct action is no-trade/shadow, not score tightening.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", action="append", default=[], help="label=path to scorecard/frozen legs parquet")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-stale-min", type=float, default=6.0)
    ap.add_argument("--cooldown-min", type=int, default=30)
    args = ap.parse_args()

    items = [_parse_labeled_path(x) for x in args.legs] if args.legs else DEFAULT_LEGS
    out_dir = args.out_dir or OUT_DIR / "old_good_vs_bad"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("load legs", flush=True)
    legs = load_legs(items)
    print(f"rows={len(legs)} windows={legs['window'].nunique()} pool={int(legs['pool'].sum())}", flush=True)
    print("compute market state", flush=True)
    state = compute_market_state(legs["base_time"], max_stale_min=args.max_stale_min)
    d = legs.merge(state, on="base_time", how="left")
    d.to_parquet(out_dir / "legs_with_market_state.parquet", index=False)

    days = day_summary(d)
    sides = side_summary(d)
    sweeps = feature_gate_sweep(d)
    profiles = profile_gate_table(d, cooldown_min=args.cooldown_min)

    days.to_csv(out_dir / "pool_by_day.csv", index=False)
    sides.to_csv(out_dir / "pool_by_day_side.csv", index=False)
    sweeps.to_csv(out_dir / "feature_gate_sweep.csv", index=False)
    profiles.to_csv(out_dir / "named_gate_profiles.csv", index=False)

    metadata = {
        "legs": "; ".join(f"{label}={path}" for label, path in items),
        "rows": int(len(d)),
        "windows": int(d["window"].nunique()),
        "pool_rows": int(d["pool"].sum()),
        "weather_symbols": int(state["weather_symbols"].max()) if len(state) else 0,
        "max_stale_min": args.max_stale_min,
        "out_dir": str(out_dir),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    report = out_dir / "HC_REGIME_GATE_ANALYSIS.md"
    write_report(report, metadata=metadata, days=days, sides=sides, sweeps=sweeps, profiles=profiles)

    print("\nPOOL BY DAY")
    print(days.to_string(index=False))
    print("\nNAMED GATE PROFILES")
    print(profiles.to_string(index=False))
    print(f"\nreport -> {report}")
    print(f"out -> {out_dir}")


if __name__ == "__main__":
    main()
