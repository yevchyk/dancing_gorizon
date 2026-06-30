"""Fit per-horizon HC probability floors on scored holdout days.

The optimizer is intentionally a research sidecar.  It answers:

    "If every horizon has its own min probability, how high should those
     thresholds be to keep roughly N signals/day while maximizing win-rate?"

It uses already-scored HC parquet files from run_hc_offgrid_sim and exact
deadline PnL from the 1m production candle tail.  It does not touch live trading.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .fast import config as FC
from .hc import config as HC
from .hc.data import to_ns
from .markets import get


OUT_DIR = C.OUTPUTS_DIR / "analysis" / "hc_offgrid" / "threshold_optimizer"
NS_PER_MIN = 60_000_000_000


@dataclass(frozen=True)
class ThresholdOption:
    threshold: float
    count: int
    wins: int
    net_sum: float


def parse_days(raw: str) -> list[str]:
    days = [p.strip().replace("-", "") for p in raw.replace(";", ",").split(",") if p.strip()]
    if not days:
        raise ValueError("--days must not be empty")
    return days


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    vals = []
    x = float(start)
    while x <= stop + step / 2:
        vals.append(round(x, 4))
        x += step
    return vals


def scored_path(analysis_root: Path, day: str, scan_stride_min: int, prob_label: str) -> Path:
    return (
        analysis_root
        / f"{day}_s{int(scan_stride_min)}_p{prob_label}_continuous10_120"
        / "hc_offgrid_scored.parquet"
    )


def load_probability_candidates(
    *,
    analysis_root: Path,
    days: list[str],
    scan_stride_min: int,
    prob_label: str,
    prob_floor: float,
    opp_cap: float,
) -> pd.DataFrame:
    cols = ["symbol", "base_time", "horizon_minutes", "up_prob", "down_prob"]
    frames: list[pd.DataFrame] = []
    for day in days:
        path = scored_path(analysis_root, day, scan_stride_min, prob_label)
        if not path.exists():
            raise FileNotFoundError(f"Missing scored parquet: {path}")
        print(f"load scored {day} -> {path}", flush=True)
        scored = pd.read_parquet(path, columns=cols)
        scored["base_time"] = pd.to_datetime(scored["base_time"], utc=True)
        for side, side_int, prob_col, opp_col in (
            ("long", 1, "up_prob", "down_prob"),
            ("short", -1, "down_prob", "up_prob"),
        ):
            mask = scored[prob_col].ge(prob_floor) & scored[opp_col].le(opp_cap)
            d = scored.loc[mask, ["symbol", "base_time", "horizon_minutes", prob_col, opp_col]].copy()
            if d.empty:
                continue
            d["day_key"] = day
            d["side"] = side_int
            d["side_name"] = side
            d["p_dir"] = d[prob_col].astype("float32")
            d["p_opp"] = d[opp_col].astype("float32")
            d["score"] = d["p_dir"].astype("float64") - d["p_opp"].astype("float64")
            frames.append(
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
                        "score",
                    ]
                ]
            )
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["horizon_minutes"] = out["horizon_minutes"].astype("int16")
    out["side"] = out["side"].astype("int8")
    return out


def attach_exact_outcomes(candidates: pd.DataFrame, *, max_stale_min: float) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    out["entry_time"] = out["base_time"] + pd.Timedelta(minutes=HC.EXEC_ENTRY_DELAY_MIN)
    out["exit_time"] = out["entry_time"] + pd.to_timedelta(out["horizon_minutes"].astype("int64"), unit="min")
    out["entry_price"] = np.nan
    out["exit_price"] = np.nan

    max_stale_ns = int(float(max_stale_min) * NS_PER_MIN)
    for idx, (symbol, part) in enumerate(out.groupby("symbol", sort=False), start=1):
        if idx == 1 or idx % 25 == 0:
            print(f"price outcomes {idx}/{out['symbol'].nunique()} {symbol}", flush=True)
        raw = get(HC.STORE_KEY).load(str(symbol))
        if raw is None or raw.empty:
            continue
        raw = raw.sort_index()
        ts = to_ns(raw.index)
        close = raw["close"].to_numpy("float64")
        for col_time, col_px in (("entry_time", "entry_price"), ("exit_time", "exit_price")):
            query = pd.DatetimeIndex(part[col_time]).to_numpy(dtype="datetime64[ns]").astype("int64")
            pos = np.searchsorted(ts, query, side="right") - 1
            ok = (pos >= 0) & ((query - ts[np.maximum(pos, 0)]) <= max_stale_ns)
            px = np.full(len(part), np.nan, dtype="float64")
            px[ok] = close[pos[ok]]
            out.loc[part.index, col_px] = px

    valid = (
        np.isfinite(out["entry_price"])
        & np.isfinite(out["exit_price"])
        & out["entry_price"].gt(0)
        & out["exit_price"].gt(0)
    )
    out = out.loc[valid].copy()
    gross = out["side"].astype("float64") * (out["exit_price"].astype("float64") / out["entry_price"].astype("float64") - 1.0)
    net = gross - float(FC.EVAL_COST)
    out["net_pnl_pct"] = net * 100.0
    out["won"] = out["net_pnl_pct"].gt(0).astype("int8")
    return out


def options_for_horizon(part: pd.DataFrame, thresholds: list[float]) -> list[ThresholdOption]:
    opts = [ThresholdOption(threshold=1.01, count=0, wins=0, net_sum=0.0)]
    for thr in thresholds:
        d = part[part["p_dir"].ge(thr)]
        if d.empty:
            opts.append(ThresholdOption(threshold=thr, count=0, wins=0, net_sum=0.0))
            continue
        opts.append(
            ThresholdOption(
                threshold=thr,
                count=int(len(d)),
                wins=int(d["won"].sum()),
                net_sum=float(d["net_pnl_pct"].sum()),
            )
        )

    # Keep only the best option for duplicate counts.
    best_by_count: dict[int, ThresholdOption] = {}
    for opt in opts:
        prev = best_by_count.get(opt.count)
        if prev is None or opt.wins > prev.wins or (opt.wins == prev.wins and opt.net_sum > prev.net_sum):
            best_by_count[opt.count] = opt
    return sorted(best_by_count.values(), key=lambda x: (x.count, x.threshold))


def optimize_thresholds(
    candidates: pd.DataFrame,
    *,
    thresholds: list[float],
    target_count: int,
    tolerance_count: int,
) -> tuple[dict[int, float], dict]:
    horizons = sorted(int(h) for h in candidates["horizon_minutes"].drop_duplicates())
    max_count = int(target_count + tolerance_count)
    min_count = max(1, int(target_count - tolerance_count))

    # count -> (wins, net_sum, choices)
    states: dict[int, tuple[int, float, dict[int, float]]] = {0: (0, 0.0, {})}
    for h in horizons:
        opts = options_for_horizon(candidates[candidates["horizon_minutes"].eq(h)], thresholds)
        new_states = dict(states)
        for count, (wins, net_sum, choices) in states.items():
            for opt in opts:
                nc = count + opt.count
                if nc > max_count:
                    continue
                nw = wins + opt.wins
                nn = net_sum + opt.net_sum
                old = new_states.get(nc)
                if old is None or nw > old[0] or (nw == old[0] and nn > old[1]):
                    ch = dict(choices)
                    ch[h] = opt.threshold
                    new_states[nc] = (nw, nn, ch)
        states = new_states

    viable = []
    for count, (wins, net_sum, choices) in states.items():
        if min_count <= count <= max_count and count > 0:
            viable.append(
                {
                    "count": count,
                    "wins": wins,
                    "win_rate": wins / count,
                    "net_sum": net_sum,
                    "avg_net": net_sum / count,
                    "choices": choices,
                }
            )
    if not viable:
        raise RuntimeError(f"No threshold combination reached target range {min_count}..{max_count}")

    viable.sort(
        key=lambda r: (
            r["win_rate"],
            -abs(r["count"] - target_count),
            r["avg_net"],
            r["net_sum"],
        ),
        reverse=True,
    )
    best = viable[0]
    return {int(h): float(t) for h, t in best["choices"].items()}, {
        k: v for k, v in best.items() if k != "choices"
    } | {"min_count": min_count, "max_count": max_count, "horizons": len(horizons)}


def apply_thresholds(candidates: pd.DataFrame, thresholds: dict[int, float]) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    thr = candidates["horizon_minutes"].map(lambda h: thresholds.get(int(h), 1.01)).astype("float64")
    return candidates[candidates["p_dir"].astype("float64").ge(thr)].copy()


def summarize(label: str, d: pd.DataFrame, n_days: int) -> dict:
    if d.empty:
        return {
            "mode": label,
            "signals": 0,
            "signals_per_day": 0.0,
            "win_pct": np.nan,
            "avg_net_pct": np.nan,
            "total_net_pct": 0.0,
            "symbols": 0,
            "days": 0,
        }
    return {
        "mode": label,
        "signals": int(len(d)),
        "signals_per_day": float(len(d) / n_days),
        "win_pct": float(d["won"].mean() * 100.0),
        "avg_net_pct": float(d["net_pnl_pct"].mean()),
        "total_net_pct": float(d["net_pnl_pct"].sum()),
        "symbols": int(d["symbol"].nunique()),
        "days": int(d["day_key"].nunique()),
    }


def horizon_table(selected: pd.DataFrame, thresholds: dict[int, float]) -> pd.DataFrame:
    rows = []
    for h in sorted(thresholds):
        d = selected[selected["horizon_minutes"].eq(h)]
        rows.append(
            {
                "horizon": h,
                "threshold": thresholds[h],
                "signals": int(len(d)),
                "signals_per_day": float(len(d) / max(1, selected["day_key"].nunique())) if len(selected) else 0.0,
                "win_pct": float(d["won"].mean() * 100.0) if len(d) else np.nan,
                "avg_net_pct": float(d["net_pnl_pct"].mean()) if len(d) else np.nan,
                "total_net_pct": float(d["net_pnl_pct"].sum()) if len(d) else 0.0,
                "avg_prob": float(d["p_dir"].mean()) if len(d) else np.nan,
                "symbols": int(d["symbol"].nunique()) if len(d) else 0,
            }
        )
    return pd.DataFrame(rows)


def symbol_time_dedup(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    d = selected.sort_values(["base_time", "symbol", "score"], ascending=[True, True, False]).copy()
    return d.drop_duplicates(["base_time", "symbol"], keep="first")


def scan_cap(selected: pd.DataFrame, max_per_scan: int) -> pd.DataFrame:
    """Keep the best N signals at each timestamp, allowing duplicate symbols.

    This is the research mode the owner asked for: if several good horizons pass
    on the same index, keep them; if the burst is too large, cap it.
    """
    if selected.empty:
        return selected.copy()
    cap = int(max_per_scan)
    if cap <= 0:
        return selected.copy()
    d = selected.sort_values(
        ["day_key", "base_time", "score", "p_dir"],
        ascending=[True, True, False, False],
    ).copy()
    return d.groupby(["day_key", "base_time"], sort=False, group_keys=False).head(cap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="20260601,20260602,20260603,20260604")
    ap.add_argument("--analysis-root", type=Path, default=C.OUTPUTS_DIR / "analysis" / "hc_offgrid")
    ap.add_argument("--scan-stride-min", type=int, default=5)
    ap.add_argument("--prob-label", default="88")
    ap.add_argument("--prob-floor", type=float, default=0.88)
    ap.add_argument("--prob-max", type=float, default=0.995)
    ap.add_argument("--prob-step", type=float, default=0.005)
    ap.add_argument("--opp-cap", type=float, default=0.20)
    ap.add_argument("--target-per-day", type=int, default=200)
    ap.add_argument("--tolerance-per-day", type=int, default=25)
    ap.add_argument("--max-per-scan", type=int, default=12)
    ap.add_argument("--price-max-stale-min", type=float, default=2.0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    days = parse_days(args.days)
    thresholds = threshold_grid(args.prob_floor, args.prob_max, args.prob_step)
    target = int(args.target_per_day) * len(days)
    tolerance = int(args.tolerance_per_day) * len(days)

    raw = load_probability_candidates(
        analysis_root=args.analysis_root,
        days=days,
        scan_stride_min=args.scan_stride_min,
        prob_label=args.prob_label,
        prob_floor=args.prob_floor,
        opp_cap=args.opp_cap,
    )
    if raw.empty:
        raise RuntimeError("No probability candidates found")
    print(f"probability candidates={len(raw)} symbols={raw['symbol'].nunique()} horizons={raw['horizon_minutes'].nunique()}", flush=True)

    outcomes = attach_exact_outcomes(raw, max_stale_min=args.price_max_stale_min)
    print(f"outcome candidates={len(outcomes)} valid_price_symbols={outcomes['symbol'].nunique()}", flush=True)

    per_h_thresholds, opt = optimize_thresholds(
        outcomes,
        thresholds=thresholds,
        target_count=target,
        tolerance_count=tolerance,
    )
    selected = apply_thresholds(outcomes, per_h_thresholds)
    capped = scan_cap(selected, args.max_per_scan)
    dedup = symbol_time_dedup(selected)

    summary = pd.DataFrame(
        [
            summarize("raw_independent", selected, len(days)),
            summarize(f"scan_cap_top{args.max_per_scan}", capped, len(days)),
            summarize("symbol_time_dedup", dedup, len(days)),
        ]
    )
    hstats = horizon_table(selected, per_h_thresholds)
    hstats_capped = horizon_table(capped, per_h_thresholds)
    active = hstats[hstats["signals"].gt(0)].copy()
    active_capped = hstats_capped[hstats_capped["signals"].gt(0)].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_parquet(args.out_dir / "candidate_outcomes.parquet", index=False)
    selected.to_parquet(args.out_dir / "selected_raw_independent.parquet", index=False)
    capped.to_parquet(args.out_dir / f"selected_scan_cap_top{args.max_per_scan}.parquet", index=False)
    dedup.to_parquet(args.out_dir / "selected_symbol_time_dedup.parquet", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    hstats.to_csv(args.out_dir / "per_horizon_thresholds.csv", index=False)
    hstats_capped.to_csv(args.out_dir / f"per_horizon_thresholds_scan_cap_top{args.max_per_scan}.csv", index=False)
    pd.DataFrame([opt]).to_csv(args.out_dir / "optimizer_choice.csv", index=False)

    print("\nOPTIMIZER")
    print(pd.DataFrame([opt]).to_string(index=False, formatters={
        "win_rate": "{:.3f}".format,
        "avg_net": "{:+.4f}".format,
        "net_sum": "{:+.2f}".format,
    }))
    print("\nSUMMARY")
    print(summary.to_string(index=False, formatters={
        "signals_per_day": "{:.1f}".format,
        "win_pct": "{:.1f}%".format,
        "avg_net_pct": "{:+.2f}%".format,
        "total_net_pct": "{:+.1f}%".format,
    }))
    print("\nACTIVE HORIZON THRESHOLDS")
    print(active.to_string(index=False, formatters={
        "threshold": "{:.3f}".format,
        "signals_per_day": "{:.1f}".format,
        "win_pct": lambda x: "" if pd.isna(x) else f"{x:.1f}%",
        "avg_net_pct": lambda x: "" if pd.isna(x) else f"{x:+.2f}%",
        "total_net_pct": "{:+.1f}%".format,
        "avg_prob": lambda x: "" if pd.isna(x) else f"{x:.3f}",
    }))
    print(f"\nACTIVE HORIZONS AFTER SCAN CAP TOP{args.max_per_scan}")
    print(active_capped.to_string(index=False, formatters={
        "threshold": "{:.3f}".format,
        "signals_per_day": "{:.1f}".format,
        "win_pct": lambda x: "" if pd.isna(x) else f"{x:.1f}%",
        "avg_net_pct": lambda x: "" if pd.isna(x) else f"{x:+.2f}%",
        "total_net_pct": "{:+.1f}%".format,
        "avg_prob": lambda x: "" if pd.isna(x) else f"{x:.3f}",
    }))
    print(f"\nout -> {args.out_dir}")


if __name__ == "__main__":
    main()
