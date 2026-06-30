"""Compare candidate test engines built from standard_* and fast_v2_* signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fast import config as FC
from .run_combined_signal_math import add_vote_columns
from .run_strictness_index_sweep import add_votes as add_worthy_votes

OUT = FC.FAST_ANALYSIS_DIR / "test_engines"
GRID = FC.FAST_ANALYSIS_DIR / "combined_signal_math" / "combined_signal_grid.parquet"
EXIT_MIN = {"2m": 2, "5m": 5, "8m": 8, "10m": 10}


def candidates(g: pd.DataFrame, mask, side, exit_h: str, engine: str,
               score_col: str, family: str = "") -> pd.DataFrame:
    d = g[mask].copy()
    if d.empty:
        return pd.DataFrame()
    if np.isscalar(side):
        d["side"] = int(side)
    else:
        d["side"] = np.asarray(side)[mask]
    d["exit"] = exit_h
    d["engine"] = engine
    d["family"] = family
    d["score"] = d[score_col].astype(float) if score_col in d else 0.0
    ret = d[f"real_ret_{exit_h}"].astype(float).to_numpy()
    side_arr = d["side"].to_numpy(float)
    d["pnl"] = side_arr * ret - FC.EVAL_COST
    d["dir_correct"] = (side_arr * ret > 0).astype(int)
    d["event_hit"] = (side_arr * ret > FC.TARGET_EDGE).astype(int)
    d["touch_green"] = np.where(
        side_arr == 1,
        d[f"real_mfe_{exit_h}"].astype(float).to_numpy() > FC.EVAL_COST,
        -d[f"real_mae_{exit_h}"].astype(float).to_numpy() > FC.EVAL_COST,
    ).astype(int)
    return d[[
        "engine", "family", "symbol", "anchor_time", "day", "side", "exit",
        "score", "pnl", "dir_correct", "event_hit", "touch_green",
    ]]


def stat(d: pd.DataFrame) -> dict:
    if len(d) == 0:
        return {
            "n": 0, "signals_per_24h": 0.0, "win": np.nan, "avg_pnl": np.nan,
            "median_pnl": np.nan, "touch_green": np.nan, "dir_correct": np.nan,
            "event_hit": np.nan, "green_days": 0, "days": 0, "total": 0.0,
            "symbols": 0, "avg_score": np.nan,
        }
    t = pd.to_datetime(d["anchor_time"], utc=True)
    hours = max(1.0, (t.max() - t.min()).total_seconds() / 3600)
    if "day" in d.columns:
        daily = d.groupby("day")["pnl"].mean() * 100
    else:
        daily = pd.Series([d["pnl"].mean() * 100])
    return {
        "n": int(len(d)),
        "signals_per_24h": float(len(d) / hours * 24.0),
        "win": float((d["pnl"] > 0).mean()),
        "avg_pnl": float(d["pnl"].mean() * 100),
        "median_pnl": float(d["pnl"].median() * 100),
        "touch_green": float(d["touch_green"].mean()),
        "dir_correct": float(d["dir_correct"].mean()),
        "event_hit": float(d["event_hit"].mean()),
        "green_days": int((daily > 0).sum()),
        "days": int(len(daily)),
        "total": float(d["pnl"].sum() * 100),
        "symbols": int(d["symbol"].nunique()),
        "avg_score": float(d["score"].mean()),
    }


def live_like(d: pd.DataFrame, *, max_open: int = 10, cooldown_min: int = 30) -> pd.DataFrame:
    if d.empty:
        return d.copy()
    x = d.sort_values(["anchor_time", "score"], ascending=[True, False]).copy()
    open_pos: list[tuple[pd.Timestamp, str]] = []
    last_open: dict[str, pd.Timestamp] = {}
    picked = []
    for _, row in x.iterrows():
        now = pd.Timestamp(row["anchor_time"])
        open_pos = [(et, sym) for et, sym in open_pos if et > now]
        if len(open_pos) >= max_open:
            continue
        sym = row["symbol"]
        if any(sym == s for _, s in open_pos):
            continue
        prev = last_open.get(sym)
        if prev is not None and now < prev + pd.Timedelta(minutes=cooldown_min):
            continue
        picked.append(row)
        last_open[sym] = now
        open_pos.append((now + pd.Timedelta(minutes=EXIT_MIN[row["exit"]]), sym))
    return pd.DataFrame(picked)


def summarize(all_cands: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    live_rows = []
    daily_rows = []
    for name, d in all_cands.items():
        r = stat(d)
        r.update({"engine": name, "mode": "raw"})
        rows.append(r)
        for cd in (0, 30, 90):
            ex = live_like(d, max_open=10, cooldown_min=cd)
            rr = stat(ex)
            rr.update({"engine": name, "mode": f"live_cap10_cd{cd}"})
            live_rows.append(rr)
        if len(d):
            daily = d.groupby("day").apply(lambda x: pd.Series(stat(x))).reset_index()
            daily["engine"] = name
            daily_rows.append(daily)
    summary = pd.DataFrame(rows).sort_values(["avg_pnl", "n"], ascending=[False, False])
    live_summary = pd.DataFrame(live_rows).sort_values(["avg_pnl", "n"], ascending=[False, False])
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return summary, live_summary, daily


def build_engines(grid: pd.DataFrame) -> dict[str, pd.DataFrame]:
    engines: dict[str, pd.DataFrame] = {}

    g082 = add_vote_columns(grid, 0.82)
    g090 = add_vote_columns(grid, 0.90)

    for exit_h in ("5m", "8m", "10m"):
        engines[f"DownCrash4_clean_std082_exit{exit_h}"] = candidates(
            g082, (g082["all_down_count"] >= 4) & (g082["all_up_count"] == 0),
            -1, exit_h, f"DownCrash4_clean_std082_exit{exit_h}", "all_down_score", "combined",
        )
        engines[f"UpImpulse3_clean_std090_exit{exit_h}"] = candidates(
            g090, (g090["all_up_count"] >= 3) & (g090["all_down_count"] == 0),
            1, exit_h, f"UpImpulse3_clean_std090_exit{exit_h}", "all_up_score", "combined",
        )

        any090 = (g090["all_up_count"] + g090["all_down_count"]) > 0
        d = g090[any090].copy()
        for k in (20, 50, 100):
            z = d.copy()
            z["_rank"] = z.groupby("day")["all_abs_score"].rank(ascending=False, method="first")
            mask_idx = z["_rank"] <= k
            engines[f"ResearchTop{k}Day_combined_std090_exit{exit_h}"] = candidates(
                z, mask_idx, z["all_side"].to_numpy(), exit_h,
                f"ResearchTop{k}Day_combined_std090_exit{exit_h}", "all_abs_score", "combined",
            )

        for k in (1, 3, 5):
            z = d.copy()
            z["_rank"] = z.groupby("anchor_time")["all_abs_score"].rank(ascending=False, method="first")
            mask_idx = z["_rank"] <= k
            engines[f"LiveTop{k}Scan_combined_std090_exit{exit_h}"] = candidates(
                z, mask_idx, z["all_side"].to_numpy(), exit_h,
                f"LiveTop{k}Scan_combined_std090_exit{exit_h}", "all_abs_score", "combined",
            )

    for idx in (0.00, 0.05, 0.10, 0.20, 0.30, 0.40):
        w, _ = add_worthy_votes(grid, idx)
        for exit_h in ("5m", "8m", "10m"):
            engines[f"PulseClean2_idx{idx:.2f}_exit{exit_h}"] = candidates(
                w, (w["side_count"] >= 2) & (w["opp_count"] == 0),
                w["side"].to_numpy(), exit_h,
                f"PulseClean2_idx{idx:.2f}_exit{exit_h}", "abs_score", "fast_v2_worthy",
            )
            engines[f"PulseClean3_idx{idx:.2f}_exit{exit_h}"] = candidates(
                w, (w["side_count"] >= 3) & (w["opp_count"] == 0),
                w["side"].to_numpy(), exit_h,
                f"PulseClean3_idx{idx:.2f}_exit{exit_h}", "abs_score", "fast_v2_worthy",
            )
            z = w[w["any_signal"]].copy()
            if len(z):
                z["_rank"] = z.groupby("day")["abs_score"].rank(ascending=False, method="first")
            for k in (20, 50):
                mask_idx = z["_rank"] <= k if len(z) else []
                engines[f"PulseTop{k}Day_idx{idx:.2f}_exit{exit_h}"] = candidates(
                    z, mask_idx, z["side"].to_numpy(), exit_h,
                    f"PulseTop{k}Day_idx{idx:.2f}_exit{exit_h}", "abs_score", "fast_v2_worthy",
                )

    # Portfolio variants: union of regimes, de-duplicated by symbol/time/side.
    def union_engine(name: str, parts: list[pd.DataFrame]) -> None:
        d = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
        if d.empty:
            engines[name] = d
            return
        d = d.sort_values("score", ascending=False)
        d = d.drop_duplicates(["symbol", "anchor_time", "side"], keep="first")
        # If opposite sides fire at same symbol/time, keep stronger score.
        d = d.sort_values("score", ascending=False).drop_duplicates(["symbol", "anchor_time"], keep="first")
        engines[name] = d.sort_values(["anchor_time", "score"], ascending=[True, False])

    union_engine("Portfolio_A_down4_up3_exit10m", [
        engines["DownCrash4_clean_std082_exit10m"],
        engines["UpImpulse3_clean_std090_exit10m"],
    ])
    union_engine("Portfolio_B_A_plus_PulseClean2_idx020_exit10m", [
        engines["DownCrash4_clean_std082_exit10m"],
        engines["UpImpulse3_clean_std090_exit10m"],
        engines["PulseClean2_idx0.20_exit10m"],
    ])
    union_engine("Portfolio_C_A_plus_PulseClean2_idx030_exit10m", [
        engines["DownCrash4_clean_std082_exit10m"],
        engines["UpImpulse3_clean_std090_exit10m"],
        engines["PulseClean2_idx0.30_exit10m"],
    ])
    union_engine("Portfolio_D_A_plus_ResearchTop20_std090_exit10m", [
        engines["DownCrash4_clean_std082_exit10m"],
        engines["UpImpulse3_clean_std090_exit10m"],
        engines["ResearchTop20Day_combined_std090_exit10m"],
    ])

    return engines


def print_report(summary: pd.DataFrame, live_summary: pd.DataFrame) -> None:
    print("=== RAW ENGINE COMPARISON: BEST n>=20 days>=3 ===")
    show = summary[(summary["n"] >= 20) & (summary["days"] >= 3)].head(45)
    print(show[["engine", "n", "signals_per_24h", "win", "avg_pnl", "touch_green",
                "green_days", "days", "symbols", "total", "avg_score"]].to_string(index=False, formatters={
        "signals_per_24h": "{:.1f}".format,
        "win": "{:.3f}".format,
        "avg_pnl": "{:+.4f}".format,
        "touch_green": "{:.3f}".format,
        "total": "{:+.2f}".format,
        "avg_score": "{:.3f}".format,
    }))

    print("\n=== LIVE-LIKE CAP10 COMPARISON: BEST n>=20 days>=3 ===")
    live = live_summary[(live_summary["n"] >= 20) & (live_summary["days"] >= 3)].head(35)
    print(live[["engine", "mode", "n", "signals_per_24h", "win", "avg_pnl", "touch_green",
                "green_days", "days", "symbols", "total"]].to_string(index=False, formatters={
        "signals_per_24h": "{:.1f}".format,
        "win": "{:.3f}".format,
        "avg_pnl": "{:+.4f}".format,
        "touch_green": "{:.3f}".format,
        "total": "{:+.2f}".format,
    }))

    print("\n=== USER-FAVORITE ENGINES ===")
    fav_names = [
        "DownCrash4_clean_std082_exit10m",
        "UpImpulse3_clean_std090_exit10m",
        "ResearchTop20Day_combined_std090_exit10m",
        "LiveTop3Scan_combined_std090_exit10m",
        "PulseClean2_idx0.20_exit10m",
        "PulseClean2_idx0.30_exit10m",
        "PulseClean3_idx0.05_exit10m",
        "Portfolio_A_down4_up3_exit10m",
        "Portfolio_B_A_plus_PulseClean2_idx020_exit10m",
        "Portfolio_C_A_plus_PulseClean2_idx030_exit10m",
    ]
    fav = summary[summary["engine"].isin(fav_names)]
    print(fav[["engine", "n", "signals_per_24h", "win", "avg_pnl", "touch_green",
               "green_days", "days", "symbols", "total"]].to_string(index=False, formatters={
        "signals_per_24h": "{:.1f}".format,
        "win": "{:.3f}".format,
        "avg_pnl": "{:+.4f}".format,
        "touch_green": "{:.3f}".format,
        "total": "{:+.2f}".format,
    }))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID)
    grid["anchor_time"] = pd.to_datetime(grid["anchor_time"], utc=True)
    engines = build_engines(grid)
    summary, live_summary, daily = summarize(engines)
    summary.to_csv(OUT / "test_engine_summary.csv", index=False)
    live_summary.to_csv(OUT / "test_engine_live_like_summary.csv", index=False)
    daily.to_csv(OUT / "test_engine_daily.csv", index=False)
    print_report(summary, live_summary)
    print(f"\nreports -> {OUT}")


if __name__ == "__main__":
    main()
