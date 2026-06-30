"""Detailed untouched-holdout reports for the fast short-horizon models.

The script consumes fast_v2 holdout_scores.parquet only. That file is produced
after training on split=="train"; split=="holdout" is the untouched last 3 days.
Reports:
  1. per-model probability thresholds
  2. per-model probability bins / calibration
  3. engine-style raw confidence, clean opposite-side filter, agreement, top scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fast import config as FC


THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95)
BIN_EDGES = np.round(np.arange(0.40, 1.0001, 0.05), 2)
FLOORS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90)
OPP_LIMITS = (0.30, 0.40, 0.50)
AGREE_LEVELS = (1, 2, 3)
TOP_PER_SCAN = (1, 3, 5, 10)
TOP_PER_DAY = (20, 50, 100, 200)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            return np.nan
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return np.nan
    return float(a.corr(b, method="spearman"))


def _fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:+.4f}"


def stat(d: pd.DataFrame, *, extra: bool = False) -> dict:
    if len(d) == 0:
        row = {
            "n": 0,
            "win": np.nan,
            "event_hit": np.nan,
            "avg_pnl": np.nan,
            "median_pnl": np.nan,
            "touch_green": np.nan,
            "green_days": 0,
            "days": 0,
            "total": 0.0,
        }
        if extra:
            row.update({"avg_prob": np.nan, "avg_opp": np.nan, "avg_spread": np.nan})
        return row

    pnl = d["pnl"].astype(float)
    if "day" in d.columns:
        daily = d.groupby("day")["pnl"].mean() * 100
    else:
        daily = pd.Series([pnl.mean() * 100])
    row = {
        "n": int(len(d)),
        "win": float((pnl > 0).mean()),
        "event_hit": float(d["event_hit"].mean()) if "event_hit" in d else np.nan,
        "avg_pnl": float(pnl.mean() * 100),
        "median_pnl": float(pnl.median() * 100),
        "touch_green": float(d["touch_green"].mean()) if "touch_green" in d else np.nan,
        "green_days": int((daily > 0).sum()),
        "days": int(len(daily)),
        "total": float(pnl.sum() * 100),
    }
    if extra:
        row.update({
            "avg_prob": float(d["prob"].mean()) if "prob" in d else np.nan,
            "avg_opp": float(d["opp"].mean()) if "opp" in d else np.nan,
            "avg_spread": float(d["spread"].mean()) if "spread" in d else np.nan,
        })
    return row


def model_candidates(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in FC.HORIZONS:
        lab = h.label
        base = scores[scores["horizon"] == lab].copy()
        if base.empty:
            continue
        for side_name, side, prob_col, opp_col in (
            ("up", 1, "p_up", "p_down"),
            ("down", -1, "p_down", "p_up"),
        ):
            prob = base[prob_col].astype(float).to_numpy()
            opp = base[opp_col].astype(float).to_numpy()
            real_ret = base["real_ret"].astype(float).to_numpy()
            if side == 1:
                event_hit = real_ret > FC.TARGET_EDGE
                touch_green = base["real_mfe"].astype(float).to_numpy() > FC.EVAL_COST
            else:
                event_hit = real_ret < -FC.TARGET_EDGE
                touch_green = -base["real_mae"].astype(float).to_numpy() > FC.EVAL_COST
            rows.append(pd.DataFrame({
                "symbol": base["symbol"].to_numpy(),
                "anchor_time": pd.to_datetime(base["anchor_time"], utc=True),
                "day": base["day"].to_numpy(),
                "horizon": lab,
                "model": f"{side_name}_{lab}",
                "side_name": side_name,
                "side": side,
                "prob": prob,
                "opp": opp,
                "spread": prob - opp,
                "real_ret": real_ret,
                "real_mfe": base["real_mfe"].astype(float).to_numpy(),
                "real_mae": base["real_mae"].astype(float).to_numpy(),
                "event_hit": event_hit.astype(int),
                "touch_green": touch_green.astype(int),
                "pnl": side * real_ret - FC.EVAL_COST,
            }))
    return pd.concat(rows, ignore_index=True)


def argmax_candidates(scores: pd.DataFrame) -> pd.DataFrame:
    s = scores.copy()
    side = np.where(s["p_up"].to_numpy() >= s["p_down"].to_numpy(), 1, -1)
    prob = np.maximum(s["p_up"].to_numpy(), s["p_down"].to_numpy())
    opp = np.minimum(s["p_up"].to_numpy(), s["p_down"].to_numpy())
    side_name = np.where(side == 1, "up", "down")
    touch_green = np.where(
        side == 1,
        s["real_mfe"].astype(float).to_numpy() > FC.EVAL_COST,
        -s["real_mae"].astype(float).to_numpy() > FC.EVAL_COST,
    )
    event_hit = np.where(
        side == 1,
        s["real_ret"].astype(float).to_numpy() > FC.TARGET_EDGE,
        s["real_ret"].astype(float).to_numpy() < -FC.TARGET_EDGE,
    )
    return pd.DataFrame({
        "symbol": s["symbol"].to_numpy(),
        "anchor_time": pd.to_datetime(s["anchor_time"], utc=True),
        "day": s["day"].to_numpy(),
        "horizon": s["horizon"].to_numpy(),
        "side": side,
        "side_name": side_name,
        "model": np.char.add(np.char.add(side_name.astype(str), "_"), s["horizon"].astype(str).to_numpy()),
        "prob": prob,
        "opp": opp,
        "spread": prob - opp,
        "real_ret": s["real_ret"].astype(float).to_numpy(),
        "event_hit": event_hit.astype(int),
        "touch_green": touch_green.astype(int),
        "pnl": side * s["real_ret"].astype(float).to_numpy() - FC.EVAL_COST,
    })


def agreement_candidates(c: pd.DataFrame, floor: float, opp: float,
                         agree_min: int) -> pd.DataFrame:
    fire = c[(c["prob"] >= floor) & (c["opp"] <= opp)].copy()
    if fire.empty:
        return fire.assign(agree=pd.Series(dtype="int64"))
    grp_cols = ["symbol", "anchor_time", "side"]
    fire["agree"] = fire.groupby(grp_cols)["horizon"].transform("nunique")
    fire = fire[fire["agree"] >= agree_min].copy()
    if fire.empty:
        return fire
    idx = fire.groupby(grp_cols)["spread"].idxmax()
    best = fire.loc[idx].copy()
    return best.sort_values(["anchor_time", "spread"], ascending=[True, False])


def per_model_reports(c: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    summary_rows = []
    threshold_rows = []
    bin_rows = []

    for model, d in c.groupby("model", sort=True):
        summary_rows.append({
            "model": model,
            "horizon": d["horizon"].iloc[0],
            "side": d["side_name"].iloc[0],
            "n": int(len(d)),
            "base_event_rate": float(d["event_hit"].mean()),
            "base_trade_win": float((d["pnl"] > 0).mean()),
            "base_avg_pnl": float(d["pnl"].mean() * 100),
            "auc_event": _auc(d["event_hit"].to_numpy(), d["prob"].to_numpy()),
            "spearman_event": _spearman(d["prob"], d["event_hit"]),
            "spearman_pnl": _spearman(d["prob"], d["pnl"]),
            "p80": float(d["prob"].quantile(0.80)),
            "p90": float(d["prob"].quantile(0.90)),
            "p95": float(d["prob"].quantile(0.95)),
        })
        for thr in THRESHOLDS:
            z = d[d["prob"] >= thr]
            r = stat(z, extra=True)
            r.update({
                "model": model,
                "horizon": d["horizon"].iloc[0],
                "side": d["side_name"].iloc[0],
                "threshold": thr,
                "coverage": float(len(z) / len(d)) if len(d) else 0.0,
            })
            threshold_rows.append(r)

        for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            z = d[(d["prob"] >= lo) & (d["prob"] < hi if hi < 1.0 else d["prob"] <= 1.0)]
            r = stat(z, extra=True)
            r.update({
                "model": model,
                "horizon": d["horizon"].iloc[0],
                "side": d["side_name"].iloc[0],
                "bin_lo": lo,
                "bin_hi": hi,
                "bin": f"[{lo:.2f},{hi:.2f}{')' if hi < 1.0 else ']'}",
            })
            bin_rows.append(r)

    summary = pd.DataFrame(summary_rows).sort_values(["horizon", "side"])
    thresholds = pd.DataFrame(threshold_rows).sort_values(["model", "threshold"])
    bins = pd.DataFrame(bin_rows).sort_values(["model", "bin_lo"])

    summary.to_csv(out_dir / "model_summary.csv", index=False)
    thresholds.to_csv(out_dir / "model_probability_thresholds.csv", index=False)
    bins.to_csv(out_dir / "model_probability_bins.csv", index=False)
    return {"summary": summary, "thresholds": thresholds, "bins": bins}


def engine_reports(c: pd.DataFrame, argmax: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    raw_rows = []
    for thr in THRESHOLDS:
        d = argmax[argmax["prob"] >= thr]
        r = stat(d, extra=True)
        r.update({"strategy": "raw_argmax", "threshold": thr})
        raw_rows.append(r)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out_dir / "engine_raw_argmax_thresholds.csv", index=False)

    clean_rows = []
    top_scan_rows = []
    top_day_rows = []
    by_model_rows = []
    daily_rows = []
    for floor in FLOORS:
        for opp in OPP_LIMITS:
            for agree in AGREE_LEVELS:
                d = agreement_candidates(c, floor, opp, agree)
                r = stat(d, extra=True)
                r.update({"floor": floor, "opp_max": opp, "agree_min": agree, "selection": "all_agreement"})
                clean_rows.append(r)

                if len(d):
                    by_model = d.groupby("model").apply(lambda x: pd.Series(stat(x, extra=True))).reset_index()
                    by_model["floor"] = floor
                    by_model["opp_max"] = opp
                    by_model["agree_min"] = agree
                    by_model_rows.append(by_model)

                    daily = d.groupby("day").apply(lambda x: pd.Series(stat(x, extra=True))).reset_index()
                    daily["floor"] = floor
                    daily["opp_max"] = opp
                    daily["agree_min"] = agree
                    daily_rows.append(daily)

                for k in TOP_PER_SCAN:
                    z = d.copy()
                    if len(z):
                        z["_rank"] = z.groupby("anchor_time")["spread"].rank(ascending=False, method="first")
                        z = z[z["_rank"] <= k]
                    r = stat(z, extra=True)
                    r.update({"floor": floor, "opp_max": opp, "agree_min": agree,
                              "top_per_scan": k, "selection": f"top{k}/scan"})
                    top_scan_rows.append(r)

                for k in TOP_PER_DAY:
                    z = d.copy()
                    if len(z):
                        z["_rank"] = z.groupby("day")["spread"].rank(ascending=False, method="first")
                        z = z[z["_rank"] <= k]
                    r = stat(z, extra=True)
                    r.update({"floor": floor, "opp_max": opp, "agree_min": agree,
                              "top_per_day": k, "selection": f"top{k}/day"})
                    top_day_rows.append(r)

    clean = pd.DataFrame(clean_rows)
    top_scan = pd.DataFrame(top_scan_rows)
    top_day = pd.DataFrame(top_day_rows)
    by_model = pd.concat(by_model_rows, ignore_index=True) if by_model_rows else pd.DataFrame()
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()

    clean.to_csv(out_dir / "engine_clean_agreement_sweep.csv", index=False)
    top_scan.to_csv(out_dir / "engine_top_per_scan_sweep.csv", index=False)
    top_day.to_csv(out_dir / "engine_top_per_day_sweep.csv", index=False)
    by_model.to_csv(out_dir / "engine_clean_by_model.csv", index=False)
    daily.to_csv(out_dir / "engine_clean_daily.csv", index=False)
    return {
        "raw": raw,
        "clean": clean,
        "top_scan": top_scan,
        "top_day": top_day,
        "by_model": by_model,
        "daily": daily,
    }


def print_report(reports: dict[str, pd.DataFrame], scores: pd.DataFrame, out_dir: Path) -> None:
    print("=== UNTOUCHED HOLDOUT WINDOW ===")
    t = pd.to_datetime(scores["anchor_time"], utc=True)
    print(f"  rows={len(scores)} symbols={scores['symbol'].nunique()} horizons={scores['horizon'].nunique()}")
    print(f"  {t.min()} -> {t.max()}  ({(t.max() - t.min()).total_seconds()/3600:.1f}h)")
    print(f"  cost={FC.EVAL_COST*100:.3f}% target_edge={FC.TARGET_EDGE*100:.3f}%")

    summ = reports["model_summary"].copy()
    print("\n=== MODEL PREDICTIVENESS SUMMARY ===")
    show = summ[["model", "n", "base_event_rate", "base_avg_pnl", "auc_event",
                 "spearman_event", "spearman_pnl", "p90"]]
    print(show.to_string(index=False, formatters={
        "base_event_rate": "{:.3f}".format,
        "base_avg_pnl": "{:+.4f}".format,
        "auc_event": "{:.3f}".format,
        "spearman_event": "{:+.3f}".format,
        "spearman_pnl": "{:+.3f}".format,
        "p90": "{:.3f}".format,
    }))

    thr = reports["model_thresholds"]
    focus = thr[thr["threshold"].isin([0.70, 0.75, 0.80, 0.82, 0.85, 0.90])]
    focus = focus[focus["n"] >= 20].copy()
    focus = focus.sort_values(["avg_pnl", "win", "n"], ascending=[False, False, False]).head(24)
    print("\n=== BEST PER-MODEL PROBABILITY THRESHOLDS (n>=20) ===")
    print(focus[["model", "threshold", "n", "win", "event_hit", "avg_pnl", "touch_green", "green_days", "days"]]
          .to_string(index=False, formatters={
              "threshold": "{:.2f}".format,
              "win": "{:.3f}".format,
              "event_hit": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
          }))

    raw = reports["engine_raw"]
    print("\n=== ENGINE RAW ARGMAX CONFIDENCE ===")
    print(raw[raw["threshold"].isin([0.70, 0.75, 0.80, 0.82, 0.85, 0.90])]
          [["threshold", "n", "win", "event_hit", "avg_pnl", "touch_green", "green_days", "days", "total"]]
          .to_string(index=False, formatters={
              "threshold": "{:.2f}".format,
              "win": "{:.3f}".format,
              "event_hit": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
              "total": "{:+.2f}".format,
          }))

    clean = reports["engine_clean"]
    sel = clean[(clean["agree_min"] == 2) & (clean["opp_max"] == 0.30)].copy()
    sel = sel[sel["floor"].isin([0.70, 0.75, 0.80, 0.82, 0.85, 0.90])]
    print("\n=== ENGINE CLEAN AGREEMENT (opp<=0.30, agree>=2, all candidates) ===")
    print(sel[["floor", "n", "win", "event_hit", "avg_pnl", "touch_green", "green_days", "days", "total"]]
          .to_string(index=False, formatters={
              "floor": "{:.2f}".format,
              "win": "{:.3f}".format,
              "event_hit": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
              "total": "{:+.2f}".format,
          }))

    scan = reports["engine_top_scan"]
    scan_sel = scan[(scan["agree_min"] == 2) & (scan["opp_max"] == 0.30)
                    & (scan["top_per_scan"].isin([1, 3, 5]))]
    scan_sel = scan_sel[scan_sel["floor"].isin([0.70, 0.80, 0.82, 0.85])].copy()
    print("\n=== ENGINE TOP-PER-SCAN (live-like rank by spread) ===")
    print(scan_sel[["floor", "top_per_scan", "n", "win", "avg_pnl", "touch_green", "green_days", "days", "total"]]
          .to_string(index=False, formatters={
              "floor": "{:.2f}".format,
              "win": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
              "total": "{:+.2f}".format,
          }))

    best_day = reports["engine_top_day"].copy()
    best_day = best_day[(best_day["agree_min"] == 2) & (best_day["opp_max"] == 0.30)]
    best_day = best_day.sort_values(["avg_pnl", "win", "n"], ascending=[False, False, False]).head(16)
    print("\n=== BEST ENGINE TOP-PER-DAY SLICES ===")
    print(best_day[["floor", "top_per_day", "n", "win", "avg_pnl", "touch_green", "green_days", "days", "total"]]
          .to_string(index=False, formatters={
              "floor": "{:.2f}".format,
              "win": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
              "total": "{:+.2f}".format,
          }))

    manifest = {
        "experiment": FC.EXPERIMENT,
        "holdout_scores": str(FC.FAST_ANALYSIS_DIR / "holdout_scores.parquet"),
        "output_dir": str(out_dir),
        "cost_pct": FC.EVAL_COST * 100,
        "target_edge_pct": FC.TARGET_EDGE * 100,
        "files": sorted(p.name for p in out_dir.glob("*.csv")),
    }
    (out_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nreports -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=FC.FAST_ANALYSIS_DIR / "holdout_scores.parquet")
    ap.add_argument("--out-dir", type=Path, default=FC.FAST_ANALYSIS_DIR / "detailed_reports")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(args.scores)
    scores["anchor_time"] = pd.to_datetime(scores["anchor_time"], utc=True)
    scores["day"] = scores["anchor_time"].dt.strftime("%Y-%m-%d")

    c = model_candidates(scores)
    argmax = argmax_candidates(scores)

    model = per_model_reports(c, args.out_dir)
    engine = engine_reports(c, argmax, args.out_dir)
    reports = {
        "model_summary": model["summary"],
        "model_thresholds": model["thresholds"],
        "model_bins": model["bins"],
        "engine_raw": engine["raw"],
        "engine_clean": engine["clean"],
        "engine_top_scan": engine["top_scan"],
        "engine_top_day": engine["top_day"],
        "engine_by_model": engine["by_model"],
        "engine_daily": engine["daily"],
    }
    print_report(reports, scores, args.out_dir)


if __name__ == "__main__":
    main()
