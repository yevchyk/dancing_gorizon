"""Threshold recommendations and standard-vs-fast agreement reports."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .fast.curve import FastCurve
from .run_fast_detailed_reports import model_candidates

OUT = FC.FAST_ANALYSIS_DIR / "family_analysis"
NS_PER_MIN = 60_000_000_000


def _stat(d: pd.DataFrame, side: int | None = None) -> dict:
    if len(d) == 0:
        return {
            "n": 0, "win": np.nan, "dir_correct": np.nan, "event_hit": np.nan,
            "avg_pnl": np.nan, "median_pnl": np.nan, "touch_green": np.nan,
            "green_days": 0, "days": 0, "total": 0.0,
        }
    if side is None:
        pnl = d["pnl"].astype(float)
        direction = d.get("event_hit", pnl > 0)
        touch_green = d.get("touch_green", pd.Series(np.nan, index=d.index))
    else:
        ret = d["real_ret"].astype(float)
        pnl = side * ret - FC.EVAL_COST
        direction = (side * ret) > 0
        if side == 1:
            touch_green = d["real_mfe"].astype(float) > FC.EVAL_COST
            event = ret > FC.TARGET_EDGE
        else:
            touch_green = -d["real_mae"].astype(float) > FC.EVAL_COST
            event = ret < -FC.TARGET_EDGE
        d = d.assign(event_hit=event.astype(int))
    daily = d.assign(_pnl=pnl).groupby("day")["_pnl"].mean() * 100
    return {
        "n": int(len(d)),
        "win": float((pnl > 0).mean()),
        "dir_correct": float(pd.Series(direction).mean()),
        "event_hit": float(d["event_hit"].mean()) if "event_hit" in d else np.nan,
        "avg_pnl": float(pnl.mean() * 100),
        "median_pnl": float(pnl.median() * 100),
        "touch_green": float(pd.Series(touch_green).mean()),
        "green_days": int((daily > 0).sum()),
        "days": int(len(daily)),
        "total": float(pnl.sum() * 100),
    }


def threshold_recommendations(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = model_candidates(scores)
    thresholds = np.round(np.arange(0.50, 0.951, 0.01), 2)
    hours = max(1.0, (c["anchor_time"].max() - c["anchor_time"].min()).total_seconds() / 3600)
    rows = []
    for model, d in c.groupby("model", sort=True):
        for thr in thresholds:
            z = d[d["prob"] >= thr]
            r = _stat(z)
            r.update({
                "family": "fast_v2",
                "model": f"fast_v2_{model}",
                "threshold": thr,
                "signals_per_24h": float(len(z) / hours * 24.0),
                "coverage": float(len(z) / len(d)),
            })
            rows.append(r)
    sweep = pd.DataFrame(rows)

    picks = []
    for model, d in sweep.groupby("model", sort=True):
        d = d.sort_values("threshold")
        profiles = {
            "active": d[(d["n"] >= 500) & (d["avg_pnl"] >= 0.02) & (d["green_days"] >= 2)].copy(),
            "balanced": d[(d["n"] >= 100) & (d["avg_pnl"] >= 0.02) & (d["green_days"] >= 2)].copy(),
            "precision": d[(d["n"] >= 20) & (d["avg_pnl"] >= 0.10) & (d["green_days"] >= 1)].copy(),
        }
        row = {"model": model}
        best_any = None
        for name, p in profiles.items():
            if len(p):
                p["_score"] = p["avg_pnl"] * np.sqrt(p["n"])
                best = p.sort_values(["_score", "n"], ascending=[False, False]).iloc[0]
                best_any = best if best_any is None else best_any
                for col in ("threshold", "n", "signals_per_24h", "win", "avg_pnl",
                            "touch_green", "green_days", "days", "total"):
                    row[f"{name}_{col}"] = best[col]
            else:
                for col in ("threshold", "n", "signals_per_24h", "win", "avg_pnl",
                            "touch_green", "green_days", "days", "total"):
                    row[f"{name}_{col}"] = np.nan

        if best_any is not None:
            row["recommendation"] = (
                "active" if not pd.isna(row.get("active_threshold", np.nan))
                else "balanced" if not pd.isna(row.get("balanced_threshold", np.nan))
                else "precision"
            )
        else:
            row["recommendation"] = "skip_or_agreement_only"
        picks.append(row)

    recs = pd.DataFrame(picks)
    return sweep, recs


def _to_ns(index: pd.DatetimeIndex) -> np.ndarray:
    return index.as_unit("ns").asi8


def score_standard_5m(fast5: pd.DataFrame) -> pd.DataFrame:
    store = CandleStore(C.CANDLES_DIR)
    curve = FastCurve(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
    model_dir = C.MODELS_DIR / "dir_prob"
    up = joblib.load(model_dir / "up_5m.joblib")
    dn = joblib.load(model_dir / "down_5m.joblib")
    up_cols = joblib.load(model_dir / "up_5m_columns.joblib")
    dn_cols = joblib.load(model_dir / "down_5m_columns.joblib")

    recs = []
    cols = curve.columns()
    for sym, d in fast5.groupby("symbol", sort=False):
        candles = store.load(sym)
        if candles is None or candles.empty:
            continue
        candles = candles.sort_index()
        anchors = pd.DatetimeIndex(pd.to_datetime(d["anchor_time"], utc=True))
        anchors_ns = anchors.as_unit("ns").asi8
        ts_ns = _to_ns(candles.index)
        close = candles["close"].to_numpy("float64")
        feats, valid = curve.build_matrix(ts_ns, close, anchors_ns)
        if valid.sum() == 0:
            continue
        idx = np.where(valid)[0]
        X = pd.DataFrame(feats[idx], columns=cols)
        part = d.iloc[idx].copy()
        part["standard_p_up_5m"] = up.predict_proba(X[up_cols])[:, 1]
        part["standard_p_down_5m"] = dn.predict_proba(X[dn_cols])[:, 1]
        recs.append(part)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def agreement_reports(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fast5 = scores[scores["horizon"] == "5m"].copy()
    fast5 = fast5.rename(columns={"p_up": "fast_v2_p_up_5m", "p_down": "fast_v2_p_down_5m"})
    joined = score_standard_5m(fast5)
    joined["day"] = pd.to_datetime(joined["anchor_time"], utc=True).dt.strftime("%Y-%m-%d")

    std_thrs = (0.70, 0.75, 0.80, 0.82, 0.85, 0.90)
    fast_thrs = (0.50, 0.52, 0.55, 0.60, 0.65, 0.70)
    rows = []
    solo_rows = []

    for thr in std_thrs:
        for direction, side in (("up", 1), ("down", -1)):
            prob = joined[f"standard_p_{direction}_5m"]
            d = joined[prob >= thr]
            r = _stat(d, side=side)
            r.update({"family": "standard", "model": f"standard_{direction}_5m", "threshold": thr})
            solo_rows.append(r)

    for thr in fast_thrs:
        for direction, side in (("up", 1), ("down", -1)):
            prob = joined[f"fast_v2_p_{direction}_5m"]
            d = joined[prob >= thr]
            r = _stat(d, side=side)
            r.update({"family": "fast_v2", "model": f"fast_v2_{direction}_5m", "threshold": thr})
            solo_rows.append(r)

    for std_thr in std_thrs:
        for fast_thr in fast_thrs:
            for direction, side in (("up", 1), ("down", -1)):
                mask = (
                    (joined[f"standard_p_{direction}_5m"] >= std_thr)
                    & (joined[f"fast_v2_p_{direction}_5m"] >= fast_thr)
                )
                d = joined[mask]
                r = _stat(d, side=side)
                r.update({
                    "direction": direction,
                    "standard_threshold": std_thr,
                    "fast_v2_threshold": fast_thr,
                    "standard_model": f"standard_{direction}_5m",
                    "fast_v2_model": f"fast_v2_{direction}_5m",
                    "rule": "same_direction_probability",
                })
                rows.append(r)

    clean_rows = []
    for std_thr in std_thrs:
        for fast_thr in fast_thrs:
            for opp in (0.30, 0.40, 0.50):
                for direction, side in (("up", 1), ("down", -1)):
                    other = "down" if direction == "up" else "up"
                    mask = (
                        (joined[f"standard_p_{direction}_5m"] >= std_thr)
                        & (joined[f"standard_p_{other}_5m"] <= opp)
                        & (joined[f"fast_v2_p_{direction}_5m"] >= fast_thr)
                        & (joined[f"fast_v2_p_{other}_5m"] <= opp)
                    )
                    d = joined[mask]
                    r = _stat(d, side=side)
                    r.update({
                        "direction": direction,
                        "standard_threshold": std_thr,
                        "fast_v2_threshold": fast_thr,
                        "opp_max": opp,
                        "rule": "same_direction_clean_both",
                    })
                    clean_rows.append(r)

    solo = pd.DataFrame(solo_rows)
    agree = pd.DataFrame(rows)
    clean = pd.DataFrame(clean_rows)
    return {"joined": joined, "solo": solo, "agreement": agree, "clean": clean}


def print_summary(recs: pd.DataFrame, agree: dict[str, pd.DataFrame]) -> None:
    print("=== FAST_V2 MODEL THRESHOLD RECOMMENDATIONS ===")
    show_cols = [
        "model", "recommendation",
        "active_threshold", "active_n", "active_signals_per_24h", "active_win", "active_avg_pnl",
        "balanced_threshold", "balanced_n", "balanced_signals_per_24h", "balanced_win", "balanced_avg_pnl",
        "precision_threshold", "precision_n", "precision_signals_per_24h", "precision_win", "precision_avg_pnl",
    ]
    print(recs[show_cols].to_string(index=False, formatters={
        "active_threshold": "{:.2f}".format,
        "active_signals_per_24h": "{:.1f}".format,
        "active_win": "{:.3f}".format,
        "active_avg_pnl": "{:+.4f}".format,
        "balanced_threshold": "{:.2f}".format,
        "balanced_signals_per_24h": "{:.1f}".format,
        "balanced_win": "{:.3f}".format,
        "balanced_avg_pnl": "{:+.4f}".format,
        "precision_threshold": "{:.2f}".format,
        "precision_signals_per_24h": "{:.1f}".format,
        "precision_win": "{:.3f}".format,
        "precision_avg_pnl": "{:+.4f}".format,
    }))

    print("\n=== STANDARD_5M vs FAST_V2_5M SOLO ===")
    solo = agree["solo"].copy()
    solo = solo[((solo["family"] == "standard") & solo["threshold"].isin([0.80, 0.82, 0.85]))
                | ((solo["family"] == "fast_v2") & solo["threshold"].isin([0.50, 0.52, 0.55, 0.60]))]
    print(solo[["model", "threshold", "n", "win", "dir_correct", "event_hit", "avg_pnl", "touch_green", "green_days", "days"]]
          .to_string(index=False, formatters={
              "threshold": "{:.2f}".format,
              "win": "{:.3f}".format,
              "dir_correct": "{:.3f}".format,
              "event_hit": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
          }))

    print("\n=== BEST STANDARD_UP_5M + FAST_V2_UP_5M AGREEMENT ===")
    a = agree["agreement"]
    up = a[(a["direction"] == "up") & (a["n"] >= 20)].sort_values(
        ["avg_pnl", "win", "n"], ascending=[False, False, False]
    ).head(20)
    print(up[["standard_threshold", "fast_v2_threshold", "n", "win", "dir_correct", "event_hit", "avg_pnl", "touch_green", "green_days", "days"]]
          .to_string(index=False, formatters={
              "standard_threshold": "{:.2f}".format,
              "fast_v2_threshold": "{:.2f}".format,
              "win": "{:.3f}".format,
              "dir_correct": "{:.3f}".format,
              "event_hit": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
          }))

    print("\n=== BEST CLEAN BOTH AGREEMENT (UP/DOWN, n>=10) ===")
    clean = agree["clean"]
    best = clean[clean["n"] >= 10].sort_values(["avg_pnl", "win", "n"], ascending=[False, False, False]).head(30)
    print(best[["direction", "standard_threshold", "fast_v2_threshold", "opp_max", "n", "win", "dir_correct", "event_hit", "avg_pnl", "touch_green", "green_days", "days"]]
          .to_string(index=False, formatters={
              "standard_threshold": "{:.2f}".format,
              "fast_v2_threshold": "{:.2f}".format,
              "opp_max": "{:.2f}".format,
              "win": "{:.3f}".format,
              "dir_correct": "{:.3f}".format,
              "event_hit": "{:.3f}".format,
              "avg_pnl": "{:+.4f}".format,
              "touch_green": "{:.3f}".format,
          }))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(FC.FAST_ANALYSIS_DIR / "holdout_scores.parquet")
    scores["anchor_time"] = pd.to_datetime(scores["anchor_time"], utc=True)
    scores["day"] = scores["anchor_time"].dt.strftime("%Y-%m-%d")

    sweep, recs = threshold_recommendations(scores)
    sweep.to_csv(OUT / "fast_v2_threshold_sweep_001.csv", index=False)
    recs.to_csv(OUT / "fast_v2_threshold_recommendations.csv", index=False)

    agree = agreement_reports(scores)
    agree["joined"].to_parquet(OUT / "standard_fast_v2_5m_joined.parquet", index=False)
    agree["solo"].to_csv(OUT / "standard_fast_v2_5m_solo.csv", index=False)
    agree["agreement"].to_csv(OUT / "standard_fast_v2_5m_agreement.csv", index=False)
    agree["clean"].to_csv(OUT / "standard_fast_v2_5m_clean_agreement.csv", index=False)

    print_summary(recs, agree)
    print(f"\nreports -> {OUT}")


if __name__ == "__main__":
    main()
