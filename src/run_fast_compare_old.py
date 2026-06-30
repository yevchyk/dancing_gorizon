"""Compare production/v4 models with fast_v1 on matching recent windows.

This is a signal-level benchmark, not a capital/cooldown event simulation.
It evaluates both model families on the same calendar windows and reports the
same decision-layer slices: raw confidence, top-K/day, risk-adjusted, and the
clean-agreement mechanic used by the live v4 engine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .fast.candles import load_1m
from .fast.curve import FastCurve

NS_PER_MIN = 60_000_000_000


@dataclass(frozen=True)
class Family:
    name: str
    horizons: tuple
    curve_points: int
    curve_min_step: float
    curve_max_depth: float
    curve_segments: tuple | None
    step_min: int
    model_dir_prob: Path
    model_dir_reg: Path
    use_fast_features: bool
    use_fast_targets: bool
    clean_floor: float
    clean_opp: float
    clean_exclude: tuple[str, ...]


OLD = Family(
    name="old_v4",
    horizons=C.HORIZONS,
    curve_points=C.CURVE_POINTS,
    curve_min_step=C.CURVE_MIN_STEP_MIN,
    curve_max_depth=C.CURVE_MAX_DEPTH_MIN,
    curve_segments=None,
    step_min=5,
    model_dir_prob=C.MODELS_DIR / "dir_prob",
    model_dir_reg=C.MODELS_DIR / "reg",
    use_fast_features=False,
    use_fast_targets=False,
    clean_floor=C.SIGNAL_FLOOR,
    clean_opp=C.CLEAN_OPP_MAX,
    clean_exclude=C.CONF_EXCLUDE,
)

FAST = Family(
    name=FC.EXPERIMENT,
    horizons=FC.HORIZONS,
    curve_points=FC.CURVE_POINTS,
    curve_min_step=FC.CURVE_MIN_STEP_MIN,
    curve_max_depth=FC.CURVE_MAX_DEPTH_MIN,
    curve_segments=FC.CURVE_SEGMENTS,
    step_min=FC.HOLDOUT_STEP_MIN,
    model_dir_prob=FC.FAST_MODELS_DIR,
    model_dir_reg=FC.FAST_MODELS_DIR,
    use_fast_features=False,
    use_fast_targets=True,
    clean_floor=0.60,
    clean_opp=0.50,
    clean_exclude=(),
)


def _to_ns(index: pd.DatetimeIndex) -> np.ndarray:
    return index.as_unit("ns").asi8


def _targets(ts_ns: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
             anchors_ns: np.ndarray, horizons) -> dict[str, np.ndarray]:
    n = len(anchors_ns)
    out = {f"{k}_{h.label}": np.full(n, np.nan, dtype="float32")
           for h in horizons for k in ("ret", "mfe", "mae")}
    entry_idx = np.searchsorted(ts_ns, anchors_ns, side="right") - 1
    for i, a_ns in enumerate(anchors_ns):
        ei = int(entry_idx[i])
        if ei < 0:
            continue
        entry = close[ei]
        if not np.isfinite(entry) or entry <= 0:
            continue
        for h in horizons:
            end = a_ns + h.minutes * NS_PER_MIN
            fj = int(np.searchsorted(ts_ns, end, side="right"))
            if fj <= ei + 1:
                continue
            hh = high[ei + 1:fj]
            ll = low[ei + 1:fj]
            out[f"ret_{h.label}"][i] = close[fj - 1] / entry - 1.0
            out[f"mfe_{h.label}"][i] = hh.max() / entry - 1.0
            out[f"mae_{h.label}"][i] = ll.min() / entry - 1.0
    return out


def _full_fast_symbols(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    syms = []
    need_feature_start = start - pd.Timedelta(minutes=FC.CURVE_MAX_DEPTH_MIN)
    store = CandleStore(C.CANDLES_DIR)
    for path in sorted(FC.FAST_CANDLES_DIR.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["timestamp"])
        t = pd.to_datetime(df["timestamp"], utc=True)
        target_ok = t.min() <= start + pd.Timedelta(minutes=2) and t.max() >= end
        feature = store.load(path.stem)
        feature_ok = (
            feature is not None
            and not feature.empty
            and feature.index.min() <= need_feature_start + pd.Timedelta(minutes=5)
            and feature.index.max() >= end
        )
        if target_ok and feature_ok:
            syms.append(path.stem)
    return syms


def _load_feature_candles(family: Family, symbol: str):
    if family.use_fast_features:
        return load_1m(symbol)
    return CandleStore(C.CANDLES_DIR).load(symbol)


def _load_target_candles(family: Family, symbol: str):
    if family.use_fast_targets:
        return load_1m(symbol)
    return CandleStore(C.CANDLES_DIR).load(symbol)


def score_family(family: Family, symbols: list[str], start: pd.Timestamp,
                 end: pd.Timestamp, cache_name: str) -> pd.DataFrame:
    out_path = FC.FAST_ANALYSIS_DIR / f"{cache_name}_{family.name}_scores.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    curve = FastCurve(
        family.curve_points,
        family.curve_min_step,
        family.curve_max_depth,
        family.curve_segments,
    )
    models = {}
    for h in family.horizons:
        lab = h.label
        models[lab] = {
            "up": joblib.load(family.model_dir_prob / f"up_{lab}.joblib"),
            "down": joblib.load(family.model_dir_prob / f"down_{lab}.joblib"),
            "ret": joblib.load(family.model_dir_reg / f"ret_{lab}.joblib"),
            "mfe": joblib.load(family.model_dir_reg / f"mfe_{lab}.joblib"),
            "mae": joblib.load(family.model_dir_reg / f"mae_{lab}.joblib"),
            "cols_up": joblib.load(family.model_dir_prob / f"up_{lab}_columns.joblib"),
            "cols_down": joblib.load(family.model_dir_prob / f"down_{lab}_columns.joblib"),
            "cols_reg": joblib.load(family.model_dir_reg / f"ret_{lab}_columns.joblib"),
        }

    recs = []
    max_h = max(h.minutes for h in family.horizons)
    end = end - pd.Timedelta(minutes=max_h)
    for i, sym in enumerate(symbols, 1):
        feature_candles = _load_feature_candles(family, sym)
        target_candles = _load_target_candles(family, sym)
        if feature_candles is None or feature_candles.empty:
            continue
        if target_candles is None or target_candles.empty:
            continue
        feature_candles = feature_candles.sort_index()
        target_candles = target_candles.sort_index()
        anchors = pd.date_range(start.ceil(f"{family.step_min}min"),
                                end.floor(f"{family.step_min}min"),
                                freq=f"{family.step_min}min")
        if anchors.empty:
            continue
        anchors = anchors[(anchors >= target_candles.index.min()) & (anchors <= target_candles.index.max())]
        if anchors.empty:
            continue
        anchors_ns = anchors.as_unit("ns").asi8
        feat_ts_ns = _to_ns(feature_candles.index)
        feat_close = feature_candles["close"].to_numpy("float64")
        tgt_ts_ns = _to_ns(target_candles.index)
        close = target_candles["close"].to_numpy("float64")
        high = target_candles["high"].to_numpy("float64")
        low = target_candles["low"].to_numpy("float64")
        feats, valid = curve.build_matrix(feat_ts_ns, feat_close, anchors_ns)
        targs = _targets(tgt_ts_ns, high, low, close, anchors_ns, family.horizons)
        for values in targs.values():
            valid &= np.isfinite(values)
        if valid.sum() == 0:
            continue
        idx = np.where(valid)[0]
        X = pd.DataFrame(feats[idx], columns=curve.columns())
        anchor_ok = anchors[idx]
        for h in family.horizons:
            lab = h.label
            m = models[lab]
            recs.append(pd.DataFrame({
                "family": family.name,
                "symbol": sym,
                "anchor_time": anchor_ok,
                "day": anchor_ok.strftime("%Y-%m-%d"),
                "horizon": lab,
                "p_up": m["up"].predict_proba(X[m["cols_up"]])[:, 1],
                "p_down": m["down"].predict_proba(X[m["cols_down"]])[:, 1],
                "pred_ret": m["ret"].predict(X[m["cols_reg"]]),
                "pred_mfe": m["mfe"].predict(X[m["cols_reg"]]),
                "pred_mae": m["mae"].predict(X[m["cols_reg"]]),
                "real_ret": targs[f"ret_{lab}"][idx],
                "real_mfe": targs[f"mfe_{lab}"][idx],
                "real_mae": targs[f"mae_{lab}"][idx],
            }))
        if i % 5 == 0 or i == len(symbols):
            print(f"  scored {family.name} {i}/{len(symbols)}", flush=True)

    scored = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_path, index=False)
    print(f"{family.name} scores -> {out_path} ({len(scored)})")
    return scored


def enrich(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["conf"] = np.maximum(s["p_up"], s["p_down"])
    s["side"] = np.where(s["p_up"] >= s["p_down"], 1, -1)
    s["opp"] = np.minimum(s["p_up"], s["p_down"])
    s["spread"] = s["conf"] - s["opp"]
    fav = np.where(s["side"] == 1, s["pred_mfe"], -s["pred_mae"])
    adv = np.where(s["side"] == 1, np.abs(s["pred_mae"]), s["pred_mfe"])
    s["rr"] = np.clip(fav / (np.abs(adv) + 1e-4), 0, 5)
    s["score_rr"] = s["conf"] * s["rr"]
    s["score_spread"] = s["spread"]
    s["score_predret"] = np.abs(s["pred_ret"])
    s["side_predret"] = np.where(s["pred_ret"] >= 0, 1, -1)
    s["pnl"] = s["side"] * s["real_ret"] - FC.EVAL_COST
    s["pnl_predret"] = s["side_predret"] * s["real_ret"] - FC.EVAL_COST
    return s


def stat(df: pd.DataFrame, pnl_col: str = "pnl") -> dict:
    if len(df) == 0:
        return {"n": 0, "win": np.nan, "avg_pnl": np.nan, "green": np.nan,
                "days": np.nan, "total": np.nan}
    pnl = df[pnl_col].to_numpy()
    daily = df.groupby("day")[pnl_col].mean() * 100
    return {"n": int(len(df)), "win": float((pnl > 0).mean()),
            "avg_pnl": float(pnl.mean() * 100), "green": int((daily > 0).sum()),
            "days": int(len(daily)), "total": float(pnl.sum() * 100)}


def evaluate(scored: pd.DataFrame, family: Family, window_name: str) -> pd.DataFrame:
    s = enrich(scored)
    rows = []
    def add(strategy: str, df: pd.DataFrame, pnl_col: str = "pnl") -> None:
        r = stat(df, pnl_col)
        r.update({"window": window_name, "family": family.name, "strategy": strategy})
        rows.append(r)

    for thr in (0.60, 0.70, 0.75, 0.80, 0.82):
        add(f"conf>={thr:.2f}", s[s.conf >= thr])

    for k in (5, 10, 20, 50):
        for score in ("conf", "score_rr", "score_spread"):
            d = s.copy()
            d["_rk"] = d.groupby("day")[score].rank(ascending=False, method="first")
            add(f"top{k}/day {score}", d[d._rk <= k])
        d = s.copy()
        d["_rk"] = d.groupby("day")["score_predret"].rank(ascending=False, method="first")
        add(f"top{k}/day predret-side", d[d._rk <= k], "pnl_predret")

    fire = s[(s.conf >= family.clean_floor) & (s.opp <= family.clean_opp)].copy()
    if family.clean_exclude:
        names = np.where(fire.side == 1, "up_" + fire.horizon.astype(str),
                         "down_" + fire.horizon.astype(str))
        fire = fire[~pd.Series(names, index=fire.index).isin(family.clean_exclude)]
    grp = fire.groupby(["symbol", "anchor_time", "side"]).size().rename("agree").reset_index()
    agree = grp[grp.agree >= 2][["symbol", "anchor_time", "side"]]
    clean = fire.merge(agree, on=["symbol", "anchor_time", "side"], how="inner")
    clean["_rk"] = clean.groupby("day")["spread"].rank(ascending=False, method="first") if len(clean) else []
    for k in (5, 10, 20, 50):
        add(f"clean agree2 top{k}/day", clean[clean._rk <= k] if len(clean) else clean)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2026-06-01T09:26:00Z")
    args = ap.parse_args()
    as_of = pd.Timestamp(args.as_of).tz_convert("UTC").floor("1min")
    eval_end = as_of - pd.Timedelta(minutes=max(h.minutes for h in OLD.horizons))
    three_start = pd.Timestamp("2026-05-29T09:14:00Z")
    ten_start = as_of - pd.Timedelta(days=10)
    symbols = _full_fast_symbols(ten_start, as_of)
    print(f"symbols={len(symbols)} as_of={as_of} eval_end={eval_end}")
    print(symbols)

    all_rows = []
    for window_name, start in (("same3d", three_start), ("last10d", ten_start)):
        print(f"\n=== scoring window {window_name}: {start} -> {eval_end} ===")
        old_scores = score_family(OLD, symbols, start, as_of, window_name)
        fast_scores = score_family(FAST, symbols, start, as_of, window_name)
        all_rows.append(evaluate(old_scores, OLD, window_name))
        all_rows.append(evaluate(fast_scores, FAST, window_name))

    out = pd.concat(all_rows, ignore_index=True)
    path = FC.FAST_ANALYSIS_DIR / "old_vs_fast_compare.csv"
    out.to_csv(path, index=False)
    show = out[out.strategy.isin([
        "top5/day predret-side",
        "top10/day predret-side",
        "top20/day score_rr",
        "top20/day score_spread",
        "clean agree2 top20/day",
        "conf>=0.80",
        "conf>=0.82",
    ])].copy()
    print("\n=== SELECTED COMPARISON ===")
    print(show.to_string(index=False, formatters={
        "win": "{:.3f}".format,
        "avg_pnl": "{:+.4f}".format,
        "total": "{:+.2f}".format,
    }))
    print(f"\nfull compare -> {path}")


if __name__ == "__main__":
    main()
