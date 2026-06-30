"""Run the short-horizon research pipeline.

Stages:
  coverage        inspect local 1m target + long feature coverage
  ensure-candles  download/cache top-N 1m candles
  build-dataset   build dense fast curve + ret/mfe/mae targets
  train           train up/down + ret/mfe/mae models and score holdout
  compare         compare simple decision layers on the untouched 3d holdout
  all             run everything in order
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time

import pandas as pd

from . import config as C
from .database import CandleFetcher, CandleStore, OKXClient
from .database.candle_fetcher import RESOLUTION_DAYS
from .fast import config as FC
from .fast.candles import ensure_1m, load_1m, top_liquid_symbols
from .fast.dataset import build_dataset
from .fast.train_eval import compare, train_and_score


def _as_of(value: str = "") -> pd.Timestamp:
    if value:
        return pd.Timestamp(value).tz_convert("UTC").floor("1min")
    return pd.Timestamp.now(tz="UTC").floor("1min")


def _symbols(top: int) -> list[str]:
    path = FC.FAST_DATASETS_DIR / "symbols.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if len(data) >= top:
            return data[:top]
    syms = top_liquid_symbols(top)
    FC.ensure_dirs()
    path.write_text(json.dumps(syms, indent=2), encoding="utf-8")
    return syms


def _recent_dense_start(candles: pd.DataFrame) -> pd.Timestamp | None:
    if candles is None or len(candles) < 3:
        return None
    idx = candles.index.sort_values()
    diffs = idx.to_series().diff().dt.total_seconds().div(60.0)
    dense = diffs <= 2.1
    i = len(dense) - 1
    while i > 0 and bool(dense.iloc[i]):
        i -= 1
    pos = min(i + 1, len(idx) - 1)
    return idx[pos]


def stage_coverage(args) -> pd.DataFrame:
    FC.ensure_dirs()
    syms = _symbols(args.top)
    as_of = _as_of(args.as_of)
    max_horizon = max(h.minutes for h in FC.HORIZONS)
    holdout_end = as_of - pd.Timedelta(minutes=max_horizon + 2)
    holdout_start = holdout_end - pd.Timedelta(days=args.holdout_days)
    train_start = holdout_start - pd.Timedelta(days=args.train_days)
    feature_start = train_start - pd.Timedelta(minutes=FC.CURVE_MAX_DEPTH_MIN)
    target_end = as_of - pd.Timedelta(minutes=2)

    store = CandleStore(C.CANDLES_DIR)
    rows = []
    for sym in syms:
        try:
            target = load_1m(sym)
        except Exception:
            target = None
        feature = store.load(sym)
        t_min = target.index.min() if target is not None and not target.empty else pd.NaT
        t_max = target.index.max() if target is not None and not target.empty else pd.NaT
        f_min = feature.index.min() if feature is not None and not feature.empty else pd.NaT
        f_max = feature.index.max() if feature is not None and not feature.empty else pd.NaT
        dense_start = _recent_dense_start(feature) if feature is not None and not feature.empty else pd.NaT
        target_ok = pd.notna(t_min) and t_min <= train_start + pd.Timedelta(minutes=2) and t_max >= target_end
        feature_ok = pd.notna(f_min) and f_min <= feature_start + pd.Timedelta(minutes=5) and f_max >= holdout_end
        rows.append({
            "symbol": sym,
            "target_1m_min": t_min,
            "target_1m_max": t_max,
            "feature_min": f_min,
            "feature_max": f_max,
            "recent_dense_start": dense_start,
            "target_ok": bool(target_ok),
            "feature_ok": bool(feature_ok),
            "usable": bool(target_ok and feature_ok),
        })

    df = pd.DataFrame(rows)
    out = FC.FAST_ANALYSIS_DIR / "coverage.csv"
    df.to_csv(out, index=False)
    usable = int(df["usable"].sum()) if len(df) else 0
    target_ok = int(df["target_ok"].sum()) if len(df) else 0
    feature_ok = int(df["feature_ok"].sum()) if len(df) else 0
    print(f"{FC.EXPERIMENT} coverage top={len(syms)}")
    print(f"  target 1m needed: {train_start} -> {target_end}")
    print(f"  feature curve needed: {feature_start} -> {holdout_end}")
    print(f"  target_ok={target_ok} feature_ok={feature_ok} usable={usable}")
    if len(df):
        show = df.sort_values(["usable", "target_ok", "feature_ok"], ascending=False).head(20)
        print(show.to_string(index=False))
    print(f"coverage -> {out}")
    return df


def stage_ensure_candles(args) -> list[str]:
    FC.ensure_dirs()
    syms = _symbols(args.top)
    as_of = _as_of(args.as_of)
    if args.require_feature_history:
        syms = _feature_ready_symbols(syms, as_of, args.train_days, args.holdout_days)
        print(f"  target download limited to feature-ready symbols: {len(syms)}/{args.top}")
    days = args.train_days + args.holdout_days + FC.DOWNLOAD_CUSHION_DAYS + 1
    start = as_of - pd.Timedelta(days=days)
    end = as_of
    print(f"{FC.EXPERIMENT} target candles: top={len(syms)} 1m range {start} -> {end} (~{days}d)")
    print(f"  cache={FC.FAST_CANDLES_DIR}")
    ok = fail = 0
    t0 = time.time()
    def work(sym: str) -> tuple[str, str]:
        try:
            r = ensure_1m(sym, start, end, client=OKXClient(timeout=25.0))
            return sym, str(r.get("status"))
        except Exception as exc:
            return sym, f"FAIL {exc}"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, sym) for sym in syms]
        for i, fut in enumerate(as_completed(futures), 1):
            sym, status = fut.result()
            if status == "ok":
                ok += 1
            else:
                fail += 1
            if i % 5 == 0 or i == len(syms):
                print(f"  candles {i}/{len(syms)} ok={ok} fail={fail} last={sym} {status} elapsed={time.time()-t0:.0f}s", flush=True)
    return syms


def _feature_ready_symbols(syms: list[str], as_of: pd.Timestamp,
                           train_days: int, holdout_days: int) -> list[str]:
    max_horizon = max(h.minutes for h in FC.HORIZONS)
    holdout_end = as_of - pd.Timedelta(minutes=max_horizon + 2)
    holdout_start = holdout_end - pd.Timedelta(days=holdout_days)
    train_start = holdout_start - pd.Timedelta(days=train_days)
    feature_start = train_start - pd.Timedelta(minutes=FC.CURVE_MAX_DEPTH_MIN)
    store = CandleStore(C.CANDLES_DIR)
    ready = []
    for sym in syms:
        existing = store.load(sym)
        if existing is None or existing.empty:
            continue
        if (
            existing.index.min() <= feature_start + pd.Timedelta(minutes=5)
            and existing.index.max() >= holdout_end
        ):
            ready.append(sym)
    return ready


def stage_ensure_features(args) -> list[str]:
    FC.ensure_dirs()
    syms = _symbols(args.top)
    as_of = _as_of(args.as_of)
    max_horizon = max(h.minutes for h in FC.HORIZONS)
    holdout_end = as_of - pd.Timedelta(minutes=max_horizon + 2)
    holdout_start = holdout_end - pd.Timedelta(days=args.holdout_days)
    train_start = holdout_start - pd.Timedelta(days=args.train_days)
    feature_start = train_start - pd.Timedelta(minutes=FC.CURVE_MAX_DEPTH_MIN)

    store = CandleStore(C.CANDLES_DIR)
    fetcher = CandleFetcher(OKXClient(timeout=25.0), store)
    print(f"{FC.EXPERIMENT} feature candles: top={len(syms)} need {feature_start} -> {holdout_end}")
    ok = cached = skipped = fail = 0
    t0 = time.time()
    for i, sym in enumerate(syms, 1):
        try:
            existing = store.load(sym)
            too_young_for_v2 = (
                existing is not None
                and not existing.empty
                and existing.index.min() > feature_start + pd.Timedelta(minutes=5)
            )
            if too_young_for_v2:
                skipped += 1
                status = "skip-too-young"
            elif (
                existing is not None
                and not existing.empty
                and existing.index.min() <= feature_start + pd.Timedelta(minutes=5)
                and existing.index.max() >= holdout_end
            ):
                cached += 1
                status = "cached"
            else:
                has_deep = (
                    existing is not None
                    and not existing.empty
                    and existing.index.min() <= feature_start + pd.Timedelta(minutes=5)
                )
                r = fetcher.fetch_symbol(sym, resolutions=RESOLUTION_DAYS, update=has_deep)
                status = str(r.get("status"))
                ok += int(status == "ok")
        except Exception as exc:
            fail += 1
            status = f"FAIL {exc}"
        if i % 5 == 0 or i == len(syms):
            print(
                f"  features {i}/{len(syms)} cached={cached} ok={ok} skipped={skipped} fail={fail} "
                f"last={sym} {status} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return syms


def stage_build_dataset(args, syms: list[str] | None = None) -> None:
    FC.ensure_dirs()
    syms = syms or _symbols(args.top)
    build_dataset(
        syms,
        _as_of(args.as_of),
        train_days=args.train_days,
        holdout_days=args.holdout_days,
        train_anchors=args.train_anchors,
        holdout_step_min=args.holdout_step,
        fresh=args.fresh_dataset,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["coverage", "ensure-candles", "ensure-features", "build-dataset", "train", "compare", "all"],
                    default="all")
    ap.add_argument("--top", type=int, default=FC.TOP_SYMBOLS)
    ap.add_argument("--train-days", type=float, default=FC.TRAIN_DAYS)
    ap.add_argument("--holdout-days", type=float, default=FC.HOLDOUT_DAYS)
    ap.add_argument("--train-anchors", type=int, default=FC.TRAIN_ANCHORS_PER_SYMBOL)
    ap.add_argument("--holdout-step", type=int, default=FC.HOLDOUT_STEP_MIN)
    ap.add_argument("--fresh-dataset", action="store_true")
    ap.add_argument("--iterations", type=int, default=450)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6, help="parallel candle downloads")
    ap.add_argument("--as-of", default="", help="fixed UTC timestamp for train/holdout cutoffs")
    ap.add_argument("--all-targets", action="store_false", dest="require_feature_history",
                    help="download 1m targets even for symbols that cannot satisfy the long feature lookback")
    ap.set_defaults(require_feature_history=True)
    args = ap.parse_args()

    syms: list[str] | None = None
    if args.stage == "coverage":
        stage_coverage(args)
        return
    if args.stage in ("ensure-features", "all"):
        syms = stage_ensure_features(args)
    if args.stage in ("ensure-candles", "all"):
        syms = stage_ensure_candles(args)
    if args.stage in ("build-dataset", "all"):
        stage_build_dataset(args, syms)
    if args.stage in ("train", "all"):
        train_and_score(iterations=args.iterations, depth=args.depth)
    if args.stage in ("compare", "all"):
        compare()


if __name__ == "__main__":
    main()
