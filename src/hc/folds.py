"""Walk-forward fold selection and embargo-safe splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config as HC
from .data import prepare_btc_frames


@dataclass(frozen=True)
class FoldSpec:
    name: str
    test_start: str
    test_end: str
    purpose: str
    reason: str
    btc_return_pct: float | None = None
    btc_range_pct: float | None = None

    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_start)

    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_end)

    def to_dict(self) -> dict:
        return asdict(self)


def _btc_window_stats(btc_5m: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict | None:
    frame = btc_5m[(btc_5m.index >= start) & (btc_5m.index < end)]
    frame = frame[np.isfinite(frame["close"]) & (frame["close"] > 0)]
    if len(frame) < 100:
        return None
    first = float(frame["close"].iloc[0])
    last = float(frame["close"].iloc[-1])
    ret_pct = (last / first - 1.0) * 100.0
    range_pct = (float(frame["close"].max()) / float(frame["close"].min()) - 1.0) * 100.0
    return {"ret_pct": ret_pct, "range_pct": range_pct}


def choose_folds(df: pd.DataFrame, *, max_folds: int = 3) -> list[FoldSpec]:
    base_time = pd.to_datetime(df["base_time"], utc=True)
    latest = base_time.max().floor("5min") + pd.Timedelta(minutes=5)
    primary_start = latest - pd.Timedelta(days=HC.TEST_DAYS)
    primary = FoldSpec(
        name="fold1_primary_last7d",
        test_start=primary_start.isoformat(),
        test_end=latest.isoformat(),
        purpose="live-like check",
        reason="latest available 7-day window in the built dataset",
    )
    if max_folds <= 1:
        return [primary]

    btc_5m = prepare_btc_frames()["5m"]
    earliest = base_time.min().ceil("1D")
    latest_candidate_start = primary_start - pd.Timedelta(days=HC.TEST_DAYS)
    candidates: list[dict] = []
    for start in pd.date_range(earliest, latest_candidate_start, freq="1D", tz="UTC"):
        end = start + pd.Timedelta(days=HC.TEST_DAYS)
        stats = _btc_window_stats(btc_5m, start, end)
        if stats is None:
            continue
        candidates.append({"start": start, "end": end, **stats})
    if not candidates:
        return [primary]

    down = min(candidates, key=lambda x: x["ret_pct"])
    fold2 = FoldSpec(
        name="fold2_down_red_week",
        test_start=down["start"].isoformat(),
        test_end=down["end"].isoformat(),
        purpose="regime stress",
        reason="lowest BTC 7-day close-to-close return before the primary fold",
        btc_return_pct=round(float(down["ret_pct"]), 4),
        btc_range_pct=round(float(down["range_pct"]), 4),
    )

    remaining = [c for c in candidates if c["start"] != down["start"]]
    non_negative = [c for c in remaining if c["ret_pct"] >= 0]
    if non_negative:
        side = min(non_negative, key=lambda x: (abs(x["ret_pct"]), x["range_pct"]))
        reason = "non-negative BTC week closest to sideways, tie-broken by lower range"
    else:
        side = max(remaining or candidates, key=lambda x: x["ret_pct"])
        reason = "least-bad BTC week available; no non-negative earlier week found"
    fold3 = FoldSpec(
        name="fold3_sideways_bull_week",
        test_start=side["start"].isoformat(),
        test_end=side["end"].isoformat(),
        purpose="regime stress",
        reason=reason,
        btc_return_pct=round(float(side["ret_pct"]), 4),
        btc_range_pct=round(float(side["range_pct"]), 4),
    )
    return [primary, fold2, fold3][:max_folds]


def choose_exec_v2_folds(
    df: pd.DataFrame,
    *,
    primary_days: int = 1,
    spring_days: int = 14,
    max_folds: int = 3,
) -> list[FoldSpec]:
    """Leak-free v2 fold plan requested for the probability sandbox.

    Fold 1 is the latest available day in the built dataset. Folds 2-3 are
    14-day spring stress windows selected from BTC behavior: the worst
    close-to-close spring drawdown and the closest-to-sideways spring window.
    """
    base_time = pd.to_datetime(df["base_time"], utc=True)
    latest = base_time.max().floor("5min") + pd.Timedelta(minutes=5)
    primary_start = latest - pd.Timedelta(days=primary_days)
    primary = FoldSpec(
        name="fold1_latest_1d",
        test_start=primary_start.isoformat(),
        test_end=latest.isoformat(),
        purpose="fresh crash/check",
        reason=f"latest available {primary_days}-day window in the leak-free executable dataset",
    )
    if max_folds <= 1:
        return [primary]

    btc_5m = prepare_btc_frames()["5m"]
    data_min = base_time.min().ceil("1D")
    data_max = primary_start - pd.Timedelta(days=spring_days)
    spring_start = max(pd.Timestamp("2026-03-01T00:00:00Z"), data_min)
    spring_end = min(pd.Timestamp("2026-05-15T00:00:00Z"), data_max)
    candidates: list[dict] = []
    if spring_start <= spring_end:
        for start in pd.date_range(spring_start, spring_end, freq="1D", tz="UTC"):
            end = start + pd.Timedelta(days=spring_days)
            stats = _btc_window_stats(btc_5m, start, end)
            if stats is None:
                continue
            candidates.append({"start": start, "end": end, **stats})
    if not candidates:
        return choose_folds(df, max_folds=max_folds)

    red = min(candidates, key=lambda x: x["ret_pct"])
    fold2 = FoldSpec(
        name="fold2_spring_red_14d",
        test_start=red["start"].isoformat(),
        test_end=red["end"].isoformat(),
        purpose="spring regime stress",
        reason=f"lowest BTC {spring_days}-day close-to-close return in spring candidate range",
        btc_return_pct=round(float(red["ret_pct"]), 4),
        btc_range_pct=round(float(red["range_pct"]), 4),
    )

    remaining = [
        c
        for c in candidates
        if c["start"] != red["start"] and (c["end"] <= red["start"] or c["start"] >= red["end"])
    ]
    if not remaining:
        remaining = [c for c in candidates if c["start"] != red["start"]]
    non_negative = [c for c in remaining if c["ret_pct"] >= 0]
    if non_negative:
        flat = min(non_negative, key=lambda x: (abs(x["ret_pct"]), x["range_pct"]))
        reason = f"non-negative BTC {spring_days}-day spring window closest to flat"
    else:
        flat = min(remaining or candidates, key=lambda x: abs(x["ret_pct"]))
        reason = f"BTC {spring_days}-day spring window closest to flat"
    fold3 = FoldSpec(
        name="fold3_spring_sideways_14d",
        test_start=flat["start"].isoformat(),
        test_end=flat["end"].isoformat(),
        purpose="spring sideways stress",
        reason=reason,
        btc_return_pct=round(float(flat["ret_pct"]), 4),
        btc_range_pct=round(float(flat["range_pct"]), 4),
    )
    return [primary, fold2, fold3][:max_folds]


def split_masks(df: pd.DataFrame, fold: FoldSpec) -> dict[str, np.ndarray | pd.Timestamp]:
    base_time = pd.to_datetime(df["base_time"], utc=True)
    if "exit_time" in df.columns:
        target_end = pd.to_datetime(df["exit_time"], utc=True)
    else:
        horizon_delta = pd.to_timedelta(df["horizon_minutes"].astype("int64"), unit="min")
        target_end = base_time + horizon_delta
    test_start = fold.start_ts()
    test_end = fold.end_ts()
    embargo = pd.Timedelta(minutes=HC.EMBARGO_MIN)

    test = (base_time >= test_start) & (base_time < test_end)
    eligible = target_end < (test_start - embargo)
    eligible_times = np.array(sorted(base_time[eligible].unique()))
    if len(eligible_times) < 10:
        raise RuntimeError(f"{fold.name}: not enough pre-test rows for train/validation split")

    cut_idx = max(1, int(len(eligible_times) * (1.0 - HC.VALIDATION_FRACTION)))
    val_start = pd.Timestamp(eligible_times[cut_idx]).tz_localize("UTC") if pd.Timestamp(eligible_times[cut_idx]).tzinfo is None else pd.Timestamp(eligible_times[cut_idx])
    val = eligible & (base_time >= val_start)
    train = eligible & (target_end < (val_start - embargo))
    purged = eligible & ~(train | val)
    return {
        "train": train.to_numpy(),
        "val": val.to_numpy(),
        "test": test.to_numpy(),
        "purged": purged.to_numpy(),
        "val_start": val_start,
    }
