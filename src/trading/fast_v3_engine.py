"""Live adapter for the fast_v3 v2 engines.

This module does not train anything. It loads the already-trained fast_v3
classifiers and applies the same rules used by the v2 simulation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .. import config as C
from ..database import CandleStore
from ..fast import config as FC
from ..fast.curve import FastCurve
from .timeutil import index_to_ns


SAFETY_OCO_PCT = 0.03

V3_HORIZONS = (
    (1, "1m", 12 * 60),
    (2, "2m", 24 * 60),
    (4, "4m", 3 * 24 * 60),
    (8, "8m", 30 * 24 * 60),
    (12, "12m", 45 * 24 * 60),
    (20, "20m", 60 * 24 * 60),
)
V3_LABELS = tuple(label for _minutes, label, _lookback in V3_HORIZONS)
V3_MODELS_DIR = C.MODELS_DIR / "fast_v3" / "base"
V3_DATASET = C.DATA_DIR / "fast_v3" / "datasets" / "master.parquet"


PROFILES = {
    "verkh_v2": {
        "kind": "flat_pool",
        "notional_usd": 30.0,
        # tuned 2026-06-03: dropped up_1m (move < fee), raised short legs; up_20m
        # KEPT at 0.85 -- it is the breadwinner, raising it to 0.90 starved the edge.
        "up_thresholds": {
            "2m": 0.95,
            "4m": 0.95,
            "8m": 0.92,
            "12m": 0.90,
            "20m": 0.85,
        },
        "apply_blacklist": False,
    },
    "unicorn_v2": {
        "kind": "agreement",
        "notional_usd": 60.0,
        "agreement_threshold": 0.85,
        "min_agree": 4,
        "exit_horizon": "20m",
        "apply_blacklist": False,
    },
    # INVERTED: when >=4 up-models scream -> go SHORT (contrarian crash mode).
    # Validated on 2026-06-02/03 crash window: +$32 vs normal -$113.
    # Use in bear/crash regime only; on bull it will lose.
    "unicorn_v2_inverted": {
        "kind": "agreement",
        "notional_usd": 60.0,
        "agreement_threshold": 0.85,
        "min_agree": 4,
        "exit_horizon": "20m",
        "inverted": True,
        "apply_blacklist": False,
    },
    "spread20_v1": {
        "kind": "spread_sign",
        "notional_usd": 30.0,
        "horizon": "20m",
        "spread_threshold": 0.38,
        "exit_horizon": "20m",
        "apply_blacklist": False,
    },
}

STACKS = {
    "v2_pair": ["unicorn_v2", "verkh_v2"],
    "inverted_solo": ["unicorn_v2_inverted"],
    "spread20_solo": ["spread20_v1"],
}


@dataclass
class FastV3Signal:
    symbol: str
    model: str
    side: str
    horizon: str
    move_pct: float
    prob: float
    score: float
    spread: float
    agree: int
    size_mult: float
    source: str
    engine: str = ""
    size_usd: float | None = None


class FastV3Engine:
    horizon_minutes = {label: minutes for minutes, label, _lookback in V3_HORIZONS}

    def __init__(self, profile: str = "verkh_v2") -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown fast_v3 profile: {profile}")
        self.profile = profile
        self.cfg = PROFILES[profile]
        self.curve = FastCurve(
            FC.CURVE_POINTS,
            FC.CURVE_MIN_STEP_MIN,
            FC.CURVE_MAX_DEPTH_MIN,
            FC.CURVE_SEGMENTS,
        )
        self.columns = self.curve.columns()
        self._models: dict[str, tuple[object, list[str]]] = {}
        for label in V3_LABELS:
            for side in ("up", "down"):
                name = f"{side}_{label}"
                self._models[name] = (
                    joblib.load(V3_MODELS_DIR / f"{name}.joblib"),
                    joblib.load(V3_MODELS_DIR / f"{name}_columns.joblib"),
                )

    def describe(self) -> str:
        kind = self.cfg["kind"]
        notional = float(self.cfg["notional_usd"])
        if kind == "flat_pool":
            th = ", ".join(
                f"up_{label}>={thr:.2f}"
                for label, thr in self.cfg["up_thresholds"].items()
            )
            return f"{self.profile}: flat long pool; {th}; own exits; ${notional:.0f} notional"
        if kind == "spread_sign":
            horizon = str(self.cfg["horizon"])
            threshold = float(self.cfg["spread_threshold"])
            exit_h = str(self.cfg["exit_horizon"])
            return (
                f"{self.profile}: spread sign p_up_{horizon}-p_down_{horizon} "
                f"abs>={threshold:.2f}; exit={exit_h}; both sides; ${notional:.0f} notional"
            )
        return (
            f"{self.profile}: agreement >= {self.cfg['min_agree']} "
            f"@ {self.cfg['agreement_threshold']:.2f}; exit={self.cfg['exit_horizon']}; "
            f"both sides; ${notional:.0f} notional"
        )

    def build_watchlist(self, store: CandleStore, top_n: int = 0, logger=None) -> list[str]:
        if V3_DATASET.exists():
            try:
                symbols = set(pd.read_parquet(V3_DATASET, columns=["symbol"])["symbol"].unique())
            except Exception:
                symbols = set(store.symbols())
        else:
            symbols = set(store.symbols())

        missing = invalid = 0
        vols: list[tuple[str, float]] = []
        blacklist = set(C.BLACKLIST_SYMBOLS) if self.cfg.get("apply_blacklist", False) else set()
        for sym in sorted(symbols):
            if sym in blacklist:
                continue
            candles = store.load(sym)
            if candles is None or candles.empty:
                missing += 1
                continue
            tail = candles.sort_index().iloc[-1440:]
            vol = float((tail["close"] * tail["volume"]).sum())
            if not np.isfinite(vol):
                invalid += 1
                continue
            vols.append((sym, vol))

        vols.sort(key=lambda item: item[1], reverse=True)
        limit = int(top_n or 0)
        watch = [sym for sym, _vol in (vols[:limit] if limit > 0 else vols)]
        if logger is not None:
            scope = f"top {limit}" if limit > 0 else "all"
            logger.event(
                f"watchlist: fast_v3 trained universe {scope}: {len(watch)} symbols "
                f"(trained={len(symbols)}, missing={missing}, invalid={invalid}, "
                f"blacklisted={len(blacklist)})"
            )
        return watch

    def snapshot(self, store: CandleStore, symbols: list[str], now: pd.Timestamp) -> pd.DataFrame:
        rows = []
        anchor = pd.Timestamp(now)
        anchor_ns = np.array([int(anchor.value)], dtype="int64")
        for sym in symbols:
            candles = store.load(sym)
            if candles is None or candles.empty:
                continue
            candles = candles.sort_index()
            ts_ns = index_to_ns(candles.index)
            close = candles["close"].to_numpy("float64")
            feats, valid = self.curve.build_matrix(ts_ns, close, anchor_ns)
            if not bool(valid[0]):
                continue
            entry_idx = int(np.searchsorted(ts_ns, anchor_ns[0], side="right")) - 1
            if entry_idx < 0:
                continue
            if candles.index[entry_idx] < anchor - pd.Timedelta(minutes=1):
                continue
            entry = float(close[entry_idx])
            if not np.isfinite(entry) or entry <= 0:
                continue
            row = {"symbol": sym, "anchor_time": anchor, "entry_price": entry}
            row.update({col: float(feats[0, i]) for i, col in enumerate(self.columns)})
            rows.append(row)
        return pd.DataFrame(rows)

    def _score_probabilities(self, feat: pd.DataFrame) -> pd.DataFrame:
        out = feat[["symbol", "anchor_time", "entry_price"]].copy()
        for name, (model, cols) in self._models.items():
            out[f"p_{name}"] = model.predict_proba(feat[cols])[:, 1]
        return out

    @staticmethod
    def _headroom(prob: float, threshold: float) -> float:
        return max(0.0, (prob - threshold) / max(1e-9, 1.0 - threshold))

    def _flat_pool(self, row) -> list[FastV3Signal]:
        out: list[FastV3Signal] = []
        notional = float(self.cfg["notional_usd"])
        for label, threshold in self.cfg["up_thresholds"].items():
            prob = float(getattr(row, f"p_up_{label}"))
            if prob < float(threshold):
                continue
            hr = self._headroom(prob, float(threshold))
            out.append(FastV3Signal(
                symbol=str(row.symbol),
                model=f"verkh_v2_up_{label}_p{threshold:.2f}",
                side="long",
                horizon=label,
                move_pct=SAFETY_OCO_PCT,
                prob=prob,
                score=20.0 + hr,
                spread=hr,
                agree=1,
                size_mult=notional / 10.0,
                source="verkh_v2",
                engine=self.profile,
                size_usd=notional,
            ))
        return out

    def _agreement(self, row) -> list[FastV3Signal]:
        threshold = float(self.cfg["agreement_threshold"])
        min_agree = int(self.cfg["min_agree"])
        notional = float(self.cfg["notional_usd"])
        up_count = down_count = 0
        up_score = down_score = 0.0
        up_probs: list[float] = []
        down_probs: list[float] = []
        for label in V3_LABELS:
            up = float(getattr(row, f"p_up_{label}"))
            down = float(getattr(row, f"p_down_{label}"))
            if up >= threshold:
                up_count += 1
                up_score += self._headroom(up, threshold)
                up_probs.append(up)
            if down >= threshold:
                down_count += 1
                down_score += self._headroom(down, threshold)
                down_probs.append(down)

        exit_h = str(self.cfg["exit_horizon"])
        inverted = bool(self.cfg.get("inverted", False))
        src = self.profile

        if up_count >= min_agree and down_count == 0:
            # normal -> long; inverted -> short
            actual_side = "short" if inverted else "long"
            return [FastV3Signal(
                symbol=str(row.symbol),
                model=f"{src}_{actual_side}_agree{up_count}_exit{exit_h}",
                side=actual_side,
                horizon=exit_h,
                move_pct=SAFETY_OCO_PCT,
                prob=max(up_probs),
                score=100.0 + up_score,
                spread=up_score,
                agree=up_count,
                size_mult=notional / 10.0,
                source=src,
                engine=self.profile,
                size_usd=notional,
            )]
        if down_count >= min_agree and up_count == 0:
            # normal -> short; inverted -> long
            actual_side = "long" if inverted else "short"
            return [FastV3Signal(
                symbol=str(row.symbol),
                model=f"{src}_{actual_side}_agree{down_count}_exit{exit_h}",
                side=actual_side,
                horizon=exit_h,
                move_pct=SAFETY_OCO_PCT,
                prob=max(down_probs),
                score=100.0 + down_score,
                spread=down_score,
                agree=down_count,
                size_mult=notional / 10.0,
                source=src,
                engine=self.profile,
                size_usd=notional,
            )]
        return []

    def _spread_sign(self, row) -> list[FastV3Signal]:
        label = str(self.cfg["horizon"])
        threshold = float(self.cfg["spread_threshold"])
        exit_h = str(self.cfg["exit_horizon"])
        notional = float(self.cfg["notional_usd"])
        up = float(getattr(row, f"p_up_{label}"))
        down = float(getattr(row, f"p_down_{label}"))
        spread = up - down
        if abs(spread) < threshold:
            return []
        side = "long" if spread > 0 else "short"
        prob = up if side == "long" else down
        score = 50.0 + abs(spread)
        return [FastV3Signal(
            symbol=str(row.symbol),
            model=f"{self.profile}_{side}_{label}_spread{threshold:.2f}",
            side=side,
            horizon=exit_h,
            move_pct=SAFETY_OCO_PCT,
            prob=prob,
            score=score,
            spread=spread,
            agree=1,
            size_mult=notional / 10.0,
            source=self.profile,
            engine=self.profile,
            size_usd=notional,
        )]

    def decide(self, feat: pd.DataFrame, top_n: int = 3) -> list[FastV3Signal]:
        if feat.empty:
            return []
        scored = self._score_probabilities(feat)
        signals: list[FastV3Signal] = []
        for row in scored.itertuples(index=False):
            if self.cfg["kind"] == "flat_pool":
                signals.extend(self._flat_pool(row))
            elif self.cfg["kind"] == "spread_sign":
                signals.extend(self._spread_sign(row))
            else:
                signals.extend(self._agreement(row))

        signals.sort(key=lambda sig: sig.score, reverse=True)
        picked: list[FastV3Signal] = []
        used: set[str] = set()
        for sig in signals:
            if sig.symbol in used:
                continue
            picked.append(sig)
            used.add(sig.symbol)
            if len(picked) >= top_n:
                break
        return picked


class FastV3Stack:
    horizon_minutes = FastV3Engine.horizon_minutes

    def __init__(self, profiles: list[str]) -> None:
        if not profiles:
            raise ValueError("FastV3Stack needs at least one profile")
        self.engines = [FastV3Engine(profile) for profile in profiles]
        self.profile = "stack:" + "+".join(profiles)

    @classmethod
    def from_stack(cls, name: str) -> "FastV3Stack":
        if name not in STACKS:
            raise ValueError(f"unknown fast_v3 stack: {name} (have {list(STACKS)})")
        return cls(STACKS[name])

    def build_watchlist(self, store, top_n: int = 0, logger=None) -> list[str]:
        return self.engines[0].build_watchlist(store, top_n, logger)

    def snapshot(self, store, symbols: list[str], now: pd.Timestamp) -> pd.DataFrame:
        return self.engines[0].snapshot(store, symbols, now)

    def decide(self, feat: pd.DataFrame, top_n: int = 3) -> list[FastV3Signal]:
        claimed: set[str] = set()
        out: list[FastV3Signal] = []
        for eng in self.engines:
            for sig in eng.decide(feat, top_n=top_n):
                if sig.symbol in claimed:
                    continue
                claimed.add(sig.symbol)
                out.append(sig)
        return out

    def describe(self) -> str:
        return "  ||  ".join(f"[{e.profile}] {e.describe()}" for e in self.engines)
