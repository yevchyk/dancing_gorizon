"""Live engine for the fast_v2 Combo00/Flat strategies.

This is the live counterpart of the research configs tested in
run_live_config_matrix.py:

* PulseClean3_idx0.00/0.05: at least 3 worthy fast_v2 models agree on one side,
  with no opposite-side worthy signal, exit at 10m.
* Flat670 / FlatStrict: high-win 2m single-model flow.
* No green harvest. The LiveTrader deadline-close owns exits.

The engine returns Signal-like objects for LiveTrader and builds its own
fast_v2 320-column hybrid curve snapshot, because the legacy live trader's
300-column CurveBuilder is not compatible with these models.
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

WORTHY = {
    "fast_v2_up_10m": ("up_10m", "up", 1, 0.77),
    "fast_v2_up_8m": ("up_8m", "up", 1, 0.77),
    "fast_v2_up_2m": ("up_2m", "up", 1, 0.92),
    "fast_v2_down_10m": ("down_10m", "down", -1, 0.82),
    "fast_v2_down_8m": ("down_8m", "down", -1, 0.83),
    "fast_v2_down_2m": ("down_2m", "down", -1, 0.92),
}

TOXIC_LAST24_2M5 = (
    "WLD_USDT_SWAP",
    "BEAT_USDT_SWAP",
    "APR_USDT_SWAP",
    "SUSHI_USDT_SWAP",
    "GRASS_USDT_SWAP",
)

TOXIC_LAST24_2M8 = TOXIC_LAST24_2M5 + (
    "MERL_USDT_SWAP",
    "ZEC_USDT_SWAP",
    "LIGHT_USDT_SWAP",
)

PROFILES = {
    "combo00_flat670": {
        "pulse_index": 0.00,
        "pulse_min_count": 3,
        "pulse_size_mult": 8.0,
        "flat_up_threshold": 0.93,
        "flat_down_threshold": 0.94,
        "flat_size_mult": 3.0,
        "include_pulse": True,
        "include_flat": True,
    },
    "combo00_flatstrict": {
        "pulse_index": 0.00,
        "pulse_min_count": 3,
        "pulse_size_mult": 8.0,
        "flat_up_threshold": 0.94,
        "flat_down_threshold": 0.94,
        "flat_size_mult": 3.0,
        "include_pulse": True,
        "include_flat": True,
    },
    "combo01_tight_toxic5": {
        "pulse_index": 0.00,
        "pulse_min_count": 3,
        "pulse_size_mult": 8.0,
        "flat_up_threshold": 0.96,
        "flat_down_threshold": 0.95,
        "flat_size_mult": 3.0,
        "include_pulse": True,
        "include_flat": True,
        "blocked_symbols": TOXIC_LAST24_2M5,
    },
    "combo01_tight_toxic8": {
        "pulse_index": 0.00,
        "pulse_min_count": 3,
        "pulse_size_mult": 8.0,
        "flat_up_threshold": 0.96,
        "flat_down_threshold": 0.95,
        "flat_size_mult": 3.0,
        "include_pulse": True,
        "include_flat": True,
        "blocked_symbols": TOXIC_LAST24_2M8,
    },
    "pulse00": {
        "pulse_index": 0.00,
        "pulse_min_count": 3,
        "pulse_size_mult": 8.0,
        "include_pulse": True,
        "include_flat": False,
        "exit_horizon": "8m",   # was "10m" — 8m beats 10m on okx_liquid 3d sim
    },
    "pulse05": {
        "pulse_index": 0.05,
        "pulse_min_count": 3,
        "pulse_size_mult": 8.0,
        "include_pulse": True,
        "include_flat": False,
    },
    # Only Forward: single long flow from the up_2m model, but with an 8m fixed
    # exit. Recent okx_liquid sims showed the 2m signal catches the start of
    # the impulse, while the 8m hold captures more of the move.
    "only_forward": {
        "pulse_index": 0.00,
        "pulse_min_count": 3,
        "pulse_size_mult": 0.0,
        "flat_up_threshold": 0.93,
        "flat_down_threshold": 2.0,
        "flat_size_mult": 1.0,
        "flat_exit_horizon": "8m",
        "flat_model_label": "OnlyForward_up2m_p093_exit8m",
        "include_pulse": False,
        "include_flat": True,
        "sides": ("long",),
    },
    # --- secondary turnover engines (run alongside the Unicorn core at small size) ---
    # B: Unicorn restricted to the long side, smaller size. Mostly overlaps the
    # core's longs (dedup gives the core priority), so it only adds the marginal
    # longs the core's top-N missed.
    "up3": {
        "pulse_index": 0.00,
        "pulse_min_count": 3,
        "pulse_size_mult": 3.0,
        "include_pulse": True,
        "include_flat": False,
        "sides": ("long",),
        "base_stake_usd": 10.0,
    },
    # "Pulse" — Pulse2 (>=2 agree) idx0.00: the high-turnover engine. Lower
    # quality / much higher volume than the >=3 Unicorn core. Small size ($5x3)
    # for a live test alongside the core; the core keeps symbol priority.
    "pulse": {
        "pulse_index": 0.00,
        "pulse_min_count": 2,
        "pulse_size_mult": 3.0,
        "include_pulse": True,
        "include_flat": False,
        "base_stake_usd": 5.0,
    },
    # "Дриль / Drill" — the crisis SHORT engine: down_8m AND down_10m both >=0.80
    # -> short, 10m hold. Built on the worthy machinery: only down_8m/down_10m can
    # vote (others overridden to never fire), min_count=2 forces BOTH, short only.
    # 72h holdout: +26% / win 0.55 / green 3-of-4 days. Runs alongside Unicorn for
    # a two-sided book (Unicorn longs, Drill shorts).
    "drill": {
        "pulse_index": 0.00,
        "pulse_min_count": 2,
        "pulse_size_mult": 3.0,
        "include_pulse": True,
        "include_flat": False,
        "sides": ("short",),
        "base_stake_usd": 5.0,
        "worthy_overrides": {
            "fast_v2_up_10m": 2.0, "fast_v2_up_8m": 2.0, "fast_v2_up_2m": 2.0,
            "fast_v2_down_2m": 2.0,            # exclude: only 8m & 10m vote
            "fast_v2_down_10m": 0.80, "fast_v2_down_8m": 0.80,
        },
    },
    # C: asymmetric — raise the abundant/weak UP thresholds toward where their
    # edge concentrates; keep DOWN at base. Tighter, higher-conviction.
    "asym": {
        "pulse_index": 0.00,
        "pulse_min_count": 3,
        "pulse_size_mult": 3.0,
        "include_pulse": True,
        "include_flat": False,
        "base_stake_usd": 10.0,
        "worthy_overrides": {
            "fast_v2_up_10m": 0.78,
            "fast_v2_up_8m": 0.84,
            "fast_v2_up_2m": 0.93,
        },
    },
}

# Named engine stacks for multi-engine live runs. First entry = the core (highest
# priority on symbol conflicts); later entries add non-overlapping turnover.
STACKS = {
    "unicorn_solo": ["pulse00"],
    "only_forward": ["only_forward"],
    "unicorn_plus": ["pulse00", "up3", "asym"],
    "unicorn_pulse": ["pulse00", "pulse"],   # core + the high-turnover Pulse2 test
    "unicorn_drill": ["pulse00", "drill"],   # two-sided book: Unicorn up + Drill down
}


@dataclass
class FastComboSignal:
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
    engine: str = ""              # profile that produced it (for per-engine logs)
    size_usd: float | None = None  # absolute $ size; None => caller's global stake


class FastComboEngine:
    horizon_minutes = {"2m": 2, "5m": 5, "8m": 8, "10m": 10}

    def __init__(self, profile: str = "combo00_flat670") -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown fast combo profile: {profile}")
        self.profile = profile
        self.cfg = PROFILES[profile]
        self.base_stake_usd = self.cfg.get("base_stake_usd")   # None => caller's global stake
        self.sides = tuple(self.cfg.get("sides", ("long", "short")))
        self.curve = FastCurve(
            FC.CURVE_POINTS,
            FC.CURVE_MIN_STEP_MIN,
            FC.CURVE_MAX_DEPTH_MIN,
            FC.CURVE_SEGMENTS,
        )
        self.columns = self.curve.columns()
        self._models: dict[str, tuple[object, list[str]]] = {}
        for h in FC.HORIZONS:
            for kind in ("up", "down"):
                name = f"{kind}_{h.label}"
                model_path = FC.FAST_MODELS_DIR / f"{name}.joblib"
                cols_path = FC.FAST_MODELS_DIR / f"{name}_columns.joblib"
                self._models[name] = (joblib.load(model_path), joblib.load(cols_path))

    def describe(self) -> str:
        flat = "off"
        if self.cfg.get("include_flat", False):
            flat_exit = str(self.cfg.get("flat_exit_horizon", "2m"))
            flat = (
                f"up2m>={self.cfg['flat_up_threshold']:.2f}, "
                f"down2m>={self.cfg['flat_down_threshold']:.2f} "
                f"exit={flat_exit} x{self.cfg['flat_size_mult']:.1f}"
            )
        blocked = tuple(self.cfg.get("blocked_symbols", ()))
        block = f"; blocked={len(blocked)}" if blocked else ""
        exit_h = self.cfg.get("exit_horizon", "10m")
        if not self.cfg.get("include_pulse", True):
            pulse = "pulse=off"
        else:
            pulse = (
                f"PulseClean3 idx={self.cfg.get('pulse_index', 0):.2f} "
                f"x{self.cfg.get('pulse_size_mult', 1):.1f}; exit={exit_h}"
            )
        return (
            f"{self.profile}: {pulse}; flat={flat}; "
            f"no-harvest{block}; safety OCO={SAFETY_OCO_PCT:.1%}"
        )

    def build_watchlist(self, store: CandleStore, top_n: int = 0,
                        logger=None) -> list[str]:
        """Use the actual fast_v2 trained universe, not arbitrary live liquidity.

        The trainable symbols are the symbols with fast_v2 chunks. A top_n of 0
        means "all trained symbols"; otherwise we rank only that trained set by
        recent local quote volume.
        """
        trained = {p.stem for p in FC.FAST_CHUNKS_DIR.glob("*.parquet")}
        blacklist = set(C.BLACKLIST_SYMBOLS)
        vols: list[tuple[str, float]] = []
        missing = invalid = 0
        for sym in sorted(trained):
            if sym in blacklist:
                continue
            candles = store.load(sym)
            if candles is None or candles.empty:
                missing += 1
                continue
            tail = candles.iloc[-1440:]
            vol = float((tail["close"] * tail["volume"]).sum())
            if not np.isfinite(vol):
                invalid += 1
                continue
            vols.append((sym, vol))
        vols.sort(key=lambda item: item[1], reverse=True)
        limit = int(top_n or 0)
        watch = [sym for sym, _ in (vols[:limit] if limit > 0 else vols)]
        if logger is not None:
            scope = f"top {limit}" if limit > 0 else "all"
            logger.event(
                f"watchlist: fast_v2 trained universe {scope}: {len(watch)} symbols "
                f"(trained={len(trained)}, missing={missing}, invalid={invalid}, "
                f"blacklisted={len(blacklist)})"
            )
        return watch

    def snapshot(self, store: CandleStore, symbols: list[str], now: pd.Timestamp) -> pd.DataFrame:
        rows = []
        anchor_ns = np.array([int(pd.Timestamp(now).value)], dtype="int64")
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
            if candles.index[entry_idx] < pd.Timestamp(now) - pd.Timedelta(minutes=1):
                continue
            entry = float(close[entry_idx])
            if not np.isfinite(entry) or entry <= 0:
                continue
            row = {"symbol": sym, "anchor_time": now, "entry_price": entry}
            row.update({col: float(feats[0, i]) for i, col in enumerate(self.columns)})
            rows.append(row)
        return pd.DataFrame(rows)

    def _score_probabilities(self, feat: pd.DataFrame) -> pd.DataFrame:
        out = feat[["symbol", "anchor_time", "entry_price"]].copy()
        for name, (model, cols) in self._models.items():
            out[f"p_{name}"] = model.predict_proba(feat[cols])[:, 1]
        return out

    def _worthy_threshold(self, base: float) -> float:
        idx = float(self.cfg.get("pulse_index", 0.0))
        return base + idx * (1.0 - base)

    def _size_usd(self, size_mult: float) -> float | None:
        """Absolute $ size when the profile carries its own stake, else None so
        LiveTrader applies its global trade_size_usd * size_mult."""
        if self.base_stake_usd is None:
            return None
        return float(self.base_stake_usd) * float(size_mult)

    def decide(self, feat: pd.DataFrame, top_n: int = 3) -> list[FastComboSignal]:
        if feat.empty:
            return []
        s = self._score_probabilities(feat)
        signals: list[FastComboSignal] = []

        for r in s.itertuples(index=False):
            sym = str(r.symbol)
            if sym in self.cfg.get("blocked_symbols", ()):
                continue
            up_count = down_count = 0
            up_score = down_score = 0.0
            up_probs: list[float] = []
            down_probs: list[float] = []

            overrides = self.cfg.get("worthy_overrides", {})
            for full_name, (model_name, side_name, side, base_thr) in WORTHY.items():
                prob = float(getattr(r, f"p_{model_name}"))
                thr = self._worthy_threshold(overrides.get(full_name, base_thr))
                if prob < thr:
                    continue
                headroom = max(0.0, (prob - thr) / max(1e-9, 1.0 - thr))
                if side > 0:
                    up_count += 1
                    up_score += headroom
                    up_probs.append(prob)
                else:
                    down_count += 1
                    down_score += headroom
                    down_probs.append(prob)

            if self.cfg.get("include_pulse", True):
                min_count = int(self.cfg.get("pulse_min_count", 3))
                mult = float(self.cfg.get("pulse_size_mult", 1.0))
                exit_h = str(self.cfg.get("exit_horizon", "10m"))
                model_label = f"PulseClean3_idx0.00_exit{exit_h}"
                if "long" in self.sides and up_count >= min_count and down_count == 0:
                    signals.append(FastComboSignal(
                        symbol=sym,
                        model=model_label,
                        side="long",
                        horizon=exit_h,
                        move_pct=SAFETY_OCO_PCT,
                        prob=max(up_probs) if up_probs else 0.0,
                        score=100.0 + up_score,
                        spread=up_score,
                        agree=up_count,
                        size_mult=mult,
                        source="pulse",
                        engine=self.profile,
                        size_usd=self._size_usd(mult),
                    ))
                if "short" in self.sides and down_count >= min_count and up_count == 0:
                    signals.append(FastComboSignal(
                        symbol=sym,
                        model=model_label,
                        side="short",
                        horizon=exit_h,
                        move_pct=SAFETY_OCO_PCT,
                        prob=max(down_probs) if down_probs else 0.0,
                        score=100.0 + down_score,
                        spread=down_score,
                        agree=down_count,
                        size_mult=mult,
                        source="pulse",
                        engine=self.profile,
                        size_usd=self._size_usd(mult),
                    ))

            if self.cfg.get("include_flat", False):
                up_thr = float(self.cfg["flat_up_threshold"])
                down_thr = float(self.cfg["flat_down_threshold"])
                fmult = float(self.cfg.get("flat_size_mult", 1.0))
                flat_exit = str(self.cfg.get("flat_exit_horizon", "2m"))
                up_model = str(self.cfg.get("flat_model_label", "fast_v2_up_2m"))
                down_model = str(self.cfg.get("flat_down_model_label", "fast_v2_down_2m"))
                up2 = float(getattr(r, "p_up_2m"))
                down2 = float(getattr(r, "p_down_2m"))
                if "long" in self.sides and up2 >= up_thr:
                    headroom = max(0.0, (up2 - up_thr) / max(1e-9, 1.0 - up_thr))
                    signals.append(FastComboSignal(
                        symbol=sym,
                        model=up_model,
                        side="long",
                        horizon=flat_exit,
                        move_pct=SAFETY_OCO_PCT,
                        prob=up2,
                        score=10.0 + headroom,
                        spread=up2 - down2,
                        agree=1,
                        size_mult=fmult,
                        source="flat",
                        engine=self.profile,
                        size_usd=self._size_usd(fmult),
                    ))
                if "short" in self.sides and down2 >= down_thr:
                    headroom = max(0.0, (down2 - down_thr) / max(1e-9, 1.0 - down_thr))
                    signals.append(FastComboSignal(
                        symbol=sym,
                        model=down_model,
                        side="short",
                        horizon=flat_exit,
                        move_pct=SAFETY_OCO_PCT,
                        prob=down2,
                        score=10.0 + headroom,
                        spread=down2 - up2,
                        agree=1,
                        size_mult=fmult,
                        source="flat",
                        engine=self.profile,
                        size_usd=self._size_usd(fmult),
                    ))

        signals.sort(key=lambda sig: sig.score, reverse=True)
        # One signal per symbol per scan. Pulse wins because its score starts at 100.
        picked: list[FastComboSignal] = []
        used_symbols: set[str] = set()
        for sig in signals:
            if sig.symbol in used_symbols:
                continue
            picked.append(sig)
            used_symbols.add(sig.symbol)
            if len(picked) >= top_n:
                break
        return picked
