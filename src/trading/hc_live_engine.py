"""Live adapter for the leak-free horizon-conditioned HC models.

Live timing matches the executable backtests:

    scan/entry at T
    features are built at base_time = T - entry_delay_min
    no feature lookup uses candles newer than base_time

The adapter returns signals in the same shape as the other trust/live engines,
so LiveTrader can reuse the existing OKX executor, position manager, logging,
deadline safety net, and OCO handling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .. import config as C
from ..database import CandleStore
from ..hc import config as HC
from ..hc.data import (
    _build_feature_matrix,
    prepare_btc_frames,
    prepare_timeframes,
    to_ns,
)


@dataclass
class HCLiveSignal:
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
    engine: str = "hc_live"
    size_usd: float | None = None
    threshold: float = 0.0


class HCLiveEngine:
    horizon_minutes = {f"{int(h)}m": int(h) for h in HC.HORIZON_ANCHORS}
    default_system_name = "Танцюючий Горизонт"
    horizon_exit_only = True

    def __init__(
        self,
        *,
        model_dir: Path = Path("models/hc_exec_stride120_nonoverlap"),
        high: float = 0.90,
        opp_cap: float = 0.20,
        horizon_min: int = 30,
        horizon_max: int = 90,
        horizons: tuple[int, ...] | list[int] | None = None,
        entry_delay_min: int = HC.EXEC_ENTRY_DELAY_MIN,
        notional_usd: float | None = None,
        thresholds_by_horizon: dict[int, float] | None = None,
        profile: str = "plain_mid_p90_opp20",
        system_name: str = default_system_name,
        max_legs: int = 1,
        conviction: bool = False,
        selection_mode: str = "plain",
        spread_floor: float | None = None,
        bdw_raw: float = 0.80,
        bdw_opp: float = 0.05,
    ) -> None:
        self.system_name = str(system_name)
        # Live policy after the ZEC incident: slots are diversified across symbols.
        # Keep the constructor arg for CLI compatibility, but never stack horizons
        # on the same coin in live decisions.
        self.max_legs = 1
        self.conviction = bool(conviction)
        self.model_dir = Path(model_dir)
        self.high = float(high)
        self.opp_cap = float(opp_cap)
        self.horizon_min = int(horizon_min)
        self.horizon_max = int(horizon_max)
        if horizons is None:
            self.horizons = tuple(int(h) for h in HC.HORIZON_ANCHORS)
        else:
            self.horizons = tuple(sorted(dict.fromkeys(int(h) for h in horizons)))
        if not self.horizons:
            raise ValueError("HC live horizons must not be empty")
        self.horizon_minutes = {f"{int(h)}m": int(h) for h in self.horizons}
        self.entry_delay_min = int(entry_delay_min)
        self.notional_usd = None if notional_usd is None else float(notional_usd)
        self.thresholds_by_horizon = (
            {int(k): float(v) for k, v in thresholds_by_horizon.items()}
            if thresholds_by_horizon else None
        )
        self.profile = profile
        self.selection_mode = str(selection_mode or "plain").strip().lower()
        if self.selection_mode not in {"plain", "squeezer", "quality", "bad_day_worker"}:
            raise ValueError(f"unknown HC selection_mode={selection_mode!r}")
        self.spread_floor = None if spread_floor is None else float(spread_floor)
        # bad_day_worker: dry-pocket extractor for calm/bad regimes.
        # Discovered on NEW 2026-06-05: p_dir >= 0.80 AND p_opp <= 0.05
        # (n=17, win 47.1%, avg net +1.332%).  Distinct AND gate, not the squeezer OR.
        self.bdw_raw = float(bdw_raw)
        self.bdw_opp = float(bdw_opp)
        self.last_near_misses: list[str] = []
        self._models = self._load_models()

    def _floor_desc(self) -> str:
        """Human-readable description of the active candidate gate."""
        if self.selection_mode == "plain":
            p = (
                f"per-horizon[{len(self.thresholds_by_horizon)}]"
                if self.thresholds_by_horizon else f"{self.high:.2f}"
            )
            return f"p_dir>={p} AND p_opp<={self.opp_cap:.2f}"
        if self.selection_mode == "bad_day_worker":
            return f"p_dir>={self.bdw_raw:.2f} AND p_opp<={self.bdw_opp:.2f}"
        raw, spread = self._mode_floors()
        sp = "off" if spread is None else f"{float(spread):.2f}"
        return f"p_dir>={raw:.2f} OR spread>={sp}"

    def describe(self) -> str:
        size = "caller-size" if self.notional_usd is None else f"${self.notional_usd:.2f} notional"
        return (
            f"{self.system_name} / HC {self.profile}: "
            f"mode={self.selection_mode}, gate=[{self._floor_desc()}], "
            f"h={self.horizon_min}-{self.horizon_max}m, "
            f"grid={','.join(str(h) for h in self.horizons)}, "
            f"legs/coin=1, conviction={'on' if self.conviction else 'off'}, "
            f"features=T-{self.entry_delay_min}m, entry=T, {size}"
        )

    def _with_thresholds(self, feat: pd.DataFrame) -> pd.DataFrame:
        d = feat[feat["horizon_minutes"].between(self.horizon_min, self.horizon_max)].copy()
        if d.empty:
            return d
        if self.thresholds_by_horizon:
            d["threshold"] = d["horizon_minutes"].map(self.thresholds_by_horizon)
            d = d[d["threshold"].notna()].copy()
            d["threshold"] = d["threshold"].astype(float)
        else:
            d["threshold"] = self.high
        return d

    def _fold_names(self) -> list[str]:
        snapshot = self.model_dir / "config_snapshot.json"
        if snapshot.exists():
            data = json.loads(snapshot.read_text(encoding="utf-8"))
            names = [str(f["name"]) for f in data.get("folds", []) if "name" in f]
            if names:
                return names
        return sorted(p.name for p in self.model_dir.iterdir() if (p / "up.cbm").exists())

    def _load_models(self) -> list[tuple[str, CatBoostClassifier, CatBoostClassifier]]:
        out = []
        for fold in self._fold_names():
            up_path = self.model_dir / fold / "up.cbm"
            down_path = self.model_dir / fold / "down.cbm"
            if not up_path.exists() or not down_path.exists():
                continue
            up = CatBoostClassifier()
            up.load_model(up_path)
            down = CatBoostClassifier()
            down.load_model(down_path)
            out.append((fold, up, down))
        if not out:
            raise FileNotFoundError(f"No HC up/down models under {self.model_dir}")
        return out

    def build_watchlist(self, store: CandleStore, top_n: int = 0, logger=None) -> list[str]:
        data = json.loads(HC.UNIVERSE_PATH.read_text(encoding="utf-8"))
        universe = data.get("symbols", data) if isinstance(data, dict) else data
        blacklist = C.hc_blacklist_symbols()
        vols: list[tuple[str, float]] = []
        missing = 0
        for sym in sorted(str(s) for s in universe):
            if sym in blacklist:
                continue
            candles = store.load(sym)
            if candles is None or candles.empty:
                missing += 1
                continue
            tail = candles.iloc[-1440:]
            vol = float((tail["close"] * tail["volume"]).sum())
            if np.isfinite(vol):
                vols.append((sym, vol))
        vols.sort(key=lambda item: item[1], reverse=True)
        limit = int(top_n or 0)
        watch = [sym for sym, _ in (vols[:limit] if limit > 0 else vols)]
        if logger is not None:
            scope = f"top {limit}" if limit > 0 else "all"
            logger.event(
                f"watchlist: HC universe {scope}: {len(watch)} symbols "
                f"(universe={len(universe)}, missing={missing}, blacklisted={len(blacklist)})"
            )
        return watch

    def snapshot(self, store: CandleStore, symbols: list[str], now: pd.Timestamp) -> pd.DataFrame:
        now = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
        base_time = now - pd.Timedelta(minutes=self.entry_delay_min)
        anchors = pd.DatetimeIndex([base_time], tz="UTC")
        anchor_ns = to_ns(anchors)

        try:
            btc_frames = prepare_btc_frames()
        except Exception:
            return pd.DataFrame()

        rows: list[dict] = []
        no_horizon_cols = HC.FEATURE_COLUMNS[:-2]
        for sym in symbols:
            candles = store.load(sym)
            if candles is None or candles.empty:
                continue
            candles = candles.sort_index()
            entry_slice = candles[candles.index <= now]
            if entry_slice.empty:
                continue
            entry_time = entry_slice.index[-1]
            if entry_time < now - pd.Timedelta(minutes=2):
                continue
            entry_price = float(entry_slice["close"].iloc[-1])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            try:
                prepared = prepare_timeframes(candles, btc_frames)
                if not prepared:
                    continue
                features, valid = _build_feature_matrix(anchors, prepared, HC.N_POINTS)
            except Exception:
                continue
            if not bool(valid[0]):
                continue
            base_feature = features[0]
            for horizon in self.horizons:
                h = int(horizon)
                row = {
                    "symbol": sym,
                    "anchor_time": now,
                    "base_time": base_time,
                    "entry_price": entry_price,
                    "entry_source_time": entry_time,
                    "horizon_minutes": h,
                    "horizon_log": float(np.log1p(h)),
                }
                for i, col in enumerate(no_horizon_cols):
                    row[col] = float(base_feature[i])
                rows.append(row)

        if not rows:
            return pd.DataFrame()
        feat = pd.DataFrame(rows)
        x = feat[HC.FEATURE_COLUMNS]
        up_preds: list[np.ndarray] = []
        down_preds: list[np.ndarray] = []
        for _fold, up, down in self._models:
            up_preds.append(up.predict_proba(x)[:, 1].astype("float32"))
            down_preds.append(down.predict_proba(x)[:, 1].astype("float32"))
        out = feat[[
            "symbol",
            "anchor_time",
            "base_time",
            "entry_price",
            "entry_source_time",
            "horizon_minutes",
        ]].copy()
        out["up_prob"] = np.vstack(up_preds).mean(axis=0).astype("float32")
        out["down_prob"] = np.vstack(down_preds).mean(axis=0).astype("float32")
        return out

    @staticmethod
    def conviction_mult(spread: float) -> float:
        """Position-size multiplier from signal conviction (p_dir - p_opp).

        Matches the validated sim curve: 0.5x at spread 0.6 -> 2.0x at spread ~1.1.
        """
        return float(min(2.0, max(0.5, 0.5 + (float(spread) - 0.6) * 3.0)))

    def _mode_floors(self) -> tuple[float, float | None]:
        if self.selection_mode == "quality":
            # Ultra-quality guard: on OLD 2026-06-05 this sits out, while OLD
            # Jun1-4 still has a strong high-tail.  Keep it conservative even if
            # the CLI --high value is lower.
            raw_floor = max(float(self.high), 0.94)
            spread_floor = self.spread_floor if self.spread_floor is not None else 0.92
            return raw_floor, spread_floor
        if self.selection_mode == "squeezer":
            raw_floor = float(self.high)
            spread_floor = self.spread_floor if self.spread_floor is not None else 0.80
            return raw_floor, spread_floor
        return float(self.high), self.spread_floor

    def _candidate_mask(self, x: pd.DataFrame) -> pd.Series:
        if self.selection_mode == "plain":
            return x["p_dir"].ge(x["threshold"]) & x["p_opp"].le(self.opp_cap)
        if self.selection_mode == "bad_day_worker":
            # AND gate: high direction confidence with an almost-dead opposite side.
            return x["p_dir"].ge(self.bdw_raw) & x["p_opp"].le(self.bdw_opp)
        raw_floor, spread_floor = self._mode_floors()
        raw = x["p_dir"].ge(raw_floor)
        spread = x["spread"].ge(float(spread_floor)) if spread_floor is not None else pd.Series(False, index=x.index)
        return raw | spread

    def _candidate_score(self, x: pd.DataFrame) -> pd.Series:
        if self.selection_mode == "plain":
            return (x["p_dir"] - x["threshold"].astype(float)) + x["spread"]
        if self.selection_mode == "bad_day_worker":
            # Reward conviction plus margin past each side of the gate.
            return x["spread"] + (x["p_dir"] - self.bdw_raw) + (self.bdw_opp - x["p_opp"])
        raw_floor, spread_floor = self._mode_floors()
        spread_floor = float(spread_floor or 0.0)
        raw_edge = x["p_dir"] - raw_floor
        spread_edge = x["spread"] - spread_floor
        return np.maximum(raw_edge, spread_edge) + 0.10 * x["p_dir"] - 0.05 * x["p_opp"]

    def decide(self, feat: pd.DataFrame, top_n: int = 3) -> list[HCLiveSignal]:
        self.last_near_misses = self.near_miss_lines(feat, top_n=max(5, int(top_n)))
        if feat.empty:
            return []
        d = self._with_thresholds(feat)
        if d.empty:
            return []

        rows = []
        for side_name, prob_col, opp_col in (
            ("long", "up_prob", "down_prob"),
            ("short", "down_prob", "up_prob"),
        ):
            base_cols = [
                "symbol",
                "anchor_time",
                "base_time",
                "entry_price",
                "horizon_minutes",
                "threshold",
                prob_col,
                opp_col,
            ]
            x = d[base_cols].copy()
            x["side"] = side_name
            x["p_dir"] = x[prob_col].astype(float)
            x["p_opp"] = x[opp_col].astype(float)
            x["spread"] = x["p_dir"] - x["p_opp"]
            m = self._candidate_mask(x)
            if not bool(m.any()):
                continue
            x = x.loc[m].copy()
            x["score"] = self._candidate_score(x)
            rows.append(x)
        if not rows:
            return []

        cand = pd.concat(rows, ignore_index=True)
        # One best signal per symbol. top_n now means distinct coins, not horizon legs.
        cand = cand.sort_values(["symbol", "score"], ascending=[True, False])
        best_side = cand.drop_duplicates("symbol", keep="first")[["symbol", "side"]]
        cand = cand.merge(best_side, on=["symbol", "side"], how="inner")
        cand = cand.drop_duplicates("symbol", keep="first")
        cand = cand.sort_values("score", ascending=False).head(int(top_n))

        use_conv = bool(getattr(self, "conviction", False))
        signals: list[HCLiveSignal] = []
        for r in cand.itertuples(index=False):
            h = int(r.horizon_minutes)
            mult = self.conviction_mult(float(r.spread)) if use_conv else 1.0
            size_usd = None if self.notional_usd is None else self.notional_usd * mult
            signals.append(
                HCLiveSignal(
                    symbol=str(r.symbol),
                    model=f"hc_{self.profile}_{h}m",
                    side=str(r.side),
                    horizon=f"{h}m",
                    move_pct=HC.threshold_pct(h) / 100.0,
                    prob=float(r.p_dir),
                    score=float(r.score),
                    spread=float(r.spread),
                    agree=len(self._models),
                    size_mult=mult,
                    source=(
                        f"mode={self.selection_mode} gate=[{self._floor_desc()}] "
                        f"thr={float(r.threshold):.3f} "
                        f"base={pd.Timestamp(r.base_time).isoformat()}"
                    ),
                    size_usd=size_usd,
                    threshold=float(r.threshold),
                )
            )
        return signals

    def near_miss_lines(self, feat: pd.DataFrame, top_n: int = 5) -> list[str]:
        """Return closest non-orders with explicit threshold shortfalls."""
        if feat.empty:
            return []
        d = self._with_thresholds(feat)
        if d.empty:
            return []

        rows = []
        for side_name, prob_col, opp_col in (
            ("long", "up_prob", "down_prob"),
            ("short", "down_prob", "up_prob"),
        ):
            x = d[["symbol", "horizon_minutes", "threshold", prob_col, opp_col]].copy()
            x["side"] = side_name
            x["p_dir"] = x[prob_col].astype(float)
            x["p_opp"] = x[opp_col].astype(float)
            x["need_prob"] = np.maximum(0.0, x["threshold"].astype(float) - x["p_dir"])
            x["need_opp"] = np.maximum(0.0, x["p_opp"] - self.opp_cap)
            x["gap"] = x["need_prob"] + x["need_opp"]
            x["spread"] = x["p_dir"] - x["p_opp"]
            x = x[x["gap"] > 0.0]
            if not x.empty:
                rows.append(x[["symbol", "horizon_minutes", "threshold", "side", "p_dir", "p_opp", "need_prob", "need_opp", "gap", "spread"]])
        if not rows:
            return []

        miss = pd.concat(rows, ignore_index=True)
        miss = miss.sort_values(["gap", "spread"], ascending=[True, False])
        miss = miss.drop_duplicates(["symbol", "side"], keep="first").head(int(top_n))

        lines: list[str] = []
        for r in miss.itertuples(index=False):
            needs = []
            if float(r.need_prob) > 0:
                needs.append(f"prob+{float(r.need_prob) * 100:.2f}pp")
            if float(r.need_opp) > 0:
                needs.append(f"opp-{float(r.need_opp) * 100:.2f}pp")
            need = ",".join(needs) if needs else "pass"
            lines.append(
                f"{r.symbol} {r.side} {int(r.horizon_minutes)}m "
                f"p={float(r.p_dir):.4f} thr={float(r.threshold):.3f} "
                f"opp={float(r.p_opp):.4f} need={need}"
            )
        return lines
