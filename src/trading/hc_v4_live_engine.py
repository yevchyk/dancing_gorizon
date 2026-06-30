"""Live adapter for the v4 (1-minute-horizon) model `min1_2to120`.

Builds the v4 feature schema LIVE (c1m + 5m/15m/1h/4h curves, NO BTC, + time)
at base_time = now - entry_delay, scores the up/down ensemble, gates on p_dir,
and returns HCLiveSignal so the existing LiveTrader handles the book and orders.

Leak-free timing is identical to the dataset: features use only candles <= base,
entry at base+entry_delay, horizon exit. Default horizons = the zone the holdout
calibration showed actually pays (60-120m); short 2-15m stay off (cost wall).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .. import config as C
from ..hc import config as HC
from ..hc import schema_v2 as S2
from ..hc import schema_v3 as S3
from ..hc.data import prepare_btc_frames, prepare_timeframes
from ..hc.data_v3 import _build_feature_matrix_v3, prepare_1m
from .hc_live_engine import HCLiveSignal


class HCV4LiveEngine:
    default_system_name = "Танцюючий Горизонт"
    horizon_exit_only = True

    def __init__(
        self,
        *,
        model_dir: Path = Path("models/min1_2to120"),
        high: float = 0.85,
        horizons: tuple[int, ...] | list[int] = (60, 75, 90, 105, 120),
        entry_delay_min: int = HC.EXEC_ENTRY_DELAY_MIN,
        notional_usd: float | None = None,
        universe_path: Path = Path("configs/hc_universe_full.json"),
        system_name: str = default_system_name,
        profile: str = "min1_2to120",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.high = float(high)
        self.horizons = tuple(sorted(int(h) for h in horizons))
        if not self.horizons:
            raise ValueError("v4 live horizons must not be empty")
        self.entry_delay_min = int(entry_delay_min)
        self.notional_usd = None if notional_usd is None else float(notional_usd)
        self.universe_path = Path(universe_path)
        self.system_name = str(system_name)
        self.profile = str(profile)
        self.horizon_minutes = {f"{h}m": h for h in self.horizons}
        self.last_near_misses: list[str] = []
        fn = self.model_dir / "feature_names.json"
        self.feat_cols = json.loads(fn.read_text(encoding="utf-8"))
        self._folds = self._load_folds()

    def _load_folds(self):
        out = []
        for sub in sorted(self.model_dir.iterdir()):
            up, dn = sub / "up.cbm", sub / "down.cbm"
            if up.exists() and dn.exists():
                u = CatBoostClassifier(); u.load_model(up)
                d = CatBoostClassifier(); d.load_model(dn)
                out.append((u, d))
        if not out:
            raise FileNotFoundError(f"no up/down folds under {self.model_dir}")
        return out

    def describe(self) -> str:
        size = "caller-size" if self.notional_usd is None else f"${self.notional_usd:.2f} notional"
        return (f"{self.system_name} / v4 {self.profile}: p_dir>={self.high:.2f}, "
                f"horizons={','.join(str(h) for h in self.horizons)}, no-BTC+1m+time, "
                f"folds={len(self._folds)}, features=T-{self.entry_delay_min}m, {size}")

    def build_watchlist(self, store, top_n: int = 0, logger=None) -> list[str]:
        data = json.loads(self.universe_path.read_text(encoding="utf-8"))
        universe = data.get("symbols", data) if isinstance(data, dict) else data
        blacklist = set(C.hc_blacklist_symbols())
        vols: list[tuple[str, float]] = []
        missing = 0
        for sym in sorted(str(s) for s in universe):
            if sym in blacklist:
                continue
            c = store.load(sym)
            if c is None or c.empty:
                missing += 1
                continue
            tail = c.iloc[-1440:]
            v = float((tail["close"] * tail["volume"]).sum())
            if np.isfinite(v):
                vols.append((sym, v))
        vols.sort(key=lambda kv: kv[1], reverse=True)
        limit = int(top_n or 0)
        watch = [s for s, _ in (vols[:limit] if limit > 0 else vols)]
        if logger is not None:
            logger.event(f"watchlist(v4): {len(watch)} symbols (universe={len(universe)}, "
                         f"missing={missing}, blacklist={len(blacklist)})")
        return watch

    def snapshot(self, store, symbols: list[str], now: pd.Timestamp) -> pd.DataFrame:
        now = pd.Timestamp(now)
        now = now.tz_convert("UTC") if now.tzinfo else now.tz_localize("UTC")
        base_time = now - pd.Timedelta(minutes=self.entry_delay_min)
        anchors = pd.DatetimeIndex([base_time], tz="UTC")
        try:
            btc = prepare_btc_frames()
        except Exception:
            return pd.DataFrame()
        hsin, hcos, wd = S2.time_features(anchors)
        hsin, hcos, wd = float(hsin[0]), float(hcos[0]), float(wd[0])

        rows: list[dict] = []
        for sym in symbols:
            candles = store.load(sym)
            if candles is None or candles.empty:
                continue
            candles = candles.sort_index()
            past = candles[candles.index <= now]
            if past.empty:
                continue
            entry_time = past.index[-1]
            if entry_time < now - pd.Timedelta(minutes=2):
                continue
            entry_price = float(past["close"].iloc[-1])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            try:
                prepared = prepare_timeframes(candles, btc)
                p1m = prepare_1m(candles)
                if not prepared or p1m is None:
                    continue
                mat, valid = _build_feature_matrix_v3(anchors, prepared, p1m, HC.N_POINTS)
            except Exception:
                continue
            if not bool(valid[0]):
                continue
            curve = mat[0]
            for h in self.horizons:
                row = {"symbol": sym, "anchor_time": now, "base_time": base_time,
                       "entry_price": entry_price, "entry_source_time": entry_time}
                for i, col in enumerate(S3.CURVE_COLUMNS_V3):
                    row[col] = float(curve[i])
                row["horizon_minutes"] = int(h)
                row["horizon_log"] = float(np.log1p(h))
                row["hour_sin"] = hsin
                row["hour_cos"] = hcos
                row["weekday"] = wd
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        feat = pd.DataFrame(rows)
        X = feat[self.feat_cols]
        up = np.mean([u.predict_proba(X)[:, 1] for u, _ in self._folds], axis=0)
        dn = np.mean([d.predict_proba(X)[:, 1] for _, d in self._folds], axis=0)
        out = feat[["symbol", "anchor_time", "base_time", "entry_price",
                    "entry_source_time", "horizon_minutes"]].copy()
        out["up_prob"] = up.astype("float32")
        out["down_prob"] = dn.astype("float32")
        return out

    def decide(self, feat: pd.DataFrame, top_n: int = 8) -> list[HCLiveSignal]:
        self.last_near_misses = []
        if feat is None or feat.empty:
            return []
        rows = []
        for side_name, prob_col, opp_col in (("long", "up_prob", "down_prob"),
                                             ("short", "down_prob", "up_prob")):
            x = feat[["symbol", "base_time", "entry_price", "horizon_minutes", prob_col, opp_col]].copy()
            x["side"] = side_name
            x["p_dir"] = x[prob_col].astype(float)
            x["p_opp"] = x[opp_col].astype(float)
            x["spread"] = x["p_dir"] - x["p_opp"]
            x = x[x["p_dir"] >= self.high]
            if len(x):
                rows.append(x)
        if not rows:
            return []
        cand = pd.concat(rows, ignore_index=True)
        cand = cand.sort_values(["symbol", "p_dir"], ascending=[True, False]).drop_duplicates("symbol", keep="first")
        cand = cand.sort_values("p_dir", ascending=False).head(int(top_n))
        sigs: list[HCLiveSignal] = []
        for r in cand.itertuples(index=False):
            h = int(r.horizon_minutes)
            sigs.append(HCLiveSignal(
                symbol=str(r.symbol), model=f"min1_{h}m", side=str(r.side), horizon=f"{h}m",
                move_pct=HC.threshold_pct(h) / 100.0, prob=float(r.p_dir), score=float(r.p_dir),
                spread=float(r.spread), agree=len(self._folds), size_mult=1.0,
                source=f"v4 p_dir={float(r.p_dir):.4f} opp={float(r.p_opp):.4f}",
                engine="min1_2to120", size_usd=self.notional_usd, threshold=self.high))
        return sigs
