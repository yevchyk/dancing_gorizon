"""Live trust-layer engine (the new decision core).

Loads the dir_prob (up/down) and reg (mfe/mae) models + per-model trust weights
(models/trust_weights.json). For a feature snapshot it builds every model-
direction candidate, keeps only TRUSTED ones whose weight >= global_trust, and
ranks by  prob * reward/risk * trust . Exit is at the horizon (research showed
tight TP/SL is worse); OCO is left wide as a crash safety net only.

Self-healing: rebuild trust_weights.json after a retrain and weak models drop /
recovered models rejoin automatically -- no code change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .. import config as C
from ..features import CurveBuilder
from ..training.horizon_slicer import HorizonSlicer

REG = C.MODELS_DIR / "reg"
DIRP = C.MODELS_DIR / "dir_prob"
OCO_SAFETY_PCT = 0.03   # wide OCO; real exit is the horizon deadline


@dataclass
class TrustSignal:
    symbol: str
    model: str
    side: str          # "long" | "short"
    horizon: str
    move_pct: float    # wide OCO safety width
    prob: float
    rr: float
    score: float


class TrustEngine:
    def __init__(self, floor: float = 0.60, global_trust: float = 0.0):
        self.floor = floor
        self.global_trust = global_trust
        self.slicer = HorizonSlicer(CurveBuilder(
            C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN))
        self.weights = json.loads((C.MODELS_DIR / "trust_weights.json").read_text())["weights"]
        self._cols, self._up, self._dn, self._mfe, self._mae = {}, {}, {}, {}, {}
        for h in C.HORIZONS:
            self._cols[h.label] = self.slicer.columns_for(h)
            self._up[h.label] = joblib.load(DIRP / f"up_{h.label}.joblib")
            self._dn[h.label] = joblib.load(DIRP / f"down_{h.label}.joblib")
            self._mfe[h.label] = joblib.load(REG / f"mfe_{h.label}.joblib")
            self._mae[h.label] = joblib.load(REG / f"mae_{h.label}.joblib")

    def trusted_models(self) -> list[str]:
        return [m for m, w in self.weights.items() if w >= self.global_trust and w > 0]

    def candidates(self, feat: pd.DataFrame) -> pd.DataFrame:
        """feat: rows with the curve columns (p_000..) + 'symbol'. Returns scored
        candidate rows (only trusted, prob>=floor)."""
        rows = []
        for h in C.HORIZONS:
            cols = self._cols[h.label]
            X = feat[cols]
            p_up = self._up[h.label].predict_proba(X)[:, 1]
            p_dn = self._dn[h.label].predict_proba(X)[:, 1]
            mfe = self._mfe[h.label].predict(X)
            mae = self._mae[h.label].predict(X)
            for kind, prob, side, fav, adv in (
                ("up", p_up, "long", mfe, np.abs(mae)),
                ("down", p_dn, "short", -mae, mfe)):
                name = f"{kind}_{h.label}"
                w = self.weights.get(name, 0.0)
                if w < self.global_trust or w <= 0:
                    continue
                rr = np.clip(fav / (np.abs(adv) + 1e-4), 0, 5)
                rows.append(pd.DataFrame({
                    "symbol": feat["symbol"].to_numpy(), "model": name,
                    "side": side, "horizon": h.label, "prob": prob, "rr": rr,
                    "score": prob * rr * w}))
        if not rows:
            return pd.DataFrame()
        c = pd.concat(rows, ignore_index=True)
        return c[c.prob >= self.floor]

    def decide(self, feat: pd.DataFrame, top_n: int) -> list[TrustSignal]:
        c = self.candidates(feat)
        if c.empty:
            return []
        c = c.sort_values("score", ascending=False).head(top_n)
        return [TrustSignal(r.symbol, r.model, r.side, r.horizon, OCO_SAFETY_PCT,
                            float(r.prob), float(r.rr), float(r.score))
                for r in c.itertuples()]
