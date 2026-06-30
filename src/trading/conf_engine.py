"""v4 engine: high-confidence CLEAN directional signals.

Take a side only when its prob >= SIGNAL_FLOOR AND the opposite side is quiet
(<= CLEAN_OPP_MAX). Rank candidates by the directional spread (p_dir - p_opp);
take the top few per scan. Exit at the horizon (deadline); OCO is left wide as a
crash safety net only. No trust multiplier (research showed a fixed high bar +
clean filter beats per-model weighting OOS).
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib

from .. import config as C
from ..features import CurveBuilder
from ..training.horizon_slicer import HorizonSlicer

DIRP = C.MODELS_DIR / "dir_prob"
OCO_SAFETY_PCT = 0.03


@dataclass
class ConfSignal:
    symbol: str
    model: str
    side: str           # "long" | "short"
    horizon: str
    move_pct: float     # wide OCO safety
    prob: float
    opp: float
    spread: float
    agree: int          # how many horizons agreed
    size_mult: float    # #3 position-size multiplier from spread


class ConfEngine:
    def __init__(self, floor: float = C.SIGNAL_FLOOR, clean_opp: float = C.CLEAN_OPP_MAX,
                 exclude: tuple[str, ...] = C.CONF_EXCLUDE,
                 min_agree: int = C.CONF_MIN_AGREE, size_by_spread: bool = C.CONF_SIZE_BY_SPREAD):
        self.floor = floor
        self.clean_opp = clean_opp
        self.exclude = set(exclude)
        self.min_agree = min_agree
        self.size_by_spread = size_by_spread
        self.slicer = HorizonSlicer(CurveBuilder(
            C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN))
        self._cols, self._up, self._dn = {}, {}, {}
        for h in C.HORIZONS:
            self._cols[h.label] = self.slicer.columns_for(h)
            self._up[h.label] = joblib.load(DIRP / f"up_{h.label}.joblib")
            self._dn[h.label] = joblib.load(DIRP / f"down_{h.label}.joblib")

    def decide(self, feat, top_n: int = C.CONF_TOP_PER_SCAN) -> list[ConfSignal]:
        # collect every clean candidate, grouped by (symbol, side)
        raw: dict[tuple, list] = {}
        syms = feat["symbol"].to_numpy()
        for h in C.HORIZONS:
            X = feat[self._cols[h.label]]
            p_up = self._up[h.label].predict_proba(X)[:, 1]
            p_dn = self._dn[h.label].predict_proba(X)[:, 1]
            for i in range(len(syms)):
                for kind, p, opp, side in (("up", p_up[i], p_dn[i], "long"),
                                           ("down", p_dn[i], p_up[i], "short")):
                    if f"{kind}_{h.label}" in self.exclude:
                        continue
                    if p >= self.floor and opp <= self.clean_opp:
                        raw.setdefault((syms[i], side), []).append(
                            (h.label, float(p), float(opp)))
        # #4: keep only symbol/side with >= min_agree horizons; take the best spread
        out = []
        for (sym, side), lst in raw.items():
            if len(lst) < self.min_agree:
                continue
            lab, p, opp = max(lst, key=lambda x: x[1] - x[2])
            spread = p - opp
            mult = min(2.0, max(0.5, spread / 0.5)) if self.size_by_spread else 1.0
            name = f"{'up' if side == 'long' else 'down'}_{lab}"
            out.append(ConfSignal(sym, name, side, lab, OCO_SAFETY_PCT,
                                  p, opp, spread, len(lst), mult))
        out.sort(key=lambda c: c.spread, reverse=True)
        return out[:top_n]
