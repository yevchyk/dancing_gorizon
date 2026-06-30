"""Turns a row of model probabilities into at most one directional Signal.

Logic (all conditions must hold):
  1. probability gate: prob_<model> >= per-model threshold (from block-5 tuning,
     else DEFAULT_SIGNAL_THRESHOLD).
  2. direction: an `up_*` model -> long, a `down_*` model -> short.
  3. stability veto: skip if the same-horizon stability model fires very high
     (market expected flat -> the touch target is unlikely to be reached).
  4. when several models fire, keep the one with the largest *normalized* margin
     over its own threshold: (p - thr) / (1 - thr). Raw probability can't be
     compared across heterogeneous models (a down_2h maxing at 0.63 would never
     win against an up_5m reaching 0.95), so we rank by headroom instead.

Thresholds are injected (dict name -> float), so the engine can feed the
per-model abs_thresholds produced by the percentile analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config as C
from ..training import ModelRegistry


@dataclass
class Signal:
    symbol: str
    model: str
    side: str          # "long" | "short"
    horizon: str
    move_pct: float
    prob: float
    threshold: float


class SignalFilter:
    def __init__(self, registry: ModelRegistry,
                 thresholds: dict[str, float] | None = None,
                 default_threshold: float = C.DEFAULT_SIGNAL_THRESHOLD,
                 stability_veto: float = C.STABILITY_VETO_THRESHOLD,
                 use_stability_veto: bool = True):
        self.registry = registry
        self.thresholds = thresholds or {}
        self.default_threshold = default_threshold
        self.stability_veto = stability_veto
        self.use_stability_veto = use_stability_veto

    def _threshold(self, name: str) -> float:
        return self.thresholds.get(name, self.default_threshold)

    @staticmethod
    def _margin(p: float, thr: float) -> float:
        """Normalized headroom over the threshold, comparable across models."""
        denom = 1.0 - thr
        return (p - thr) / denom if denom > 1e-9 else 0.0

    def evaluate(self, symbol: str, probs: dict[str, float]) -> Signal | None:
        """probs: {model_name: probability} for one symbol at one moment."""
        best: Signal | None = None
        best_margin = -1.0
        for name in self.registry.names:
            spec = self.registry.spec(name)
            if spec.kind != "direction":
                continue
            p = probs.get(name)
            if p is None:
                continue
            thr = self._threshold(name)
            if p < thr:
                continue
            if self.use_stability_veto:
                stable_p = probs.get(f"stable_{spec.horizon.label}")
                if stable_p is not None and stable_p >= self.stability_veto:
                    continue
            margin = self._margin(float(p), thr)
            if margin > best_margin:
                best_margin = margin
                side = "long" if spec.direction == "up" else "short"
                best = Signal(symbol, name, side, spec.horizon.label,
                              spec.horizon.move_pct, float(p), thr)
        return best

    def evaluate_all(self, symbol: str, probs: dict[str, float]) -> list[Signal]:
        """Every directional model that passes its gate fires independently.
        Used when each model trades its own position (no cross-model competition),
        so per-model precision matches the block-5 analysis."""
        out: list[Signal] = []
        for name in self.registry.names:
            spec = self.registry.spec(name)
            if spec.kind != "direction":
                continue
            p = probs.get(name)
            if p is None or p < self._threshold(name):
                continue
            if self.use_stability_veto:
                stable_p = probs.get(f"stable_{spec.horizon.label}")
                if stable_p is not None and stable_p >= self.stability_veto:
                    continue
            side = "long" if spec.direction == "up" else "short"
            out.append(Signal(symbol, name, side, spec.horizon.label,
                              spec.horizon.move_pct, float(p), self._threshold(name)))
        return out
