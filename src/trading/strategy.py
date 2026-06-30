"""The strategy layer: turns raw model signals into final trade decisions by
stacking the gates the benchmarks justified.

  1. per-model probability threshold  (SignalFilter)
  2. stability veto                    (SignalFilter)
  3. horizon agreement: >= AGREEMENT_MIN same-direction models must fire
  4. regime filter: longs only in an up regime, shorts only in a down regime
  5. tradeable allow-list: only emit the models cleared in analysis

Agreement is counted over ALL directional models of a side (so a lone tradeable
model like down_5m can still fire when down_15m etc. confirm), but only models
on the allow-list are actually emitted. Positions are independent (one per
emitted model), matching the earlier decision.
"""

from __future__ import annotations

from .. import config as C
from ..training import ModelRegistry
from .signal_filter import SignalFilter, Signal


class Strategy:
    def __init__(self, registry: ModelRegistry, thresholds: dict[str, float] | None = None,
                 agreement_min: int = C.AGREEMENT_MIN, use_regime: bool = True,
                 use_stability_veto: bool = True,
                 tradeable: tuple[str, ...] | None = C.TRADEABLE_MODELS):
        self.registry = registry
        self.filter = SignalFilter(registry, thresholds=thresholds,
                                   use_stability_veto=use_stability_veto)
        self.agreement_min = agreement_min
        self.use_regime = use_regime
        self.tradeable = set(tradeable) if tradeable else None

    def entries(self, symbol: str, probs: dict[str, float], regime: str) -> list[Signal]:
        fired = self.filter.evaluate_all(symbol, probs)   # passes threshold + veto
        longs = [s for s in fired if s.side == "long"]
        shorts = [s for s in fired if s.side == "short"]

        out: list[Signal] = []
        for side_sigs, side, want_regime in (
            (longs, "long", "up"), (shorts, "short", "down")):
            if len(side_sigs) < self.agreement_min:          # horizon agreement
                continue
            if self.use_regime and regime != want_regime:    # regime gate
                continue
            for s in side_sigs:
                if self.tradeable is None or s.model in self.tradeable:
                    out.append(s)
        return out

    def flat_entries(self, symbol: str, probs: dict[str, float]) -> list[Signal]:
        """Baseline: just per-model threshold (+ veto), allow-list only. No regime,
        no agreement. Used to measure what the strategy gates add."""
        fired = self.filter.evaluate_all(symbol, probs)
        return [s for s in fired
                if self.tradeable is None or s.model in self.tradeable]
