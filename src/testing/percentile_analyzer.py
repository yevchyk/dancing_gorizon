"""Per-model *adaptive* thresholds: instead of fixed absolute probabilities
(0.70..0.85), take the top-X% most confident predictions of each model on its
own scale. This makes weak/poorly-calibrated models comparable to confident ones.

A model whose max prob is only 0.63 produces zero signals under the absolute
grid, yet its top-1% slice still answers the real question: "if I trust only
this model's most confident calls, are they better than random?" (lift > 1).

The derived absolute threshold (abs_threshold) is also the practical per-model
cutoff one would wire into the trading engine.
"""

from __future__ import annotations

import pandas as pd

from ..training import ModelRegistry

# fraction of rows kept = top X%. 0.01 -> top 1% most confident signals.
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.25)


class PercentileThresholdAnalyzer:
    def __init__(self, registry: ModelRegistry, fractions=TOP_FRACTIONS):
        self.registry = registry
        self.fractions = fractions

    def analyze(self, scored: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for name in self.registry.names:
            spec = self.registry.spec(name)
            prob = scored[f"prob_{name}"]
            actual = scored[spec.target_column].astype(int)
            base_rate = float(actual.mean())
            for frac in self.fractions:
                # threshold = the (1-frac) quantile -> keeps the top `frac` share
                abs_thr = float(prob.quantile(1.0 - frac))
                sig = prob >= abs_thr
                n = int(sig.sum())
                wins = int(actual[sig].sum())
                wr = wins / n if n else 0.0
                rows.append({
                    "model": name,
                    "kind": spec.kind,
                    "horizon": spec.horizon.label,
                    "direction": spec.direction or "",
                    "top_pct": round(frac * 100, 1),
                    "abs_threshold": round(abs_thr, 4),
                    "n_signals": n,
                    "n_wins": wins,
                    "win_rate": round(wr, 4),
                    "base_rate": round(base_rate, 4),
                    "lift": round(wr / base_rate, 2) if base_rate else 0.0,
                })
        return pd.DataFrame(rows)
