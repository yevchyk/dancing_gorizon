"""Win-rate breakdown per model per probability threshold -> one tidy table.

A "signal" = a row where prob_<model> >= threshold. A "win" = the model's actual
target was 1 for that row. win_rate is therefore precision at that threshold.
lift = win_rate / base_rate (how much better than always-firing).
"""

from __future__ import annotations

import pandas as pd

from ..config import PROB_THRESHOLDS
from ..training import ModelRegistry


class ThresholdAnalyzer:
    def __init__(self, registry: ModelRegistry, thresholds=PROB_THRESHOLDS):
        self.registry = registry
        self.thresholds = thresholds

    def analyze(self, scored: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for name in self.registry.names:
            spec = self.registry.spec(name)
            prob = scored[f"prob_{name}"]
            actual = scored[spec.target_column].astype(int)
            base_rate = float(actual.mean())
            for thr in self.thresholds:
                sig = prob >= thr
                n = int(sig.sum())
                wins = int(actual[sig].sum())
                wr = wins / n if n else 0.0
                rows.append({
                    "model": name,
                    "kind": spec.kind,
                    "horizon": spec.horizon.label,
                    "direction": spec.direction or "",
                    "threshold": thr,
                    "n_signals": n,
                    "n_wins": wins,
                    "win_rate": round(wr, 4),
                    "base_rate": round(base_rate, 4),
                    "lift": round(wr / base_rate, 2) if base_rate else 0.0,
                })
        return pd.DataFrame(rows)
