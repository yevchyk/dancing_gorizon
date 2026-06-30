"""Per-model best coins: highest win rate among coins with enough signals."""

from __future__ import annotations

import pandas as pd

from ..training import ModelRegistry


class TopCoinsAnalyzer:
    def __init__(self, registry: ModelRegistry, min_signals: int = 5):
        self.registry = registry
        self.min_signals = min_signals

    def analyze(self, scored: pd.DataFrame, threshold: float, top_n: int = 20) -> pd.DataFrame:
        rows: list[dict] = []
        for name in self.registry.names:
            spec = self.registry.spec(name)
            sig = scored[scored[f"prob_{name}"] >= threshold]
            if sig.empty:
                continue
            grp = sig.groupby("symbol")[spec.target_column].agg(["count", "mean"])
            grp = grp[grp["count"] >= self.min_signals]
            for symbol, r in grp.sort_values("mean", ascending=False).head(top_n).iterrows():
                rows.append({
                    "model": name, "threshold": threshold, "symbol": symbol,
                    "n_signals": int(r["count"]), "win_rate": round(float(r["mean"]), 4),
                })
        return pd.DataFrame(rows)
