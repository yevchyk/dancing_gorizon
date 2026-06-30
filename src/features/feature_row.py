"""Single feature sample = one symbol at one anchor moment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureRow:
    symbol: str
    anchor_time: object              # kept for sampling/test mapping, NOT a model input
    curve: dict[str, float] = field(default_factory=dict)   # 600 curve columns

    def to_features(self) -> dict[str, float]:
        """Only model-facing features (curve only — no symbol, no date)."""
        return dict(self.curve)

    def to_record(self) -> dict:
        """Full row for the dataset (curve + metadata for mapping/sampling)."""
        return {"symbol": self.symbol, "anchor_time": self.anchor_time, **self.curve}
