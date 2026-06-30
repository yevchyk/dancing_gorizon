"""Describes one of the 15 trainable models."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import HORIZONS, DIRECTIONS, HorizonSpec


@dataclass(frozen=True)
class ModelSpec:
    kind: str            # "direction" or "stability"
    horizon: HorizonSpec
    direction: str | None   # "up"/"down" for direction kind, else None

    @property
    def name(self) -> str:
        if self.kind == "direction":
            return f"{self.direction}_{self.horizon.label}"
        return f"stable_{self.horizon.label}"

    @property
    def target_column(self) -> str:
        return self.name


def all_model_specs() -> list[ModelSpec]:
    """The full set of 15 specs (10 directional + 5 stability)."""
    specs: list[ModelSpec] = []
    for h in HORIZONS:
        for d in DIRECTIONS:
            specs.append(ModelSpec("direction", h, d))
        specs.append(ModelSpec("stability", h, None))
    return specs
