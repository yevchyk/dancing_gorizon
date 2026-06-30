"""Specs for the next-version regression models: 3 path quantities x 5 horizons."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import HORIZONS, HorizonSpec

REG_KINDS = ("ret", "mfe", "mae")   # expected close return / up excursion / down excursion


@dataclass(frozen=True)
class RegModelSpec:
    kind: str            # "ret" | "mfe" | "mae"
    horizon: HorizonSpec

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.horizon.label}"

    @property
    def target_column(self) -> str:
        return self.name


def all_reg_specs() -> list[RegModelSpec]:
    return [RegModelSpec(k, h) for h in HORIZONS for k in REG_KINDS]
