"""Run several FastComboEngine profiles together as one live engine.

The core engine (first in the list) has priority: if two engines want the same
symbol on the same scan, the earlier one keeps it. Later engines only add the
symbols the core did not take, at their own (smaller) size. Every signal carries
its `engine` profile name so the trade log attributes each fill to its engine.

All sub-engines share the same fast_v2 320-col snapshot (same models/curve), so
the snapshot is built once and each engine scores it independently.
"""

from __future__ import annotations

import pandas as pd

from .fast_combo_engine import STACKS, FastComboEngine


class MultiEngine:
    horizon_minutes = FastComboEngine.horizon_minutes

    def __init__(self, profiles: list[str]) -> None:
        if not profiles:
            raise ValueError("MultiEngine needs at least one profile")
        self.engines = [FastComboEngine(p) for p in profiles]
        self.profile = "stack:" + "+".join(profiles)

    @classmethod
    def from_stack(cls, name: str) -> "MultiEngine":
        if name not in STACKS:
            raise ValueError(f"unknown stack: {name} (have {list(STACKS)})")
        return cls(STACKS[name])

    def build_watchlist(self, store, top_n: int = 0, logger=None) -> list[str]:
        return self.engines[0].build_watchlist(store, top_n, logger)

    def snapshot(self, store, symbols: list[str], now: pd.Timestamp) -> pd.DataFrame:
        # Curve features only — identical across sub-engines, so build once.
        return self.engines[0].snapshot(store, symbols, now)

    def decide(self, feat: pd.DataFrame, top_n: int = 3) -> list:
        claimed: set[str] = set()
        out: list = []
        for eng in self.engines:                 # priority = list order (core first)
            for sig in eng.decide(feat, top_n=top_n):
                if sig.symbol in claimed:
                    continue
                claimed.add(sig.symbol)
                out.append(sig)
        return out

    def describe(self) -> str:
        return "  ||  ".join(f"[{e.profile}] {e.describe()}" for e in self.engines)
