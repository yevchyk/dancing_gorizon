"""Position lifecycle + risk management.

Enforces the trading guards: one open position per symbol, a global concurrency
cap, a per-symbol cooldown after a trade, and a daily-drawdown halt. State is
held in memory and can be persisted by the engine via to_dict/from_dict.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .. import config as C


@dataclass
class Position:
    symbol: str
    side: str
    model: str
    entry_price: float
    size_usd: float
    move_pct: float
    horizon: str
    opened_at: str            # ISO timestamp
    engine: str = ""          # which engine/profile opened it
    sz: str = ""              # exchange contract size of this leg (for partial close)
    leg_id: str = ""          # unique key; "symbol" for 1-leg, "symbol#horizon" for multi-leg


@dataclass
class PositionManager:
    max_concurrent: int = C.MAX_CONCURRENT
    cooldown_min: int = C.COOLDOWN_MIN
    daily_stop_pct: float = C.DAILY_STOP_PCT
    max_legs: int = 1                       # kept for compatibility; live caps symbols to one leg
    open_positions: dict[str, Position] = field(default_factory=dict)
    _last_trade_at: dict[str, dt.datetime] = field(default_factory=dict)
    _day_pnl_pct: float = 0.0
    _day: dt.date | None = None

    def _roll_day(self, now: dt.datetime) -> None:
        if self._day != now.date():
            self._day = now.date()
            self._day_pnl_pct = 0.0

    def legs_for(self, symbol: str) -> list[Position]:
        return [p for p in self.open_positions.values() if p.symbol == symbol]

    def can_open(self, symbol: str, now: dt.datetime) -> tuple[bool, str]:
        self._roll_day(now)
        if self._day_pnl_pct <= -abs(self.daily_stop_pct) * 100:
            return False, "daily stop hit"
        if self.legs_for(symbol):
            return False, "symbol already open"
        if len(self.open_positions) >= self.max_concurrent:
            return False, "max concurrent"
        last = self._last_trade_at.get(symbol)
        if last and (now - last).total_seconds() < self.cooldown_min * 60:
            return False, "cooldown"
        return True, "ok"

    def open(self, pos: Position, now: dt.datetime) -> None:
        key = pos.leg_id or pos.symbol
        self.open_positions[key] = pos
        self._last_trade_at[pos.symbol] = now

    def close(self, key: str, pnl_pct: float, now: dt.datetime) -> Position | None:
        pos = self.open_positions.pop(key, None)
        if pos is None:
            return None
        self._roll_day(now)
        self._day_pnl_pct += pnl_pct
        return pos
