"""Execution backends. The engine talks to an Executor interface so the same
decision logic runs in paper, shadow, or live mode unchanged.

  PaperExecutor  - simulated fills, no orders, no money (used for backtest/dry-run)
  ShadowExecutor - logs what it *would* do, places nothing
  OKXExecutor    - real OKX orders (filled in during the live phase)

A "directional" trade: enter long on an up-signal / short on a down-signal,
exit when price touches +/-move_pct (the model's own target) or the horizon
elapses. Paper fills resolve the outcome from the realized future candles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Fill:
    symbol: str
    side: str            # "long" | "short"
    entry_price: float
    size_usd: float
    ok: bool = True
    info: str = ""
    sz: str = ""         # exchange contract size of THIS fill (for per-leg partial close)
    filled_at: str = ""  # UTC ISO timestamp after the backend accepted/faked the fill


class Executor:
    """Backend interface. Implementations must be safe to call once per signal.
    move_pct (when given) lets a live backend attach a TP/SL; paper ignores it."""
    mode = "base"

    def enter(self, symbol: str, side: str, price: float, size_usd: float,
              move_pct: float | None = None) -> Fill:
        raise NotImplementedError

    def equity(self) -> float:
        raise NotImplementedError

    def open_positions(self) -> list[dict]:
        """[{symbol, side}] currently open on the backend. Default: none."""
        return []


class PaperExecutor(Executor):
    """Simulated account. Fills at the given price with no slippage/fees by
    default (fee_pct configurable). Tracks a virtual equity balance."""
    mode = "paper"

    def __init__(self, start_equity: float = 1000.0, fee_pct: float = 0.05):
        self._equity = float(start_equity)
        self.fee_pct = fee_pct

    def enter(self, symbol: str, side: str, price: float, size_usd: float,
              move_pct: float | None = None) -> Fill:
        if price <= 0:
            return Fill(symbol, side, price, size_usd, ok=False, info="bad price")
        self._equity -= size_usd * self.fee_pct / 100.0   # entry fee only
        return Fill(symbol, side, price, size_usd, ok=True, info="paper", filled_at=_utc_iso())

    def settle(self, size_usd: float, pnl_pct: float) -> None:
        """Apply a closed trade's realized PnL (and exit fee) to equity."""
        self._equity += size_usd * pnl_pct / 100.0
        self._equity -= size_usd * self.fee_pct / 100.0

    def equity(self) -> float:
        return self._equity


class ShadowExecutor(Executor):
    """Live-data dry run: decides exactly like live but never sends an order."""
    mode = "shadow"

    def __init__(self, equity: float = 1000.0):
        self._equity = float(equity)

    def enter(self, symbol: str, side: str, price: float, size_usd: float,
              move_pct: float | None = None) -> Fill:
        return Fill(symbol, side, price, size_usd, ok=True, info="shadow (no order)", filled_at=_utc_iso())

    def equity(self) -> float:
        return self._equity
