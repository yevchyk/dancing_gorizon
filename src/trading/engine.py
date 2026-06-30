"""Trading engine: takes a snapshot of model probabilities, runs each symbol
through the SignalFilter, applies risk guards, and acts via a pluggable Executor.

A *snapshot* is a DataFrame with a `symbol` column and one `prob_<model>` column
per model (exactly what ModelRegistry.score produces). The same scan_once works
for live (snapshot built from fresh candles) and backtest (a holdout slice).
Outcome resolution differs by mode, so the engine only handles decision + entry;
closing is driven by the caller (live loop waits the horizon; backtest knows it).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .. import config as C
from .signal_filter import SignalFilter, Signal
from .position_manager import PositionManager, Position
from .trade_logger import TradeLogger
from .executor import Executor


class TradingEngine:
    def __init__(self, signal_filter: SignalFilter, positions: PositionManager,
                 logger: TradeLogger, executor: Executor,
                 trade_size_usd: float = C.TRADE_SIZE_USD):
        self.signal_filter = signal_filter
        self.positions = positions
        self.logger = logger
        self.executor = executor
        self.trade_size_usd = trade_size_usd

    @staticmethod
    def _probs_from_row(row: pd.Series) -> dict[str, float]:
        return {c[len("prob_"):]: float(row[c])
                for c in row.index if c.startswith("prob_")}

    def consider(self, symbol: str, probs: dict[str, float],
                 price: float, now: dt.datetime) -> Position | None:
        """Evaluate one symbol; open a position if a signal passes all guards."""
        sig = self.signal_filter.evaluate(symbol, probs)
        if sig is None:
            return None

        ok, reason = self.positions.can_open(symbol, now)
        if not ok:
            self.logger.log_decision(symbol=symbol, model=sig.model, side=sig.side,
                                     prob=sig.prob, threshold=sig.threshold,
                                     action="skip", reason=reason)
            return None

        fill = self.executor.enter(symbol, sig.side, price, self.trade_size_usd)
        if not fill.ok:
            self.logger.log_decision(symbol=symbol, model=sig.model, side=sig.side,
                                     prob=sig.prob, threshold=sig.threshold,
                                     action="skip", reason=fill.info)
            return None

        pos = Position(symbol=symbol, side=sig.side, model=sig.model,
                       entry_price=price, size_usd=self.trade_size_usd,
                       move_pct=sig.move_pct, horizon=sig.horizon,
                       opened_at=now.isoformat())
        self.positions.open(pos, now)
        self.logger.log_decision(symbol=symbol, model=sig.model, side=sig.side,
                                 prob=sig.prob, threshold=sig.threshold,
                                 action="open", reason=fill.info)
        self.logger.log_trade({"event": "open", "symbol": symbol, "model": sig.model,
                               "side": sig.side, "horizon": sig.horizon,
                               "entry_price": price, "size_usd": self.trade_size_usd})
        return pos

    def scan_once(self, snapshot: pd.DataFrame, now: dt.datetime | None = None) -> list[Position]:
        """Run every row of a probability snapshot through the decision pipeline."""
        now = now or dt.datetime.now(dt.timezone.utc)
        opened: list[Position] = []
        for _, row in snapshot.iterrows():
            symbol = row["symbol"]
            price = float(row["entry_price"]) if "entry_price" in row else float("nan")
            pos = self.consider(symbol, self._probs_from_row(row), price, now)
            if pos is not None:
                opened.append(pos)
        return opened
