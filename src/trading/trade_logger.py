"""Structured logging of every engine decision and trade.

Writes two CSVs + a human-readable event log under outputs/trading_logs/<run>/:
  decisions.csv - every signal considered (fired or skipped, with reason)
  trades.csv    - every opened/closed position with realized PnL
  events.log    - plain-text running narration
Everything is append-only and flushed per write so a crash keeps history.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

DECISION_FIELDS = ["ts", "engine", "symbol", "model", "side", "prob", "threshold",
                   "action", "reason"]
TRADE_FIELDS = ["ts", "engine", "event", "symbol", "model", "side", "horizon",
                "entry_price", "exit_price", "size_usd", "pnl_pct", "outcome",
                "opened_at", "closed_at"]


class TradeLogger:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._decisions = self.out_dir / "decisions.csv"
        self._trades = self.out_dir / "trades.csv"
        self._events = self.out_dir / "events.log"
        self._ensure_header(self._decisions, DECISION_FIELDS)
        self._ensure_header(self._trades, TRADE_FIELDS)

    @staticmethod
    def _ensure_header(path: Path, fields: list[str]) -> None:
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fields)

    @staticmethod
    def _ts() -> str:
        return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _append(self, path: Path, fields: list[str], row: dict) -> None:
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(
                {k: row.get(k, "") for k in fields})

    def log_decision(self, *, symbol: str, model: str, side: str, prob: float,
                     threshold: float, action: str, reason: str = "",
                     engine: str = "") -> None:
        self._append(self._decisions, DECISION_FIELDS, {
            "ts": self._ts(), "engine": engine, "symbol": symbol, "model": model,
            "side": side, "prob": round(prob, 4), "threshold": threshold,
            "action": action, "reason": reason,
        })

    def log_trade(self, event: dict) -> None:
        event = {"ts": self._ts(), **event}
        self._append(self._trades, TRADE_FIELDS, event)

    def event(self, msg: str) -> None:
        line = f"{self._ts()}  {msg}"
        print(line)
        with self._events.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
