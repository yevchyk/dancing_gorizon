"""Paper backtest of the directional strategy on the scored holdout.

Each model trades *independently* (decision: "independent positions"): on every
anchor, every directional model that passes its gate opens its own position.
That keeps per-model precision aligned with block-5 win_rate.

PnL is realized from the actual future price path via ExitSimulator (target/stop
within the horizon, else exit at horizon close), not the symmetric placeholder.
Candles are loaded once per symbol and reused across that symbol's anchors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config as C
from ..database import CandleStore
from ..training import ModelRegistry
from .signal_filter import SignalFilter
from .trade_logger import TradeLogger
from .exit_simulator import ExitSimulator
from .thresholds import load_signal_thresholds

# minutes per horizon label, for the exit simulator
HORIZON_MIN = {h.label: h.minutes for h in C.HORIZONS}


class PaperBacktester:
    def __init__(self, registry: ModelRegistry, thresholds: dict[str, float] | None = None,
                 use_stability_veto: bool = True,
                 candle_store: CandleStore | None = None,
                 exit_sim: ExitSimulator | None = None):
        self.registry = registry
        self.thresholds = thresholds if thresholds is not None else load_signal_thresholds()
        self.use_stability_veto = use_stability_veto
        self.store = candle_store or CandleStore(C.CANDLES_DIR)
        self.exit_sim = exit_sim or ExitSimulator()

    def run(self, scored: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
        filt = SignalFilter(self.registry, thresholds=self.thresholds,
                            use_stability_veto=self.use_stability_veto)
        logger = TradeLogger(out_dir)
        logger.event(f"paper backtest (independent positions): {len(scored)} anchors, "
                     f"{len(self.thresholds)} tuned thresholds, "
                     f"stability_veto={self.use_stability_veto}")

        prob_cols = [c for c in scored.columns if c.startswith("prob_")]
        records: list[dict] = []
        n_unresolved = 0

        for symbol, g in scored.groupby("symbol"):
            candles = self.store.load(symbol)
            if candles is None:
                continue
            for _, row in g.iterrows():
                anchor = pd.Timestamp(row["anchor_time"])
                probs = {c[5:]: float(row[c]) for c in prob_cols}
                for sig in filt.evaluate_all(symbol, probs):
                    res = self.exit_sim.resolve(candles, anchor, sig.side,
                                                sig.move_pct, HORIZON_MIN[sig.horizon])
                    if res is None:
                        n_unresolved += 1
                        continue
                    logger.log_trade({"event": "close", "symbol": symbol, "model": sig.model,
                                      "side": sig.side, "horizon": sig.horizon,
                                      "entry_price": "", "exit_price": round(res.exit_price, 6),
                                      "size_usd": C.TRADE_SIZE_USD,
                                      "pnl_pct": res.pnl_pct, "outcome": res.outcome})
                    records.append({"model": sig.model, "side": sig.side,
                                    "horizon": sig.horizon, "prob": sig.prob,
                                    "won": res.won, "outcome": res.outcome,
                                    "pnl_pct": res.pnl_pct})

        trades = pd.DataFrame(records)
        summary = self._summarize(trades)
        summary.to_csv(out_dir / "backtest_summary.csv", index=False)
        logger.event(f"backtest done: {len(trades)} trades, {n_unresolved} unresolved")
        return summary

    def _summarize(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame([{"model": "ALL", "n_trades": 0, "win_rate": 0.0,
                                  "target_rate": 0.0, "avg_pnl_pct": 0.0,
                                  "total_pnl_pct": 0.0}])
        rows = []
        for model, grp in trades.groupby("model"):
            rows.append(self._row(model, grp))
        rows.append(self._row("ALL", trades))
        return pd.DataFrame(rows)

    @staticmethod
    def _row(model: str, grp: pd.DataFrame) -> dict:
        return {
            "model": model,
            "n_trades": len(grp),
            "win_rate": round(grp["won"].mean(), 4),               # target-hit rate
            "target_rate": round((grp["outcome"] == "target").mean(), 4),
            "avg_pnl_pct": round(grp["pnl_pct"].mean(), 4),
            "total_pnl_pct": round(grp["pnl_pct"].sum(), 2),
        }
