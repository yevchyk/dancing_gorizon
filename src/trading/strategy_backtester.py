"""Backtest the Strategy against the flat baseline on the same scored window,
so the value added by the regime + agreement gates is visible in numbers.

Both variants trade the same allow-list models with the same per-model
thresholds and the same target/stop exit; the only difference is that the
strategy also requires horizon agreement and a matching trend regime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..database import CandleStore
from ..training import ModelRegistry
from .strategy import Strategy
from .regime import RegimeDetector
from .optimizer import _resolve_one, HORIZON_MIN
from .thresholds import load_signal_thresholds
from .timeutil import index_to_ns, anchors_to_ns


class StrategyBacktester:
    def __init__(self, registry: ModelRegistry, strategy: Strategy,
                 regime: RegimeDetector | None = None,
                 candle_store: CandleStore | None = None,
                 stop_ratio: float = C.STOP_PCT_RATIO, fee: float = C.OKX_FEE_PER_SIDE):
        self.registry = registry
        self.strategy = strategy
        self.regime = regime or RegimeDetector()
        self.store = candle_store or CandleStore(C.CANDLES_DIR)
        self.stop_ratio = stop_ratio
        self.fee = fee

    def run(self, scored: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
        out_dir.mkdir(parents=True, exist_ok=True)
        prob_cols = [c for c in scored.columns if c.startswith("prob_")]
        flat_rows: list[dict] = []
        strat_rows: list[dict] = []

        for symbol, g in scored.groupby("symbol"):
            candles = self.store.load(symbol)
            if candles is None:
                continue
            ts = index_to_ns(candles.index)
            high = candles["high"].to_numpy(float)
            low = candles["low"].to_numpy(float)
            close = candles["close"].to_numpy(float)
            anchors_ns = anchors_to_ns(g["anchor_time"])
            pmat = {c[5:]: g[c].to_numpy(float) for c in prob_cols}

            for k, a_ns in enumerate(anchors_ns):
                a_ns = int(a_ns)
                ei = int(np.searchsorted(ts, a_ns, side="right")) - 1
                if ei < 0:
                    continue
                entry = close[ei]
                probs = {name: arr[k] for name, arr in pmat.items()}
                reg = self.regime.detect(ts, close, a_ns, entry)

                self._collect(self.strategy.flat_entries(symbol, probs),
                              flat_rows, ts, high, low, close, a_ns)
                self._collect(self.strategy.entries(symbol, probs, reg),
                              strat_rows, ts, high, low, close, a_ns)

        flat = self._summarize(pd.DataFrame(flat_rows), "flat")
        strat = self._summarize(pd.DataFrame(strat_rows), "strategy")
        report = pd.concat([flat, strat], ignore_index=True)
        report.to_csv(out_dir / "flat_vs_strategy.csv", index=False)
        return report

    def _collect(self, sigs, sink, ts, high, low, close, a_ns):
        for s in sigs:
            res = _resolve_one(ts, high, low, close, a_ns, s.side, s.move_pct,
                               HORIZON_MIN[s.horizon], self.stop_ratio, self.fee)
            if res is None:
                continue
            sink.append({"model": s.model, "won": res[0], "pnl_pct": res[1]})

    @staticmethod
    def _summarize(trades: pd.DataFrame, variant: str) -> pd.DataFrame:
        rows = []
        if not trades.empty:
            for model, grp in trades.groupby("model"):
                rows.append({"variant": variant, "model": model, "n_trades": len(grp),
                             "win_rate": round(grp["won"].mean(), 4),
                             "avg_pnl_pct": round(grp["pnl_pct"].mean(), 4),
                             "total_pnl_pct": round(grp["pnl_pct"].sum(), 2)})
            rows.append({"variant": variant, "model": "ALL", "n_trades": len(trades),
                         "win_rate": round(trades["won"].mean(), 4),
                         "avg_pnl_pct": round(trades["pnl_pct"].mean(), 4),
                         "total_pnl_pct": round(trades["pnl_pct"].sum(), 2)})
        else:
            rows.append({"variant": variant, "model": "ALL", "n_trades": 0,
                         "win_rate": 0.0, "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0})
        return pd.DataFrame(rows)
