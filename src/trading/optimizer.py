"""Probability-threshold sweep per model -> the optimal cutoff.

For each directional model we resolve the realized outcome (target/stop/timeout
PnL) of every anchor whose probability clears a low floor, then sweep a fine
threshold grid. At each threshold we report n_trades, win_rate (target hit with
the stop active) and avg/total PnL. The "optimal" threshold maximises avg PnL
per trade subject to a minimum trade count (so we don't pick a lucky n=3 point).

Exit resolution is vectorised with numpy per symbol (the per-anchor pandas slice
in ExitSimulator is too slow for ~10 models x ~20k anchors).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..database import CandleStore
from ..training import ModelRegistry
from .timeutil import NS_PER_MIN, index_to_ns, anchors_to_ns

HORIZON_MIN = {h.label: h.minutes for h in C.HORIZONS}
DEFAULT_GRID = tuple(round(x, 2) for x in np.arange(0.40, 0.971, 0.01))


def _resolve_one(ts: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 anchor_ns: int, side: str, move_pct: float, horizon_min: int,
                 stop_ratio: float, fee: float):
    """Return (won, pnl_pct) for one trade, or None if no forward data."""
    ei = np.searchsorted(ts, anchor_ns, side="right") - 1   # last candle <= anchor
    if ei < 0:
        return None
    entry = close[ei]
    if entry <= 0:
        return None
    end_ns = anchor_ns + horizon_min * NS_PER_MIN
    fj = np.searchsorted(ts, end_ns, side="right")          # one past horizon end
    if fj <= ei + 1:
        return None
    h = high[ei + 1:fj]
    l = low[ei + 1:fj]
    if side == "long":
        target, stop = entry * (1 + move_pct), entry * (1 - move_pct * stop_ratio)
        t_hits = np.nonzero(h >= target)[0]
        s_hits = np.nonzero(l <= stop)[0]
    else:
        target, stop = entry * (1 - move_pct), entry * (1 + move_pct * stop_ratio)
        t_hits = np.nonzero(l <= target)[0]
        s_hits = np.nonzero(h >= stop)[0]
    t_i = t_hits[0] if t_hits.size else None
    s_i = s_hits[0] if s_hits.size else None
    if s_i is not None and (t_i is None or s_i <= t_i):       # stop wins ties
        exit_price, won = stop, 0
    elif t_i is not None:
        exit_price, won = target, 1
    else:
        exit_price, won = close[fj - 1], 0                    # timeout at last close
    gross = (exit_price / entry - 1.0) if side == "long" else (1.0 - exit_price / entry)
    return won, gross * 100 - 2 * fee


class ThresholdOptimizer:
    def __init__(self, registry: ModelRegistry, grid=DEFAULT_GRID,
                 floor: float = 0.40, min_trades: int = 30, select_min: float = 0.60,
                 stop_ratio: float = C.STOP_PCT_RATIO, fee: float = C.OKX_FEE_PER_SIDE,
                 candle_store: CandleStore | None = None):
        self.registry = registry
        self.grid = grid
        self.floor = floor
        self.min_trades = min_trades
        self.select_min = select_min   # optimum searched only at thr >= this
        self.stop_ratio = stop_ratio
        self.fee = fee
        self.store = candle_store or CandleStore(C.CANDLES_DIR)

    def _resolved_table(self, scored: pd.DataFrame) -> pd.DataFrame:
        """One row per (model, anchor) with prob, won, pnl_pct — for directional
        models only, anchors above the probability floor."""
        dir_models = [n for n in self.registry.names
                      if self.registry.spec(n).kind == "direction"]
        rows: list[dict] = []
        for symbol, g in scored.groupby("symbol"):
            candles = self.store.load(symbol)
            if candles is None:
                continue
            ts = index_to_ns(candles.index)
            high = candles["high"].to_numpy(float)
            low = candles["low"].to_numpy(float)
            close = candles["close"].to_numpy(float)
            anchors_ns = anchors_to_ns(g["anchor_time"])
            for name in dir_models:
                spec = self.registry.spec(name)
                side = "long" if spec.direction == "up" else "short"
                move = spec.horizon.move_pct
                hmin = HORIZON_MIN[spec.horizon.label]
                probs = g[f"prob_{name}"].to_numpy(float)
                for a_ns, p in zip(anchors_ns, probs):
                    if p < self.floor:
                        continue
                    res = _resolve_one(ts, high, low, close, int(a_ns), side, move,
                                       hmin, self.stop_ratio, self.fee)
                    if res is None:
                        continue
                    rows.append({"model": name, "prob": float(p),
                                 "won": res[0], "pnl_pct": res[1]})
        return pd.DataFrame(rows)

    def run(self, scored: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved = self._resolved_table(scored)
        sweep_rows: list[dict] = []
        optima: list[dict] = []
        for name, g in resolved.groupby("model"):
            prob = g["prob"].to_numpy()
            won = g["won"].to_numpy()
            pnl = g["pnl_pct"].to_numpy()
            per_model: list[dict] = []
            for thr in self.grid:
                m = prob >= thr
                n = int(m.sum())
                if n == 0:
                    continue
                row = {"model": name, "threshold": thr, "n_trades": n,
                       "win_rate": round(float(won[m].mean()), 4),
                       "avg_pnl_pct": round(float(pnl[m].mean()), 4),
                       "total_pnl_pct": round(float(pnl[m].sum()), 2)}
                per_model.append(row)
                sweep_rows.append(row)
            # drift baseline = "take all above floor" (no selectivity = pure drift)
            take_all = pnl.mean()
            # optimum searched only in the selective zone (thr >= select_min) so the
            # degenerate take-all drift point can't masquerade as model skill
            elig = [r for r in per_model
                    if r["n_trades"] >= self.min_trades and r["threshold"] >= self.select_min]
            if elig:
                best = max(elig, key=lambda r: r["avg_pnl_pct"])
                optima.append({**best,
                               "baseline_pnl_pct": round(float(take_all), 4),
                               "edge_vs_drift": round(best["avg_pnl_pct"] - float(take_all), 4),
                               "tradeable": best["avg_pnl_pct"] > 0})
        sweep = pd.DataFrame(sweep_rows)
        opt = pd.DataFrame(optima).sort_values("avg_pnl_pct", ascending=False)
        sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)
        opt.to_csv(out_dir / "optimal_thresholds.csv", index=False)
        return sweep, opt
