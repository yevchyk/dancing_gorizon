"""Resolve a directional trade's realized PnL from the future price path.

Given the entry anchor and the symbol's candles, walk forward over the horizon:
  long:  target = entry*(1+move),  stop = entry*(1-move*stop_ratio)
  short: target = entry*(1-move),  stop = entry*(1+move*stop_ratio)
First candle to touch target -> win (+move%); first to touch stop -> loss.
If both fall inside the same candle we assume the STOP filled first (conservative,
since intrabar order is unknown). If neither is hit by the horizon end, exit at
the last candle's close (the realized drift). Fees are charged per side.

This replaces the placeholder symmetric-PnL: a touch target is not a captured
return, so PnL must come from an actual exit rule on the path.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import config as C


@dataclass
class ExitResult:
    outcome: str        # "target" | "stop" | "timeout"
    pnl_pct: float      # realized %, fees included
    exit_price: float
    won: int            # 1 if target hit, else 0


class ExitSimulator:
    def __init__(self, stop_ratio: float = C.STOP_PCT_RATIO,
                 fee_per_side: float = C.OKX_FEE_PER_SIDE):
        self.stop_ratio = stop_ratio
        self.fee_per_side = fee_per_side

    def resolve(self, candles: pd.DataFrame, anchor_time: pd.Timestamp,
                side: str, move_pct: float, horizon_min: int) -> ExitResult | None:
        past = candles.loc[candles.index <= anchor_time]
        if past.empty:
            return None
        entry = float(past["close"].iloc[-1])
        if entry <= 0:
            return None
        end = anchor_time + pd.Timedelta(minutes=horizon_min)
        fut = candles.loc[(candles.index > anchor_time) & (candles.index <= end)]
        if fut.empty:
            return None

        if side == "long":
            target = entry * (1 + move_pct)
            stop = entry * (1 - move_pct * self.stop_ratio)
        else:
            target = entry * (1 - move_pct)
            stop = entry * (1 + move_pct * self.stop_ratio)

        for _, c in fut.iterrows():
            hi, lo = float(c["high"]), float(c["low"])
            hit_target = hi >= target if side == "long" else lo <= target
            hit_stop = lo <= stop if side == "long" else hi >= stop
            if hit_stop:   # conservative: stop wins ties within a candle
                return self._result("stop", side, entry, stop, won=0)
            if hit_target:
                return self._result("target", side, entry, target, won=1)

        exit_price = float(fut["close"].iloc[-1])
        return self._result("timeout", side, entry, exit_price, won=0)

    def _result(self, outcome: str, side: str, entry: float,
                exit_price: float, won: int) -> ExitResult:
        gross = (exit_price / entry - 1.0) if side == "long" else (1.0 - exit_price / entry)
        pnl_pct = gross * 100 - 2 * self.fee_per_side
        return ExitResult(outcome, round(pnl_pct, 4), exit_price, won)
