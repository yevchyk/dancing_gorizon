"""Per-instrument round-trip cost (fee + slippage) — Fix 2 of the sim plan.

The old sims used a FLAT 0.75% for every symbol. That over-charged liquid majors
and badly UNDER-charged thin/jumpy names: on 2026-06-09, H_USDT_SWAP cost ~1.5%
of slippage on the exit alone (logged exit 0.14397 vs real fill 0.14610), turning
a logged -13.5% into a real -24% at 4x. MRVL (tokenized equity) behaved similarly.

We model cost from a UNIT-FREE liquidity proxy: the median 1-minute bar range
(high-low)/close. It needs no volume units (which are contaminated by per-contract
multipliers) and directly measures the jumpiness that produces slippage. It is
calibrated against the two real fills we observed:

    name   barrange%(240m)   observed 1-side slip   model 1-side slip
    BTC          0.13               ~tight                 0.10  (floored)
    MRVL         0.53               ~1.1%                  ~0.99 (equity x2.5)
    H            1.60               ~1.5%                  1.20

  one-way slip% = clamp(0.75 * barrange%, 0.05, 2.0);  equity: *2.5 (cap 2.5)
  round-trip cost% = max(0.45, fee_roundtrip + 2*slip_oneway)

The 0.45% floor is the liquid-only figure from CLAUDE.md §1 (fee 0.15 + slip 0.30);
the fee component uses the configured OKX taker fee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..markets import is_equity

# round-trip taker fee, in PERCENT (OKX_FEE_PER_SIDE is % per side)
FEE_ROUNDTRIP_PCT = 2.0 * float(C.OKX_FEE_PER_SIDE)

# calibration constants (see module docstring)
SLIP_K = 0.75            # slip per unit of bar-range
SLIP_MIN_PCT = 0.05      # liquid floor (one-way)
SLIP_MAX_PCT = 2.0       # crypto cap (one-way)
EQUITY_MULT = 2.5        # tokenized equities: off-hours gaps + tracking error
EQUITY_MAX_PCT = 2.5     # equity one-way cap
COST_FLOOR_PCT = 0.45    # round-trip floor (CLAUDE.md liquid-only)

# fallback bar-range when candles are unavailable (≈ universe p85, conservative)
DEFAULT_BARRANGE_PCT = 0.30


def barrange_pct(candles: pd.DataFrame | None, t: pd.Timestamp | None = None,
                 lookback_min: int = 240) -> float | None:
    """Median 1-min (high-low)/close over the lookback ending at t, in percent."""
    if candles is None or candles.empty:
        return None
    idx = candles.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df = candles
    if t is not None:
        t = pd.Timestamp(t)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        df = candles[idx <= t]
    if df.empty:
        return None
    w = df.iloc[-lookback_min:]
    r = ((w["high"] - w["low"]) / w["close"]).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return None
    return float(r.median()) * 100.0


def slip_oneway_pct(barrange: float, equity: bool) -> float:
    s = min(max(SLIP_K * float(barrange), SLIP_MIN_PCT), SLIP_MAX_PCT)
    if equity:
        s = min(s * EQUITY_MULT, EQUITY_MAX_PCT)
    return s


def roundtrip_cost_pct(barrange: float, equity: bool) -> float:
    """Round-trip cost % from a bar-range% and asset class."""
    return max(COST_FLOOR_PCT, FEE_ROUNDTRIP_PCT + 2.0 * slip_oneway_pct(barrange, equity))


def cost_pct(symbol: str, *, candles: pd.DataFrame | None = None,
             t: pd.Timestamp | None = None, barrange: float | None = None) -> float:
    """Round-trip cost % for one symbol.

    Provide `barrange` (precomputed %) OR `candles` (+optional `t`); otherwise a
    conservative default is used. Equities are detected via markets.is_equity.
    """
    if barrange is None:
        barrange = barrange_pct(candles, t)
    if barrange is None:
        barrange = DEFAULT_BARRANGE_PCT
    return roundtrip_cost_pct(barrange, is_equity(symbol))


def cost_fn_from_store(store=None, lookback_min: int = 240):
    """A memoized symbol -> round-trip cost% function backed by the candle store.

    For sims/exports that score many legs: one bar-range read per symbol. Pass the
    result as `cost_fn` to run_hc_dense_eval.add_outcomes, or call directly.
    """
    if store is None:
        from ..database import CandleStore  # lazy: avoid import cycle at module load
        store = CandleStore(C.CANDLES_DIR)
    cache: dict[str, float] = {}

    def fn(symbol: str) -> float:
        symbol = str(symbol)
        if symbol not in cache:
            br = barrange_pct(store.load(symbol), lookback_min=lookback_min)
            cache[symbol] = roundtrip_cost_pct(
                br if br is not None else DEFAULT_BARRANGE_PCT, is_equity(symbol))
        return cache[symbol]

    return fn
