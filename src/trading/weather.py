"""Weather Station — single source of truth for per-market causal market-state.

One module both the live engines and the offline analysis consume, so the gate
that was VALIDATED (causal herd gate + clean-wick filter, src/run_ladder_validate.py)
is computed in exactly ONE place. Per-market: pass any store + symbol universe +
a liquid "lead" symbol (BTC for crypto, QQQ/SPX for tradfi).

Cross-market lead-lag was REJECTED (src/run_xmkt_leadlag.py) -> the station is
PER-MARKET; there is no crypto->tradfi bridge.

Raw state per anchor (all causal, lookback only, vectorized):
  breadth        frac of universe up over 60m            (direction up/down)
  togetherness   max(breadth, 1-breadth)                 (herd intensity)
  breadth_slope  breadth(now) - breadth(30m ago)         (turning?)
  dispersion     cross-sectional std of 60m returns (%)  (idio vs herd)
  mkt_ret        median 60m return of universe (%)       (level)
  wick_frac      universe mean lower-wick dominance       (intrabar bounce)
  lead_r15       lead-symbol 15m return (%)               (knifing now?)
  lead_vol       lead-symbol 30m realized vol             (instability)
  lead_dd        lead-symbol drawdown from 24h high (%)   (depth)
  lead_accel     lead 15m ret - prior 15m ret (%)         (decelerating?)

add_causal(df) adds trailing-only thresholds + gate/clean flags + a descriptive
STAGE label (telemetry; the TRADING gate stays herd+wick, which we validated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..database import CandleStore
from .timeutil import index_to_ns

LEAD = {"crypto": "BTC_USDT_SWAP", "tradfi": "QQQ_USDT_SWAP"}
RAW_COLS = ["breadth", "togetherness", "breadth_slope", "dispersion", "mkt_ret",
            "wick_frac", "lead_r15", "lead_vol", "lead_dd", "lead_accel"]


@dataclass
class WeatherStation:
    store: CandleStore
    symbols: list[str]
    lead_symbol: str = "BTC_USDT_SWAP"
    min_bars: int = 1440  # need 24h history for drawdown

    def compute(self, anchor_ns: np.ndarray) -> pd.DataFrame:
        """Vectorized raw market-state at the given anchor timestamps (ns, UTC)."""
        n = len(anchor_ns)
        nsym = len(self.symbols)
        r60n = np.full((n, nsym), np.nan)
        r60p = np.full((n, nsym), np.nan)
        wick = np.full((n, nsym), np.nan)
        lead_r15 = np.full(n, np.nan); lead_vol = np.full(n, np.nan)
        lead_dd = np.full(n, np.nan); lead_accel = np.full(n, np.nan)
        for j, sym in enumerate(self.symbols):
            c = self.store.load(sym)
            if c is None or c.empty:
                continue
            c = c.sort_index(); ts = index_to_ns(c.index)
            close = c["close"].to_numpy("float64"); high = c["high"].to_numpy("float64")
            low = c["low"].to_numpy("float64"); op = c["open"].to_numpy("float64")
            ei = np.searchsorted(ts, anchor_ns, side="right") - 1
            ok = ei >= 90
            idx = ei[ok]
            r60n[ok, j] = close[idx] / close[idx - 60] - 1
            r60p[ok, j] = close[idx - 30] / close[idx - 90] - 1
            rng = high[idx] - low[idx]
            lw = np.minimum(op[idx], close[idx]) - low[idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                wick[ok, j] = np.where(rng > 0, lw / rng, np.nan)
            if sym == self.lead_symbol:
                s = pd.Series(close)
                lr = np.log(s).diff()
                vol = lr.rolling(30).std().to_numpy()
                r15s = (s / s.shift(15) - 1).to_numpy()
                r15p = (s.shift(15) / s.shift(30) - 1).to_numpy()
                dd = (s / s.rolling(1440, min_periods=60).max() - 1).to_numpy()
                okl = ei >= self.min_bars
                el = ei[okl]
                lead_vol[okl] = vol[el]
                lead_r15[okl] = r15s[el] * 100
                lead_accel[okl] = (r15s[el] - r15p[el]) * 100
                lead_dd[okl] = dd[el] * 100
        nv = (~np.isnan(r60n)).sum(1).clip(min=1)
        breadth = np.nansum((r60n > 0), 1) / nv
        nvp = (~np.isnan(r60p)).sum(1).clip(min=1)
        breadth_p = np.nansum((r60p > 0), 1) / nvp
        out = pd.DataFrame({
            "anchor_ns": anchor_ns,
            "breadth": breadth,
            "togetherness": np.maximum(breadth, 1 - breadth),
            "breadth_slope": breadth - breadth_p,
            "dispersion": np.nanstd(r60n, axis=1) * 100,
            "mkt_ret": np.nanmedian(r60n, axis=1) * 100,
            "wick_frac": np.nanmean(wick, axis=1),
            "lead_r15": lead_r15, "lead_vol": lead_vol,
            "lead_dd": lead_dd, "lead_accel": lead_accel,
        })
        return out

    @staticmethod
    def add_causal(df: pd.DataFrame, gate_pct: float = 80.0,
                   day_col: str = "day") -> pd.DataFrame:
        """Add trailing-only (causal) thresholds, gate/clean flags, and STAGE.
        Requires a sorted `day` column. Thresholds at day d use anchors in days < d."""
        df = df.sort_values(day_col).reset_index(drop=True)
        days = list(dict.fromkeys(df[day_col]))
        tog_thr = {}; wick_thr = {}; vol_hi = {}
        for d in days:
            past = df[df[day_col] < d]
            if len(past) < 50:
                tog_thr[d] = np.inf; wick_thr[d] = -np.inf; vol_hi[d] = np.inf
            else:
                tog_thr[d] = np.nanpercentile(past["togetherness"], gate_pct)
                wick_thr[d] = np.nanmedian(past["wick_frac"])
                vol_hi[d] = np.nanpercentile(past["lead_vol"], 80)
        df["tog_thr"] = df[day_col].map(tog_thr)
        df["wick_thr"] = df[day_col].map(wick_thr)
        df["vol_hi"] = df[day_col].map(vol_hi)
        df["gate"] = df["togetherness"] >= df["tog_thr"]
        df["clean"] = df["gate"] & (df["wick_frac"] <= df["wick_thr"])
        df["knife"] = df["clean"] & (df["lead_r15"] <= -0.3)
        df["stage"] = _stage(df)
        return df


def _stage(df: pd.DataFrame) -> pd.Series:
    """Descriptive regime label (telemetry, the user's hourly view)."""
    s = pd.Series("calm", index=df.index, dtype=object)
    herd = df["gate"].to_numpy()
    brslope = df["breadth_slope"].to_numpy()
    r15 = df["lead_r15"].to_numpy()
    dd = df["lead_dd"].to_numpy()
    breadth = df["breadth"].to_numpy()
    volhi = (df["lead_vol"] >= df["vol_hi"]).to_numpy()
    s[~herd & (df["mkt_ret"].to_numpy() < 0)] = "chop_down"
    s[~herd & (df["mkt_ret"].to_numpy() >= 0)] = "chop_up"
    s[herd & (breadth >= 0.5)] = "euphoria"            # herd to the upside
    s[herd & (breadth < 0.5) & (r15 <= -0.3)] = "dump"  # still knifing
    s[herd & (breadth < 0.5) & (r15 > -0.3) & (brslope > 0)] = "recovery"
    s[herd & (breadth < 0.5) & (dd <= -3.0) & volhi] = "capitulation"
    return s
