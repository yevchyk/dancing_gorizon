"""Unicorn (pulse00) simulation over the last 12 hours, broken down by hour.

Re-inference: rebuild the 320-col curve + score the 8 fast_v2 models at every
2-minute anchor in the last 12h (production candles, current blacklist), form the
Unicorn signals (>=3 worthy models agree, none oppose, 10m hold), and run the
real candle-replay simulator. Prices come from the PRODUCTION candle store (which
the live loop keeps fresh) -- the fast 1m cache is stale for recent windows.

Run: python -m src.run_unicorn_12h [hours]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .fast import config as FC
from .trading.fast_combo_engine import FastComboEngine
from .trading.timeutil import index_to_ns
from .run_unicorn_cadence_sim import score_1min, unicorn_signals, watchlist
from .run_engine_compare_report import to_engine_input
from .run_test_engine_harvest_sim import PriceSeries, simulate_engine

TOP_PER_SCAN, MAX_OPEN, COOLDOWN_MIN = 3, 8, 10


class ProdBook:
    """Price book backed by the production candle store (fresh for recent windows)."""

    def __init__(self, store: CandleStore) -> None:
        self.store = store
        self._cache: dict[str, PriceSeries | None] = {}

    def at(self, symbol: str, t: pd.Timestamp) -> float | None:
        if symbol not in self._cache:
            c = self.store.load(symbol)
            if c is None or c.empty:
                self._cache[symbol] = None
            else:
                c = c.sort_index()
                self._cache[symbol] = PriceSeries(index_to_ns(c.index),
                                                  c["close"].to_numpy("float64"))
        ps = self._cache[symbol]
        return None if ps is None else ps.at(t)


def main() -> None:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    eng = FastComboEngine("pulse00")
    store = CandleStore(C.CANDLES_DIR)
    syms = watchlist()
    now = pd.Timestamp.now(tz="UTC").floor("1min")
    start = now - pd.Timedelta(hours=hours)
    end = now - pd.Timedelta(minutes=11)
    anchors = pd.date_range(start.ceil("2min"), end.floor("2min"), freq="2min")
    print(f"window {start:%m-%d %H:%M} -> {end:%m-%d %H:%M} UTC  "
          f"anchors={len(anchors)}  symbols={len(syms)}")

    scored = score_1min(eng, store, syms, anchors)
    cand = unicorn_signals(scored)
    if cand.empty:
        print("no Unicorn signals fired in window"); return
    st = [pd.Timestamp(t) for t in anchors]
    trades, _ = simulate_engine("unicorn_12h", to_engine_input(cand, "unicorn_12h"), st,
                                ProdBook(store), harvest=False, top_per_scan=TOP_PER_SCAN,
                                max_open=MAX_OPEN, cooldown_min=COOLDOWN_MIN)
    if trades.empty:
        print("no Unicorn trades after throttle"); return
    trades = trades.assign(hour=pd.to_datetime(trades["opened_at"], utc=True).dt.hour)
    p = trades["net_pnl_pct"]
    print(f"\nOVERALL: {len(trades)} trades  win={trades['won'].mean():.3f}  "
          f"avg={p.mean():+.4f}%  total={p.sum():+.2f}%  "
          f"long={(trades['side']=='long').mean():.0%}  symbols={trades['symbol'].nunique()}")

    rows = []
    for h in range(24):
        g = trades[trades["hour"] == h]
        if len(g) == 0:
            continue
        rows.append({"utc": h, "kyiv": (h + 3) % 24, "n": int(len(g)),
                     "long%": float((g["side"] == "long").mean()),
                     "win": float(g["won"].mean()),
                     "avg%": float(g["net_pnl_pct"].mean()),
                     "total%": float(g["net_pnl_pct"].sum())})
    hourly = pd.DataFrame(rows)
    print("\nBY HOUR (UTC / Kyiv=UTC+3):")
    print(hourly.to_string(index=False, formatters={
        "long%": "{:.0%}".format, "win": "{:.3f}".format,
        "avg%": "{:+.4f}".format, "total%": "{:+.2f}".format}))
    print(f"\nrunning total% (close order): {p.values.cumsum().round(2).tolist()}")


if __name__ == "__main__":
    main()
