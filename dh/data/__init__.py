"""Data layer: race-safe candle store, feature building, universe.

The live fetcher rewrites candle parquets non-atomically, so a read can hit a
mid-write file. We patch the store's load to retry, so any sim/report is robust.
"""
from __future__ import annotations

import time

from src import config as _C
from src import markets as _mk
from src.hc.data import read_json_symbols as _read_json_symbols
from src.run_hc_offgrid_sim import build_feature_rows as _build_feature_rows

# --- race-safe load (retry on transient truncated read while fetcher writes) ---
_orig_load = _mk.Store.load
def _safe_load(self, symbol):
    for _ in range(6):
        try:
            return _orig_load(self, symbol)
        except Exception:
            time.sleep(0.3)
    try:
        return _orig_load(self, symbol)
    except Exception:
        return None
_mk.Store.load = _safe_load


def universe(drop_blacklist: bool = True) -> list[str]:
    """Tradable HC symbols (universe minus toxic/blacklist)."""
    syms = [str(s) for s in _read_json_symbols()]
    if drop_blacklist:
        bl = set(getattr(_C, "HC_BLACKLIST_SYMBOLS", ())) | set(_C.BLACKLIST_SYMBOLS)
        syms = [s for s in syms if s not in bl]
    return syms


def build_features(symbols, entries, horizons, entry_delay_min):
    """One feature row per (symbol, entry, horizon): 4 timeframes + BTC(1h/4h) + horizon."""
    return _build_feature_rows(symbols=symbols, entries=entries,
                               horizons=tuple(int(h) for h in horizons),
                               entry_delay_min=int(entry_delay_min))
