"""V5 schema = V3 (305) + an 18-col market-regime/volatility block (BINANCE_V5_PLAN §1).

The regime block re-injects MARKET STATE that v3 dropped with the BTC curves —
but as compact scalars, not 60 curve columns. All symbol-blind, %/ratio space,
causal at base_time, NaN-free after a 24h warm-up:

  A. BTC reference ("вприски"): btc_ret_15m/1h/4h/24h, btc_vol_1h/24h,
     btc_range_pos_24h                                                  (7)
  B. Relative strength: rs_1h, rs_24h (symbol log-ret − BTC log-ret)    (2)
  C. Breadth / panic over the FROZEN trade universe: breadth_above_4h,
     breadth_red_1h, panic_cascade, univ_vol_1h                         (4)
  D. Own volatility context: sym_vol_1h, sym_vol_24h, sym_vol_ratio,
     sym_range_pos_24h                                                  (4)
  E. funding_level: current 8h rate / own yearly median |rate|          (1)

323 columns total.
"""

from __future__ import annotations

from .schema_v2 import TAIL_COLUMNS_V2
from .schema_v3 import CURVE_COLUMNS_V3

MARKET_COLUMNS_V5 = [
    "btc_ret_15m", "btc_ret_1h", "btc_ret_4h", "btc_ret_24h",
    "btc_vol_1h", "btc_vol_24h", "btc_range_pos_24h",
    "breadth_above_4h", "breadth_red_1h", "panic_cascade", "univ_vol_1h",
]
SYMBOL_COLUMNS_V5 = [
    "rs_1h", "rs_24h",
    "sym_vol_1h", "sym_vol_24h", "sym_vol_ratio", "sym_range_pos_24h",
    "funding_level",
]
REGIME_COLUMNS_V5 = MARKET_COLUMNS_V5 + SYMBOL_COLUMNS_V5      # 18

FEATURE_COLUMNS_V5 = list(CURVE_COLUMNS_V3) + REGIME_COLUMNS_V5 + list(TAIL_COLUMNS_V2)
EXPECTED_FEATURE_COUNT_V5 = len(FEATURE_COLUMNS_V5)            # 323
