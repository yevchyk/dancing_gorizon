"""Pick ~25 'survivorship-fix' Binance USDT-M perps: listed >=13 months ago but
CURRENTLY below our $10M/day liquid cutoff — i.e. the year's faded/loser names.
Written to configs/binance_universe_train_extra.json. TRAIN-ONLY (never trade):
their job is to de-bias the training year, which otherwise contains only today's
winners (survivorship). True corpses (delisted) are not fetchable via the public
klines endpoint, so this is a partial but honest fix.

  python -m src.binance_pick_extra
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .binance_fetcher import _get

EXINFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
T24 = "https://fapi.binance.com/fapi/v1/ticker/24hr"
LIQUID = Path("configs/binance_universe_liquid.json")
OUT = Path("configs/binance_universe_train_extra.json")
N = 25
VOL_LO, VOL_HI = 1e6, 10e6     # USDT/day: tradeable-but-faded band
MIN_AGE_DAYS = 395             # full training year of history + margin


def main() -> None:
    liquid = json.loads(LIQUID.read_text())
    liquid = set(liquid.get("symbols", liquid) if isinstance(liquid, dict) else liquid)
    info = _get(EXINFO)
    cutoff_ms = (time.time() - MIN_AGE_DAYS * 86400) * 1000
    old = {s["symbol"]: int(s.get("onboardDate", 0)) for s in info["symbols"]
           if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"
           and s.get("quoteAsset") == "USDT" and s.get("onboardDate", 9e18) <= cutoff_ms}
    vols = {t["symbol"]: float(t["quoteVolume"]) for t in _get(T24)}
    cand = sorted(((s, vols.get(s, 0.0)) for s in old
                   if s not in liquid and VOL_LO <= vols.get(s, 0.0) < VOL_HI),
                  key=lambda kv: -kv[1])
    pick = [s for s, _ in cand[:N]]
    print(f"band ${VOL_LO/1e6:.0f}-{VOL_HI/1e6:.0f}M/d, age>={MIN_AGE_DAYS}d, not in liquid175: "
          f"{len(cand)} candidates -> picking {len(pick)}")
    for s, v in cand[:N]:
        onb = time.strftime("%Y-%m", time.gmtime(old[s] / 1000))
        print(f"  {s:16s} ${v/1e6:5.1f}M/d  listed {onb}")
    OUT.write_text(json.dumps({
        "purpose": "survivorship-fix for the 365d train set - TRAIN-ONLY, never trade",
        "criteria": f"USDT perp, TRADING, onboard>={MIN_AGE_DAYS}d ago, "
                    f"24h vol ${VOL_LO/1e6:.0f}-{VOL_HI/1e6:.0f}M, excluded from liquid175",
        "symbols": pick}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
