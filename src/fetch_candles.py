"""Refresh the local candle cache from OKX (delta update from last cached bar).

Usage:
  python -m src.fetch_candles                 # update all cached symbols
  python -m src.fetch_candles --max 50        # first 50 only
"""

from __future__ import annotations

import argparse
import time

from . import config as C
from .database import OKXClient, CandleStore, CandleFetcher


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=0, help="0 = all symbols")
    args = p.parse_args()

    store = CandleStore(C.CANDLES_DIR)
    fetcher = CandleFetcher(OKXClient(), store)
    symbols = store.symbols()
    if args.max:
        symbols = symbols[:args.max]

    t0 = time.time()
    ok = fail = 0
    for i, sym in enumerate(symbols, 1):
        try:
            r = fetcher.fetch_symbol(sym, update=True)
            ok += int(r.get("status") == "ok")
        except Exception as e:
            fail += 1
            print(f"  {sym} FAIL: {e}", flush=True)
        if i % 25 == 0 or i == len(symbols):
            print(f"  {i}/{len(symbols)}  ok={ok} fail={fail}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"done: {ok} updated, {fail} failed, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
