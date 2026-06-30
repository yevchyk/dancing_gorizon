"""Always-on parallel candle-fetcher daemon (the decoupled data layer).

Keeps the production CandleStore fresh for the trained universe by pulling the
recent 1m candles for every symbol in parallel, on a tight loop. Trading engines
then run read-only (`run_live ... --no-fetch`) and just READ this store.

Why this exists:
  * ONE writer => no candle-file write races => the corrupt-parquet crash that
    killed the live loop cannot happen, and multiple read-only engines are safe.
  * Fetch is I/O-bound, so a thread pool cuts ~135s serial -> ~10-15s, keeping the
    store within a few seconds of real time. Scan cycles stop skipping anchors.

NOTE: do NOT run this while a *fetching* engine is up (two writers race). Run the
engines with --no-fetch once this daemon is live.

Run: python -m src.run_fetcher [--interval-sec 25] [--workers 10] [--lookback-min 20]
     python -m src.run_fetcher --once --workers 10 --lookback-min 240
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import config as C
from .database import CandleStore, OKXClient
from .database.candle_fetcher import CandleFetcher
from .fast import config as FC
from .hc.data import read_json_symbols


def universe(kind: str) -> list[str]:
    if kind == "hc":
        return sorted(set(read_json_symbols()) - C.hc_blacklist_symbols())
    if kind == "store":
        return []
    trained = {p.stem for p in FC.FAST_CHUNKS_DIR.glob("*.parquet")}
    return sorted(trained - set(C.BLACKLIST_SYMBOLS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-sec", type=float, default=25.0,
                    help="target seconds between refresh cycles")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--lookback-min", type=int, default=20)
    ap.add_argument("--once", action="store_true",
                    help="run one refresh cycle and exit")
    ap.add_argument("--universe", choices=["fast", "hc", "store"], default="fast")
    args = ap.parse_args()

    store = CandleStore(C.CANDLES_DIR)
    syms = store.symbols() if args.universe == "store" else universe(args.universe)

    # one CandleFetcher (own OKXClient) per worker thread -> no shared HTTP session.
    local = threading.local()

    def fetcher() -> CandleFetcher:
        f = getattr(local, "f", None)
        if f is None:
            f = local.f = CandleFetcher(OKXClient(timeout=25.0), store)
        return f

    def work(sym: str) -> str:
        try:
            return fetcher().update_recent(sym, args.lookback_min).get("status", "?")
        except Exception as exc:                       # never let one symbol kill the loop
            return f"FAIL {type(exc).__name__}"

    print(f"fetcher daemon: {len(syms)} symbols  workers={args.workers}  "
          f"interval={args.interval_sec}s  lookback={args.lookback_min}m", flush=True)
    cycle = 0
    while True:
        t0 = time.time()
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(work, s) for s in syms]
            for f in as_completed(futs):
                if f.result() == "ok":
                    ok += 1
                else:
                    fail += 1
        dur = time.time() - t0
        cycle += 1
        # freshness check: newest candle lag for a probe symbol
        probe = store.load(syms[0]) if syms else None
        lag = ((pd.Timestamp.now(tz="UTC") - probe.index.max()).total_seconds()
               if probe is not None and not probe.empty else float("nan"))
        print(f"[{pd.Timestamp.now(tz='UTC'):%H:%M:%S}] cycle {cycle}: {ok} ok {fail} fail "
              f"in {dur:.1f}s  probe_lag={lag:.0f}s", flush=True)
        if args.once:
            break
        time.sleep(max(0.0, args.interval_sec - dur))


if __name__ == "__main__":
    main()
