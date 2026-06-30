"""Direct, robust 1m backfill for the bluechip store using fetch_1m_range
(the confirmed-working path), bypassing the flaky CandleFetcher. Threaded, with
per-symbol progress. Idempotent: skips symbols already on disk with enough span.

  python -m src.run_bluechip_fetch --days 140 --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .database import OKXClient
from .fast.candles import fetch_1m_range

OUT = Path("data/bluechip/candles_1m")
MISSING = Path("configs/_bluechip_missing.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=140)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-span-d", type=float, default=120,
                    help="skip symbols already on disk with at least this span")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    declared = [s.strip() for s in MISSING.read_text().split(",") if s.strip()]
    todo = []
    for s in declared:
        p = OUT / f"{s}.parquet"
        if p.exists():
            try:
                ts = pd.to_datetime(pd.read_parquet(p, columns=["timestamp"])["timestamp"], utc=True)
                if (ts.max() - ts.min()).total_seconds() / 86400 >= args.min_span_d:
                    continue
            except Exception:
                pass
        todo.append(s)
    print(f"to fetch: {len(todo)}/{len(declared)} (rest already on disk), days={args.days}, workers={args.workers}", flush=True)

    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days)
    end = pd.Timestamp.now(tz="UTC")

    def work(sym):
        try:
            df = fetch_1m_range(OKXClient(timeout=25.0), sym, start, end, sleep_seconds=0.1)
            if df is None or df.empty:
                return sym, "empty", 0
            df.reset_index().to_parquet(OUT / f"{sym}.parquet", index=False)
            return sym, "ok", len(df)
        except Exception as exc:
            return sym, f"FAIL {type(exc).__name__}", 0

    t0 = time.time()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, s) for s in todo]
        for i, f in enumerate(as_completed(futs), 1):
            sym, status, n = f.result()
            ok += int(status == "ok")
            fail += int(status not in ("ok",))
            print(f"  [{i}/{len(todo)}] {sym:<18} {status:<14} rows={n:<7} "
                  f"ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"DONE ok={ok} fail={fail} in {time.time()-t0:.0f}s. store now has "
          f"{len(list(OUT.glob('*.parquet')))} files.", flush=True)


if __name__ == "__main__":
    main()
