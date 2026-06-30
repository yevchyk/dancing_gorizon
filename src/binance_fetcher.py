"""Binance USDⓈ-M perpetual klines fetcher -> data/binance/candles/ (separate
from the OKX store in data/candles). Writes the SAME parquet schema OKX uses
(timestamp, open, high, low, close, volume) so the existing CandleStore + feature
pipeline can read it drop-in. Higher timeframes (5m/15m/1h/4h) are resampled from
1m by the feature builder, so we only fetch 1m here.

PARALLEL: per-symbol network latency (~1s/call) is the bottleneck, so we run a
thread pool and a GLOBAL rate limiter spaces all calls >= --min-interval apart
(~200 calls/min keeps us safely under Binance's 2400 weight/min). RESUME-able:
re-running continues each symbol from its last stored candle; completed symbols
are skipped.

  python -m src.binance_fetcher                       # 175 liquid, 365d, 1m, 8 workers
  python -m src.binance_fetcher --limit-symbols 1 --days 2   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = "https://fapi.binance.com/fapi/v1/klines"
OUT_DIR = Path("data/binance/candles")
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE
_MIN_MS = 60_000

# global call-rate limiter (spacing, not concurrency): all threads share it
_RL = threading.Lock()
_LAST = [0.0]
_MIN_INTERVAL = 0.3      # seconds between any two API calls (~200/min). Set in main().
_PRINT = threading.Lock()


def _throttle() -> None:
    with _RL:
        wait = _MIN_INTERVAL - (time.time() - _LAST[0])
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.time()


def _get(url: str, retries: int = 6):
    for i in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla"})
            return json.loads(urllib.request.urlopen(req, timeout=30, context=_CTX).read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 418):           # rate-limited / temp ban -> back off hard
                time.sleep(min(120, 5 * 2 ** i))
                continue
            if e.code >= 500:
                time.sleep(2 ** i)
                continue
            raise                               # 400 etc: bad symbol -> bubble up
        except Exception:
            time.sleep(2 ** i)
    raise RuntimeError(f"giving up: {url}")


def norm_symbol(binance_sym: str) -> str:
    """BTCUSDT -> BTC_USDT_SWAP (matches our OKX key naming)."""
    base = binance_sym[:-4] if binance_sym.endswith("USDT") else binance_sym
    return f"{base}_USDT_SWAP"


def fetch_symbol(binance_sym: str, interval: str = "1m", days: int = 365) -> int:
    out = OUT_DIR / f"{norm_symbol(binance_sym)}.parquet"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    prev = None
    if out.exists():
        try:
            prev = pd.read_parquet(out)
            if len(prev):
                pmin = int(pd.Timestamp(prev["timestamp"].min()).value // 1_000_000)
                pmax = int(pd.Timestamp(prev["timestamp"].max()).value // 1_000_000)
                if pmin <= start_ms + 86_400_000:   # file reaches the requested start -> resume forward
                    start_ms = max(start_ms, pmax + _MIN_MS)
                else:                                 # file too shallow for this depth -> refetch full
                    prev = None
        except Exception:
            prev = None
    raw = []
    cur = start_ms
    while cur < end_ms:
        data = _get(f"{BASE}?symbol={binance_sym}&interval={interval}&startTime={cur}&limit=1500")
        if not data:
            break
        raw += data
        cur = data[-1][0] + _MIN_MS
        if len(data) < 1500:
            break
    if not raw:
        return len(prev) if prev is not None else 0
    df = pd.DataFrame(raw, columns=["t", "open", "high", "low", "close", "volume",
                                    "ct", "qv", "n", "tb", "tq", "ig"])
    out_df = pd.DataFrame({
        "timestamp": pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True),
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float), "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
    })
    if prev is not None and len(prev):
        out_df = (pd.concat([prev, out_df], ignore_index=True)
                  .drop_duplicates("timestamp").sort_values("timestamp")
                  .reset_index(drop=True))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ATOMIC write: a kill mid-write must never leave a half-written parquet
    # (that is exactly what corrupted 28 files on 2026-06-13). Write a tmp then
    # os.replace (atomic rename on the same filesystem).
    tmp = out.with_suffix(".parquet.tmp")
    out_df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return len(out_df)


def main() -> None:
    global _MIN_INTERVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=Path("configs/binance_universe_liquid.json"))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-interval", type=float, default=0.3, help="sec between any two API calls")
    ap.add_argument("--limit-symbols", type=int, default=0)
    args = ap.parse_args()
    _MIN_INTERVAL = args.min_interval

    syms = json.loads(args.universe.read_text())
    syms = syms.get("symbols", syms) if isinstance(syms, dict) else syms
    # accept both naming forms: raw Binance (BTCUSDT) and store (BTC_USDT_SWAP)
    syms = [s.replace("_USDT_SWAP", "") + "USDT" if str(s).endswith("_USDT_SWAP") else str(s)
            for s in syms]
    if args.limit_symbols:
        syms = syms[:args.limit_symbols]
    print(f"binance fetch: {len(syms)} symbols x {args.days}d {args.interval}  "
          f"workers={args.workers} spacing={args.min_interval}s (~{60/args.min_interval:.0f} calls/min) -> {OUT_DIR}",
          flush=True)
    t0 = time.time()
    done = [0]

    def work(idx_sym):
        i, s = idx_sym
        try:
            n = fetch_symbol(s, args.interval, args.days)
            tag = f"rows={n}"
        except Exception as e:
            tag = f"FAIL {str(e)[:50]}"
        with _PRINT:
            done[0] += 1
            print(f"  {done[0]}/{len(syms)} {s:16s} {tag}  ({(time.time()-t0)/60:.1f}m)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed(ex.submit(work, (i, s)) for i, s in enumerate(syms, 1)))
    print(f"DONE {len(syms)} symbols in {(time.time()-t0)/60:.1f} min -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
