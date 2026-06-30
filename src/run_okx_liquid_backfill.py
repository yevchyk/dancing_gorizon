"""Backfill an isolated OKX liquid-symbol candle store.

This uses the same OKXClient + CandleFetcher + CandleStore pipeline as the
crypto cache, but writes to data/okx_liquid/candles_mixed by default so the
production crypto/live store is not touched.

Example:
  python -m src.run_okx_liquid_backfill --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import config as C
from .database import CandleFetcher, CandleStore, OKXClient
from .database.candle_fetcher import RESOLUTION_DAYS


DEFAULT_SYMBOLS_FILE = C.CONFIGS_DIR / "okx_liquid_symbols_100.json"
DEFAULT_OUT_DIR = C.DATA_DIR / "okx_liquid" / "candles_mixed"


class SingleInstanceLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            try:
                owner = self.path.read_text(encoding="utf-8").strip()
            except Exception:
                owner = "unknown"
            raise SystemExit(
                f"OKX liquid backfill lock exists: {self.path} (owner: {owner}). "
                "Stop the other backfill or remove a stale lock."
            ) from exc
        os.write(self.fd, f"pid={os.getpid()} started={pd.Timestamp.now(tz='UTC')}\n".encode())
        return self

    def __exit__(self, *_exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def load_symbols(path: Path, explicit: str | None) -> list[str]:
    if explicit:
        raw = [s.strip() for s in explicit.split(",")]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("symbols", data) if isinstance(data, dict) else data

    symbols: list[str] = []
    seen: set[str] = set()
    for value in raw:
        symbol = str(value).strip().upper().replace("-", "_")
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def fmt_ts(ts) -> str:
    if ts is None or pd.isna(ts):
        return "-"
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M UTC")


def coverage_status(df: pd.DataFrame | None) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    if df is None or df.empty:
        return 0, None, None
    return int(len(df)), df.index.min(), df.index.max()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--page-sleep", type=float, default=0.08)
    ap.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS_FILE)
    ap.add_argument("--symbols", default=None, help="comma-separated symbols; overrides --symbols-file")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--update", action="store_true",
                    help="delta-update existing mixed files instead of full backfill")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip symbols that already have a parquet in --out-dir")
    args = ap.parse_args()

    symbols = load_symbols(args.symbols_file, args.symbols)
    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]
    if not symbols:
        raise SystemExit("no symbols to fetch")

    store = CandleStore(Path(args.out_dir))
    lock_path = Path(args.out_dir).parent / "okx_liquid_backfill.lock"
    local = threading.local()

    def fetcher() -> CandleFetcher:
        f = getattr(local, "fetcher", None)
        if f is None:
            f = local.fetcher = CandleFetcher(
                OKXClient(timeout=args.timeout),
                store,
                sleep_seconds=args.page_sleep,
            )
        return f

    def work(sym: str) -> dict:
        try:
            if args.skip_existing and store.has(sym):
                df = store.load(sym)
                rows, first, last = coverage_status(df)
                return {"symbol": sym, "status": "cached", "rows": rows, "first": first, "last": last}

            result = fetcher().fetch_symbol(sym, resolutions=RESOLUTION_DAYS, update=args.update)
            df = store.load(sym)
            rows, first, last = coverage_status(df)
            return {
                "symbol": sym,
                "status": result.get("status", "?"),
                "rows": rows,
                "first": first,
                "last": last,
            }
        except Exception as exc:
            return {"symbol": sym, "status": "fail", "message": f"{type(exc).__name__}: {exc}"}

    with SingleInstanceLock(lock_path):
        t0 = time.time()
        print(
            f"okx liquid mixed backfill: symbols={len(symbols)} workers={args.workers} "
            f"resolutions={RESOLUTION_DAYS} out={Path(args.out_dir)}",
            flush=True,
        )
        ok = cached = fail = empty = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, sym) for sym in symbols]
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                status = str(r.get("status"))
                ok += int(status == "ok")
                cached += int(status == "cached")
                empty += int(status == "empty")
                fail += int(status not in {"ok", "cached", "empty"})
                if status in {"ok", "cached"}:
                    print(
                        f"  {i:3d}/{len(symbols)} {status.upper():6s} {r['symbol']:16s} "
                        f"rows={r['rows']:7d} {fmt_ts(r['first'])} -> {fmt_ts(r['last'])} "
                        f"elapsed={time.time() - t0:.0f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"  {i:3d}/{len(symbols)} {status.upper():6s} {r['symbol']:16s} "
                        f"{r.get('message', '')} elapsed={time.time() - t0:.0f}s",
                        flush=True,
                    )

        print(
            f"done: ok={ok} cached={cached} empty={empty} fail={fail} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
