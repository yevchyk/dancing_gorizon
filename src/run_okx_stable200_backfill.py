"""Backfill/update the isolated OKX stable-200 candle store.

This is intentionally separate from data/candles and the existing crypto/live
store. It uses the same battle-tested CandleFetcher depth:
1m ~7d, 5m ~240d, 1H ~730d, 1D ~1460d merged into one parquet per symbol.

Examples:
  python -m src.run_okx_stable200_build
  python -m src.run_okx_stable200_backfill --workers 4
  python -m src.run_okx_stable200_backfill --update --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import config as C
from .database import CandleFetcher, CandleStore, OKXClient
from .database.candle_fetcher import RESOLUTION_DAYS


DEFAULT_SYMBOLS_FILE = C.CONFIGS_DIR / "okx_stable_200.json"
DEFAULT_OUT_DIR = C.DATA_DIR / "okx_stable" / "candles_mixed"
DEFAULT_SEED_FROM_DIR = C.CANDLES_DIR


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
                f"OKX stable-200 backfill lock exists: {self.path} (owner: {owner}). "
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


def seed_from_existing(symbols: list[str], out_dir: Path, seed_from_dir: Path | None) -> dict[str, int]:
    if seed_from_dir is None:
        return {"seeded": 0, "already": 0, "missing": 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    seeded = already = missing = 0
    for sym in symbols:
        dst = out_dir / f"{sym}.parquet"
        if dst.exists():
            already += 1
            continue
        src = seed_from_dir / f"{sym}.parquet"
        if not src.exists():
            missing += 1
            continue
        shutil.copy2(src, dst)
        seeded += 1
    return {"seeded": seeded, "already": already, "missing": missing}


def prune_extra_files(symbols: list[str], out_dir: Path) -> int:
    out_dir = out_dir.resolve()
    expected_root = DEFAULT_OUT_DIR.resolve().parent
    if expected_root not in (out_dir, *out_dir.parents):
        raise SystemExit(f"refusing to prune outside stable store root: {out_dir}")

    keep = {f"{sym}.parquet" for sym in symbols}
    removed = 0
    for path in out_dir.glob("*.parquet"):
        if path.name in keep:
            continue
        path.unlink()
        removed += 1
    return removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--page-sleep", type=float, default=0.10)
    ap.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS_FILE)
    ap.add_argument("--symbols", default=None, help="comma-separated symbols; overrides --symbols-file")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed-from-dir", type=Path, default=DEFAULT_SEED_FROM_DIR,
                    help="copy matching existing parquets into --out-dir before fetching")
    ap.add_argument("--no-seed", action="store_true", help="do not seed from data/candles")
    ap.add_argument("--update", action="store_true",
                    help="delta-update existing mixed files instead of full backfill")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip symbols that already have a parquet in --out-dir")
    ap.add_argument("--prune-extra", action="store_true",
                    help="delete parquets in --out-dir that are not declared in the symbols file")
    args = ap.parse_args()

    symbols = load_symbols(args.symbols_file, args.symbols)
    if args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]
    if not symbols:
        raise SystemExit("no symbols to fetch")

    store = CandleStore(Path(args.out_dir))
    lock_path = Path(args.out_dir).parent / "okx_stable_200_backfill.lock"
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
        pruned = prune_extra_files(symbols, Path(args.out_dir)) if args.prune_extra else 0
        seed_stats = seed_from_existing(
            symbols,
            Path(args.out_dir),
            None if args.no_seed else Path(args.seed_from_dir),
        )
        print(
            f"okx stable-200 mixed backfill: symbols={len(symbols)} workers={args.workers} "
            f"resolutions={RESOLUTION_DAYS} out={Path(args.out_dir)} "
            f"pruned={pruned} "
            f"seeded={seed_stats['seeded']} already={seed_stats['already']} "
            f"seed_missing={seed_stats['missing']}",
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
