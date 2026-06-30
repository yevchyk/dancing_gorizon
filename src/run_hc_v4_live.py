"""Run the v4 (1-min-horizon) model min1_2to120 live.

Safe by default (shadow = live data, no orders):
  python -m src.run_hc_v4_live --shadow
  python -m src.run_hc_v4_live --shadow --once
REAL OKX orders:
  python -m src.run_hc_v4_live --live --stake-margin 5 --leverage 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .trading import PaperExecutor, ShadowExecutor
from .trading.hc_v4_live_engine import HCV4LiveEngine
from .trading.live_trader import LiveTrader
from .trading.okx_executor import OKXExecutor


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _horizons(raw: str) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in raw.replace(";", ",").split(",") if x.strip()))


def _prefetch_and_check(lookback_min: int) -> None:
    """Self-sufficient startup: deep-fetch to fill candle gaps + integrity log."""
    import subprocess
    import sys
    import pandas as pd
    from . import config as C
    from .hc import config as HC
    print(f"[startup] integrity + gap fill: fetching last {lookback_min}m for all symbols...", flush=True)
    t0 = pd.Timestamp.now()
    subprocess.run([sys.executable, "-m", "src.run_fetcher", "--once", "--universe", "store",
                    "--workers", "12", "--lookback-min", str(lookback_min)])
    try:
        edge = pd.read_parquet(C.CANDLES_DIR / f"{HC.BTC_SYMBOL}.parquet", columns=["timestamp"])["timestamp"].max()
        lag = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(edge)).total_seconds() / 60.0
        took = (pd.Timestamp.now() - t0).total_seconds()
        print(f"[startup] gap fill done in {took:.0f}s. freshest candle={edge} (lag {lag:.0f}m)", flush=True)
    except Exception as e:
        print(f"[startup] integrity read failed: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="place REAL OKX orders")
    ap.add_argument("--shadow", action="store_true", help="live data, no orders")
    ap.add_argument("--demo", action="store_true", help="OKX simulated-trading sandbox")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--model-dir", type=Path, default=Path("models/min1_2to120"))
    ap.add_argument("--high", type=float, default=0.85)
    ap.add_argument("--horizons", default="60,75,90,105,120")
    ap.add_argument("--stake-margin", type=float, default=5.0)
    ap.add_argument("--leverage", type=int, default=3)
    ap.add_argument("--scan-interval-min", type=int, default=5)
    ap.add_argument("--confirm-sec", type=float, default=65.0)
    ap.add_argument("--deadline-check-sec", type=float, default=5.0)
    ap.add_argument("--max-anchor-lag-sec", type=float, default=180.0)
    ap.add_argument("--top-per-scan", type=int, default=8)
    ap.add_argument("--max-concurrent", type=int, default=10)
    ap.add_argument("--cooldown-min", type=int, default=30)
    ap.add_argument("--watchlist-size", type=int, default=0)
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_full.json"))
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--prefetch-min", type=int, default=1500,
                    help="deep startup fetch to fill candle gaps before trading (0=skip)")
    args = ap.parse_args()

    _load_dotenv()
    notional = float(args.stake_margin) * float(args.leverage)

    if args.balance:
        ex = OKXExecutor(live=True, demo=args.demo, leverage=args.leverage)
        if not ex.has_credentials():
            raise SystemExit("no OKX credentials in env")
        print(f"USDT available: {ex.equity():.2f}  (demo={ex.demo})")
        return

    if args.live:
        executor = OKXExecutor(live=True, demo=args.demo, leverage=args.leverage)
        if not executor.has_credentials():
            raise SystemExit("--live needs OKX_API_KEY/OKX_SECRET_KEY/OKX_PASSPHRASE")
    elif args.shadow:
        executor = ShadowExecutor()
    else:
        executor = PaperExecutor()

    engine = HCV4LiveEngine(
        model_dir=args.model_dir,
        high=args.high,
        horizons=_horizons(args.horizons),
        notional_usd=notional,
        universe_path=args.universe,
    )
    print(f"v4 live engine: {engine.describe()}")
    print(f"backend={executor.mode} demo={getattr(executor, 'demo', False)} "
          f"stake_margin=${args.stake_margin:.2f} leverage={args.leverage} notional=${notional:.2f} "
          f"scan={args.scan_interval_min}m cap={args.max_concurrent} cd={args.cooldown_min}m "
          f"top={args.top_per_scan} fetch={not args.no_fetch} once={args.once} live={args.live}")

    import src.config as _C
    _C.LIVE_WATCHLIST_SIZE = int(args.watchlist_size)

    # self-sufficient startup: fill candle gaps + integrity check before trading
    if not args.no_fetch and args.prefetch_min > 0:
        _prefetch_and_check(int(args.prefetch_min))

    trader = LiveTrader(
        executor,
        trust_engine=engine,
        top_per_scan=args.top_per_scan,
        fetch=not args.no_fetch,
        scan_interval_min=args.scan_interval_min,
        trade_size_usd=notional,
        max_concurrent=args.max_concurrent,
        cooldown_min=args.cooldown_min,
        max_legs=1,
        green_harvest=False,
        scan_confirmation_sec=args.confirm_sec,
        deadline_check_sec=args.deadline_check_sec,
        max_anchor_lag_sec=args.max_anchor_lag_sec,
    )
    if args.once:
        trader.scan_once()
    else:
        trader.run()


if __name__ == "__main__":
    main()
