"""Run a SAVED PORTFOLIO of explorer builds live (shadow by default).

A portfolio config = {stake_margin, leverage, max_concurrent, cooldown_min,
top_per_scan, universe, builds:[ <explorer build>, ... ]}.  All builds share ONE
risk book (1 position/symbol, shared max-concurrent + cooldown), cross-deduped by
p_dir.  Safe by default — only --live places real OKX orders.

  # dry run on live data (no orders), one scan:
  python -m src.run_hc_portfolio_live --portfolio configs/builds/portfolio_5x3.json --shadow --once

  # continuous shadow:
  python -m src.run_hc_portfolio_live --portfolio configs/builds/portfolio_5x3.json --shadow

  # REAL money ($5 margin x3 leverage per position, 3 engines, one book):
  python -m src.run_hc_portfolio_live --portfolio configs/builds/portfolio_5x3.json --live
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .trading import PaperExecutor, ShadowExecutor
from .trading.hc_portfolio_engine import HCPortfolioEngine
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
        print(f"[startup] gap fill done in {(pd.Timestamp.now()-t0).total_seconds():.0f}s. "
              f"freshest candle={edge} (lag {lag:.0f}m)", flush=True)
    except Exception as e:
        print(f"[startup] integrity read failed: {e}", flush=True)


def _other_real_live_pids() -> list[int]:
    """Find already-running real-money portfolio runners to avoid duplicate orders."""
    if os.name != "nt":
        return []
    cmd = (
        f"Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.ProcessId -ne {os.getpid()} -and "
        f"$_.ProcessId -ne {os.getppid()} -and "
        "$_.CommandLine -match 'src.run_hc_portfolio_live' -and "
        "$_.CommandLine -match '--live' -and "
        "$_.CommandLine -notmatch '--demo' }} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            pass
    return pids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", type=Path, default=Path("configs/builds/portfolio_5x3.json"))
    ap.add_argument("--live", action="store_true", help="place REAL OKX orders")
    ap.add_argument("--shadow", action="store_true", help="live data, no orders")
    ap.add_argument("--demo", action="store_true", help="OKX simulated-trading sandbox")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--stake-margin", type=float, default=None, help="override config stake")
    ap.add_argument("--leverage", type=int, default=None, help="override config leverage")
    ap.add_argument("--scan-interval-min", type=int, default=5)
    ap.add_argument("--confirm-sec", type=float, default=65.0)
    ap.add_argument("--deadline-check-sec", type=float, default=5.0)
    ap.add_argument("--max-anchor-lag-sec", type=float, default=180.0)
    ap.add_argument("--watchlist-size", type=int, default=0, help="0 = all universe")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--prefetch-min", type=int, default=1500,
                    help="deep startup fetch to fill candle gaps before trading (0=skip)")
    args = ap.parse_args()

    _load_dotenv()
    cfg = json.loads(args.portfolio.read_text(encoding="utf-8"))
    builds = cfg.get("builds", [])
    if not builds:
        raise SystemExit(f"no builds in {args.portfolio}")

    stake = float(args.stake_margin if args.stake_margin is not None else cfg.get("stake_margin", 5.0))
    leverage = int(args.leverage if args.leverage is not None else cfg.get("leverage", 3))
    notional = stake * leverage
    max_concurrent = int(cfg.get("max_concurrent", 12))
    cooldown_min = int(cfg.get("cooldown_min", 30))
    top_per_scan = int(cfg.get("top_per_scan", 10))
    universe = Path(cfg.get("universe", "configs/hc_universe_full.json"))

    if args.balance:
        ex = OKXExecutor(live=True, demo=args.demo, leverage=leverage)
        if not ex.has_credentials():
            raise SystemExit("no OKX credentials in env (.env)")
        print(f"USDT available: {ex.equity():.2f}  (demo={ex.demo})")
        return

    if args.live:
        if not args.demo:
            others = _other_real_live_pids()
            if others:
                raise SystemExit(
                    "another real live portfolio runner is already active: "
                    + ",".join(str(x) for x in others)
                )
        executor = OKXExecutor(live=True, demo=args.demo, leverage=leverage)
        if not executor.has_credentials():
            raise SystemExit("--live needs OKX_API_KEY/OKX_SECRET_KEY/OKX_PASSPHRASE in .env")
    elif args.shadow:
        executor = ShadowExecutor()
    else:
        executor = PaperExecutor()

    engine = HCPortfolioEngine(
        builds,
        notional_usd=notional,
        universe_path=universe,
        profile=cfg.get("name", "portfolio"),
        min_p_dir=float(cfg.get("min_p_dir", 0.70)),
        slots_per_engine=int(cfg.get("slots_per_engine", 0)),
        consensus_boost=cfg.get("consensus_boost"),
        max_stake_mult=float(cfg.get("max_stake_mult", 3.0)),
    )
    print(f"portfolio engine: {engine.describe()}")
    print(
        f"backend={executor.mode} demo={getattr(executor, 'demo', False)} "
        f"stake_margin=${stake:.2f} leverage={leverage} notional=${notional:.2f}/pos "
        f"builds={len(builds)} scan={args.scan_interval_min}m "
        f"maxconc={max_concurrent} cooldown={cooldown_min}m top={top_per_scan} "
        f"universe={universe.name} fetch={not args.no_fetch} once={args.once} live={args.live}"
    )

    import src.config as _C
    _C.LIVE_WATCHLIST_SIZE = int(args.watchlist_size)

    if not args.no_fetch and args.prefetch_min > 0:
        _prefetch_and_check(int(args.prefetch_min))

    trader = LiveTrader(
        executor,
        trust_engine=engine,
        top_per_scan=top_per_scan,
        fetch=not args.no_fetch,
        scan_interval_min=args.scan_interval_min,
        trade_size_usd=notional,
        max_concurrent=max_concurrent,
        cooldown_min=cooldown_min,
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
