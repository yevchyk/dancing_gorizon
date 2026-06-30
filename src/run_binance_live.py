"""Run a portfolio of Binance-model explorer builds live on BINANCE (shadow by default).

Mirror of run_hc_portfolio_live, pointed at the Binance world end-to-end:
  * candles  = data/binance/candles (the year store the models were trained on);
    the HC pipeline store is PATCHED exactly like run_binance_dataset, so live
    features are bit-identical to the dataset features (verified separately);
  * scoring  = the binance_y1 v4 seed-ensembles via HCPortfolioEngine/HCV4LiveEngine;
  * builds   = saved explorer builds (configs/builds/*.json) applied as ONE risk
    book with cross-dedup, shared max-concurrent + cooldown (LiveTrader);
  * orders   = BinanceExecutor (USDT-M fapi). Safe by default: --shadow places
    nothing; --testnet trades the Binance futures sandbox; only --live is real.

Credentials (.env): BINANCE_API_KEY / BINANCE_SECRET_KEY (live),
BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_SECRET_KEY (testnet).

  # dry run on live Binance data (no orders), one scan:
  python -m src.run_binance_live --portfolio configs/builds/binance_shadow_portfolio.json --shadow --once

  # continuous shadow (the pre-registered forward-shadow stage):
  python -m src.run_binance_live --shadow

  # sandbox orders / REAL money (gated, keep small):
  python -m src.run_binance_live --testnet
  python -m src.run_binance_live --live
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .markets import REGISTRY, Store
from . import config as C
from .hc import config as HC

BINANCE_CANDLES = Path("data/binance/candles")

# --- point the HC pipeline at the Binance store BEFORE importing live engines
# (same patch as run_binance_dataset: _load_raw()/prepare_btc_frames() read
# STORE_KEY at call time; HC_ERA_START keeps the full Binance year visible).
REGISTRY["binance_feature"] = Store(
    "binance_feature", "crypto", "feature", "1m",
    C.ROOT / BINANCE_CANDLES, C.ROOT / "configs" / "binance_train_universe.json",
    "Binance USDT-M 365d 1m store for the year-rebuild (BINANCE_PLAN.md).")
HC.STORE_KEY = "binance_feature"
HC.HC_ERA_START = pd.Timestamp("2025-06-01T00:00:00Z")

from . import binance_fetcher as BF  # noqa: E402
from .trading import PaperExecutor, ShadowExecutor  # noqa: E402
from .trading.binance_executor import BinanceExecutor, to_binance_sym  # noqa: E402
from .trading.hc_portfolio_engine import HCPortfolioEngine  # noqa: E402
from .trading.live_trader import LiveTrader  # noqa: E402


class BinancePortfolioEngine(HCPortfolioEngine):
    """HCPortfolioEngine minus the OKX toxic blacklist: the Binance trade
    universe is already liquidity- and trusted-cost-filtered, and the builds
    were tuned in the explorer WITHOUT the OKX list — applying it live would
    silently shrink the book vs the sim. Build bans still apply."""

    def build_watchlist(self, store, top_n: int = 0, logger=None) -> list[str]:
        data = json.loads(self.universe_path.read_text(encoding="utf-8"))
        universe = data.get("symbols", data) if isinstance(data, dict) else data
        # drop only symbols banned by EVERY build (intersection): a ban is a
        # per-build filter (applied in _legs_for_build), not a portfolio veto —
        # the union here silenced 63/153 symbols for ALL builds.
        banned_every = (set.intersection(*[set(b["banned"]) for b in self.builds])
                        if self.builds else set())
        watch = []
        missing = 0
        for sym in sorted(str(s) for s in universe):
            if sym in banned_every:
                continue
            candles = store.load(sym)
            if candles is None or candles.empty:
                missing += 1
                continue
            watch.append(sym)
        if logger is not None:
            logger.event(f"watchlist: binance trade universe: {len(watch)} symbols "
                         f"(universe={len(universe)}, missing={missing}, "
                         f"banned-by-all={len(banned_every)}, okx-blacklist NOT applied)")
        return watch


class BinanceLiveFetcher:
    """LiveTrader-compatible per-symbol candle top-up via the Binance klines API.

    fetch_symbol() is resume-able (continues from the last stored candle), so a
    small `days` floor keeps each call cheap once the store is warm.
    """

    def __init__(self, min_interval_sec: float = 0.12):
        BF._MIN_INTERVAL = float(min_interval_sec)

    def update_recent(self, symbol: str, lookback_min: int) -> int:
        days = max(1, math.ceil(float(lookback_min) / 1440.0))
        return BF.fetch_symbol(to_binance_sym(symbol), "1m", days=days)


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


def _prefetch_and_check(universe: Path, days: int = 2) -> None:
    """Self-sufficient startup: top up the Binance store + freshness log."""
    print(f"[startup] binance gap fill: top-up {days}d for the trade universe...", flush=True)
    t0 = pd.Timestamp.now()
    subprocess.run([sys.executable, "-m", "src.binance_fetcher",
                    "--universe", str(universe), "--days", str(days),
                    "--workers", "8", "--min-interval", "0.12"])
    try:
        edge = None
        for sym in ("ETH_USDT_SWAP", "SOL_USDT_SWAP", "BTC_USDT_SWAP"):
            p = BINANCE_CANDLES / f"{sym}.parquet"
            if p.exists():
                e = pd.read_parquet(p, columns=["timestamp"])["timestamp"].max()
                edge = e if edge is None else max(edge, e)
        lag = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(edge)).total_seconds() / 60.0
        print(f"[startup] gap fill done in {(pd.Timestamp.now() - t0).total_seconds():.0f}s. "
              f"freshest candle={edge} (lag {lag:.0f}m)", flush=True)
    except Exception as e:
        print(f"[startup] freshness read failed: {e}", flush=True)


def _other_real_live_pids() -> list[int]:
    if os.name != "nt":
        return []
    cmd = (
        f"Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.ProcessId -ne {os.getpid()} -and "
        f"$_.ProcessId -ne {os.getppid()} -and "
        "$_.CommandLine -match 'src.run_binance_live' -and "
        "$_.CommandLine -match '--live' }} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd],
                                      text=True, stderr=subprocess.DEVNULL)
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
    ap.add_argument("--portfolio", type=Path,
                    default=Path("configs/builds/binance_shadow_portfolio.json"))
    ap.add_argument("--live", action="store_true", help="place REAL Binance orders")
    ap.add_argument("--testnet", action="store_true", help="Binance futures sandbox orders")
    ap.add_argument("--shadow", action="store_true", help="live data, no orders (default)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--stake-margin", type=float, default=None)
    ap.add_argument("--leverage", type=int, default=None)
    ap.add_argument("--scan-interval-min", type=int, default=5)
    ap.add_argument("--confirm-sec", type=float, default=65.0)
    ap.add_argument("--max-anchor-lag-sec", type=float, default=180.0)
    ap.add_argument("--watchlist-size", type=int, default=0)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--prefetch-days", type=int, default=2,
                    help="startup top-up depth for the candle store (0=skip)")
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
    universe = Path(cfg.get("universe", "configs/binance_universe_trade.json"))

    if args.balance:
        ex = BinanceExecutor(live=True, demo=args.testnet, leverage=leverage)
        if not ex.has_credentials():
            raise SystemExit("no Binance credentials in env (.env)")
        print(f"USDT available: {ex.equity():.2f}  (testnet={ex.demo})")
        return

    if args.live or args.testnet:
        if args.live and not args.testnet:
            others = _other_real_live_pids()
            if others:
                raise SystemExit("another real binance live runner is already active: "
                                 + ",".join(str(x) for x in others))
        executor = BinanceExecutor(live=True, demo=args.testnet, leverage=leverage)
        if not executor.has_credentials():
            which = ("BINANCE_TESTNET_API_KEY/BINANCE_TESTNET_SECRET_KEY" if args.testnet
                     else "BINANCE_API_KEY/BINANCE_SECRET_KEY")
            raise SystemExit(f"--{'testnet' if args.testnet else 'live'} needs {which} in .env")
    elif args.shadow:
        executor = ShadowExecutor()
    else:
        executor = ShadowExecutor()
        print("no mode flag given -> defaulting to SHADOW (no orders)", flush=True)

    engine = BinancePortfolioEngine(
        builds,
        notional_usd=notional,
        universe_path=universe,
        profile=cfg.get("name", "binance_portfolio"),
        min_p_dir=float(cfg.get("min_p_dir", 0.55)),
        slots_per_engine=int(cfg.get("slots_per_engine", 0)),
        consensus_boost=cfg.get("consensus_boost"),
        max_stake_mult=float(cfg.get("max_stake_mult", 3.0)),
    )
    print(f"binance portfolio engine: {engine.describe()}")
    print(f"backend={executor.mode} testnet={getattr(executor, 'demo', False)} "
          f"stake_margin=${stake:.2f} leverage={leverage} notional=${notional:.2f}/pos "
          f"builds={len(builds)} scan={args.scan_interval_min}m maxconc={max_concurrent} "
          f"cooldown={cooldown_min}m top={top_per_scan} universe={universe.name} "
          f"fetch={not args.no_fetch} once={args.once} live={args.live}")

    import src.config as _C
    _C.LIVE_WATCHLIST_SIZE = int(args.watchlist_size)

    if not args.no_fetch and args.prefetch_days > 0:
        _prefetch_and_check(universe, int(args.prefetch_days))

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
        max_anchor_lag_sec=args.max_anchor_lag_sec,
        store_root=BINANCE_CANDLES,
        fetcher=BinanceLiveFetcher(),
    )

    if args.once:
        trader.scan_once()
    else:
        trader.run()


if __name__ == "__main__":
    main()
