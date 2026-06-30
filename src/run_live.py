"""Run the live trader. Credentials come from the environment (never hard-coded):
  OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE   (only needed for --live)

Backends (safe by default):
  (none)     paper  -> simulated fills, no orders            [default]
  --shadow   shadow -> live data, logs intended trades only
  --live     okx    -> REAL orders (requires credentials)
  --demo     use OKX simulated-trading sandbox (with --live)

Examples:
  python -m src.run_live --once                 # one paper scan from cached candles
  python -m src.run_live --shadow --once        # one live-data dry scan
  python -m src.run_live --live --demo          # loop on OKX demo account
  python -m src.run_live --live                 # loop with REAL money
  python -m src.run_live --balance              # check OKX balance and exit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .trading import PaperExecutor, ShadowExecutor, load_signal_thresholds
from .trading.okx_executor import OKXExecutor
from .trading.live_trader import LiveTrader


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a project-root .env into the environment
    (without overriding values already set in the shell)."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _load_settings() -> dict:
    """User-editable settings.json (money, frequency...) overriding code defaults."""
    import json
    from . import config as C
    s = {"trade_size_usd": C.TRADE_SIZE_USD, "scan_interval_min": C.SCAN_INTERVAL_MIN,
         "signal_floor": C.SIGNAL_FLOOR, "watchlist_size": C.LIVE_WATCHLIST_SIZE,
         "max_concurrent": C.MAX_CONCURRENT, "cooldown_min": C.COOLDOWN_MIN,
         "top_per_scan": C.CONF_TOP_PER_SCAN, "green_harvest": C.GREEN_HARVEST,
         "leverage": 1, "scan_confirmation_sec": 65.0}
    f = Path(__file__).resolve().parent.parent / "settings.json"
    if f.exists():
        s.update({k: v for k, v in json.loads(f.read_text()).items() if k in s})
    return s


def _load_symbols_file(path: str | None) -> list[str] | None:
    if not path:
        return None
    import json

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("symbols", data) if isinstance(data, dict) else data
    symbols: list[str] = []
    seen: set[str] = set()
    for value in raw:
        sym = str(value).strip().upper().replace("-", "_")
        if sym and sym not in seen:
            symbols.append(sym)
            seen.add(sym)
    return symbols


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="place REAL OKX orders")
    p.add_argument("--shadow", action="store_true", help="live data, no orders")
    p.add_argument("--demo", action="store_true", help="OKX simulated-trading sandbox")
    p.add_argument("--once", action="store_true", help="single scan then exit")
    p.add_argument("--balance", action="store_true", help="print OKX balance and exit")
    p.add_argument("--no-fetch", action="store_true", help="use cached candles, don't download")
    p.add_argument("--top-pct", type=float, default=1.0)
    p.add_argument("--trust", action="store_true", help="use the trust-layer engine")
    p.add_argument("--conf", action="store_true", help="use the v4 high-confidence clean engine")
    p.add_argument("--fast-combo", action="store_true", help="use fast_v2 combo live engine")
    p.add_argument("--fast-profile", default="combo00_flat670",
                   choices=("combo00_flat670", "combo00_flatstrict",
                            "combo01_tight_toxic5", "combo01_tight_toxic8",
                            "pulse00", "pulse05", "only_forward",
                            "up3", "asym", "pulse", "drill"),
                   help="single fast-combo profile (pulse00 = Unicorn)")
    p.add_argument("--fast-stack", default=None,
                   choices=("unicorn_solo", "unicorn_plus", "unicorn_pulse",
                            "unicorn_drill", "only_forward"),
                   help="run a named multi-engine stack instead of one profile")
    p.add_argument("--fast-v3-profile", default=None,
                   choices=("verkh_v2", "unicorn_v2", "v2_pair",
                            "unicorn_v2_inverted", "inverted_solo"),
                   help="run a fast_v3 live adapter (v2_pair = unicorn_v2 + verkh_v2)")
    p.add_argument("--store-dir", default=None,
                   help="override candle store root, e.g. data/okx_liquid/candles_mixed")
    p.add_argument("--symbols-file", default=None,
                   help="JSON list/dict of symbols to trade; overrides engine watchlist")
    p.add_argument("--fast-size-mult", type=float, default=None,
                   help="override fast-combo pulse_size_mult for small live tests")
    p.add_argument("--global-trust", type=float, default=0.0, help="trust pedal (higher=fewer)")
    args = p.parse_args()

    _load_dotenv()

    if args.balance:
        ex = OKXExecutor(live=True, demo=args.demo)
        if not ex.has_credentials():
            raise SystemExit("no OKX credentials in env (OKX_API_KEY/SECRET/PASSPHRASE)")
        print(f"USDT available: {ex.equity():.2f}  (demo={ex.demo})")
        return

    cfg = _load_settings()

    if args.live:
        executor = OKXExecutor(live=True, demo=args.demo, leverage=cfg["leverage"])
        if not executor.has_credentials():
            raise SystemExit("--live needs OKX_API_KEY/OKX_SECRET_KEY/OKX_PASSPHRASE in env")
    elif args.shadow:
        executor = ShadowExecutor()
    else:
        executor = PaperExecutor()
    watch = "all" if not cfg["watchlist_size"] else str(cfg["watchlist_size"])
    print(f"settings: size=${cfg['trade_size_usd']}  scan={cfg['scan_interval_min']}min  "
          f"floor={cfg['signal_floor']}  watchlist={watch}")

    engine = None
    if args.fast_v3_profile:
        from .trading.fast_v3_engine import FastV3Engine, FastV3Stack, STACKS
        if args.fast_v3_profile in STACKS:
            engine = FastV3Stack.from_stack(args.fast_v3_profile)
        else:
            engine = FastV3Engine(profile=args.fast_v3_profile)
        if cfg.get("scan_interval_min") != 2:
            print(f"fast v3 forcing scan_interval_min=2 "
                  f"(settings had {cfg.get('scan_interval_min')})")
        cfg["scan_interval_min"] = 2
        cfg["green_harvest"] = False
        print(f"fast v3 engine: {engine.describe()}")
    elif args.fast_combo or args.fast_stack:
        if args.fast_stack:
            from .trading.multi_engine import MultiEngine
            engine = MultiEngine.from_stack(args.fast_stack)
        else:
            from .trading.fast_combo_engine import FastComboEngine
            engine = FastComboEngine(profile=args.fast_profile)
        if args.fast_size_mult is not None and hasattr(engine, "cfg"):
            engine.cfg["pulse_size_mult"] = float(args.fast_size_mult)
        # The tested fast-combo logic is patient/no-harvest, with 2m scan and
        # short per-symbol cooldown. Keep the live cadence on the same 2m grid
        # as the holdout tests.
        if cfg.get("scan_interval_min") != 2:
            print(f"fast combo forcing scan_interval_min=2 "
                  f"(settings had {cfg.get('scan_interval_min')})")
        cfg["scan_interval_min"] = 2
        cfg["green_harvest"] = False
        print(f"fast combo engine: {engine.describe()}")
    elif args.conf:
        from .trading.conf_engine import ConfEngine
        engine = ConfEngine(floor=cfg["signal_floor"])
        print(f"conf engine: floor={engine.floor} clean_opp={engine.clean_opp} "
              f"min_agree={engine.min_agree}")
    elif args.trust:
        from .trading.trust_engine import TrustEngine
        engine = TrustEngine(global_trust=args.global_trust)
        print(f"trust engine: trusted models = {engine.trusted_models()}")

    import src.config as _C
    _C.LIVE_WATCHLIST_SIZE = cfg["watchlist_size"]
    symbols = _load_symbols_file(args.symbols_file)
    trader = LiveTrader(executor, symbols=symbols, trust_engine=engine,
                        thresholds=load_signal_thresholds(top_pct=args.top_pct),
                        top_per_scan=cfg["top_per_scan"],
                        fetch=not args.no_fetch,
                        scan_interval_min=cfg["scan_interval_min"],
                        trade_size_usd=cfg["trade_size_usd"],
                        max_concurrent=cfg["max_concurrent"],
                        cooldown_min=cfg["cooldown_min"],
                        green_harvest=cfg["green_harvest"],
                        scan_confirmation_sec=cfg["scan_confirmation_sec"],
                        store_root=args.store_dir)
    eng_kind = ("fast-v3" if args.fast_v3_profile
                else "fast-combo" if (args.fast_combo or args.fast_stack)
                else "conf" if args.conf else "trust" if engine else "legacy")
    print(f"backend={executor.mode}  symbols={len(trader.symbols)}  "
          f"engine={eng_kind}  once={args.once}  fetch={not args.no_fetch}")
    if args.once:
        trader.scan_once()
    else:
        trader.run()


if __name__ == "__main__":
    main()
