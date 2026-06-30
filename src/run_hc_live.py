"""Run the HC live adapter.

Safe by default:
  python -m src.run_hc_live --shadow --once

Real OKX orders:
  python -m src.run_hc_live --live --stake-margin 5 --leverage 3
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from .trading import PaperExecutor, ShadowExecutor
from .trading.hc_live_engine import HCLiveEngine
from .trading.live_trader import LiveTrader
from .trading.okx_executor import OKXExecutor


def _parse_horizons(raw: str) -> tuple[int, ...] | None:
    if not raw.strip():
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    horizons = tuple(sorted(dict.fromkeys(int(p) for p in parts)))
    bad = [h for h in horizons if h <= 0]
    if bad:
        raise ValueError(f"--horizons must be positive minute values, got {bad}")
    return horizons


def _prob_label(value: float) -> str:
    return f"{int(round(float(value) * 100)):02d}"


def _load_horizon_thresholds(path: Path, shift: float = 0.0) -> dict[int, float]:
    out: dict[int, float] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = int(row["horizon"])
            threshold = float(row["threshold"]) + float(shift)
            out[h] = min(0.999, max(0.0, threshold))
    if not out:
        raise ValueError(f"no thresholds loaded from {path}")
    return dict(sorted(out.items()))


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="place REAL OKX orders")
    ap.add_argument("--shadow", action="store_true", help="live data, no orders")
    ap.add_argument("--demo", action="store_true", help="OKX simulated-trading sandbox")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--stake-margin", type=float, default=5.0)
    ap.add_argument("--leverage", type=int, default=3)
    ap.add_argument("--scan-interval-min", type=int, default=10)
    ap.add_argument("--confirm-sec", type=float, default=65.0)
    ap.add_argument("--deadline-check-sec", type=float, default=5.0)
    ap.add_argument("--max-anchor-lag-sec", type=float, default=180.0)
    ap.add_argument("--top-per-scan", type=int, default=3)
    ap.add_argument("--max-concurrent", type=int, default=10)
    ap.add_argument("--max-legs", type=int, default=1,
                    help="deprecated safety no-op; live now caps one active position per symbol")
    ap.add_argument("--conviction", action="store_true",
                    help="scale position size by signal conviction (p_dir - p_opp), 0.5x..2x")
    ap.add_argument("--cooldown-min", type=int, default=30)
    ap.add_argument("--watchlist-size", type=int, default=0, help="0 = all HC universe")
    ap.add_argument("--model-dir", type=Path, default=Path("models/hc_exec_stride120_nonoverlap"))
    ap.add_argument("--high", type=float, default=0.90)
    ap.add_argument("--opp-cap", type=float, default=0.20)
    ap.add_argument(
        "--selection-mode",
        choices=("plain", "squeezer", "quality", "bad_day_worker"),
        default="plain",
        help=(
            "plain: old p+opp gate; squeezer: p>=high OR spread>=floor; "
            "quality: ultra-strict p/spread tail (litmus regime gate); "
            "bad_day_worker: calm-day extractor, p_dir>=bdw-raw AND p_opp<=bdw-opp"
        ),
    )
    ap.add_argument("--spread-floor", type=float, default=None)
    ap.add_argument("--bdw-raw", type=float, default=0.80,
                    help="bad_day_worker: minimum p_dir (default 0.80)")
    ap.add_argument("--bdw-opp", type=float, default=0.05,
                    help="bad_day_worker: maximum p_opp (default 0.05)")
    ap.add_argument("--horizon-min", type=int, default=30)
    ap.add_argument("--horizon-max", type=int, default=90)
    ap.add_argument("--thresholds-csv", type=Path, default=None)
    ap.add_argument("--threshold-shift", type=float, default=0.0)
    ap.add_argument(
        "--horizons",
        default="",
        help="comma-separated HC horizon grid, e.g. 10,15,20,25,30,35,40,45,50,60,75,90,120",
    )
    ap.add_argument("--profile", default="")
    ap.add_argument("--system-name", default="Dancing Horizon")
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()

    _load_dotenv()
    notional = float(args.stake_margin) * float(args.leverage)
    threshold_map = (
        _load_horizon_thresholds(args.thresholds_csv, args.threshold_shift)
        if args.thresholds_csv else None
    )
    horizons = _parse_horizons(args.horizons)
    if horizons is None and threshold_map:
        horizons = tuple(sorted(threshold_map))
    if args.profile.strip():
        profile = args.profile.strip()
    elif threshold_map:
        profile = f"thr{len(threshold_map)}_shift{args.threshold_shift:+.3f}_opp{_prob_label(args.opp_cap)}"
    elif args.selection_mode == "bad_day_worker":
        profile = f"bad_day_worker_p{_prob_label(args.bdw_raw)}_opp{_prob_label(args.bdw_opp)}"
    elif args.selection_mode == "quality":
        profile = "quality_litmus"
    elif args.selection_mode == "squeezer":
        profile = f"squeezer_p{_prob_label(args.high)}_sf{_prob_label(args.spread_floor or 0.80)}"
    else:
        profile = f"plain_mid_p{_prob_label(args.high)}_opp{_prob_label(args.opp_cap)}"

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

    engine = HCLiveEngine(
        model_dir=args.model_dir,
        high=args.high,
        opp_cap=args.opp_cap,
        horizon_min=args.horizon_min,
        horizon_max=args.horizon_max,
        horizons=horizons,
        notional_usd=notional,
        thresholds_by_horizon=threshold_map,
        profile=profile,
        system_name=args.system_name,
        max_legs=args.max_legs,
        conviction=args.conviction,
        selection_mode=args.selection_mode,
        spread_floor=args.spread_floor,
        bdw_raw=args.bdw_raw,
        bdw_opp=args.bdw_opp,
    )
    print(f"hc live engine: {engine.describe()}")
    print(
        f"backend={executor.mode} demo={getattr(executor, 'demo', False)} "
        f"stake_margin=${args.stake_margin:.2f} leverage={args.leverage} "
        f"notional=${notional:.2f} scan={args.scan_interval_min}m "
        f"deadline={args.deadline_check_sec:.1f}s max_lag={args.max_anchor_lag_sec:.0f}s "
        f"top={args.top_per_scan} cap={args.max_concurrent} cd={args.cooldown_min}m "
        f"fetch={not args.no_fetch} once={args.once}"
    )

    import src.config as _C

    _C.LIVE_WATCHLIST_SIZE = int(args.watchlist_size)
    trader = LiveTrader(
        executor,
        trust_engine=engine,
        top_per_scan=args.top_per_scan,
        fetch=not args.no_fetch,
        scan_interval_min=args.scan_interval_min,
        trade_size_usd=notional,
        max_concurrent=args.max_concurrent,
        cooldown_min=args.cooldown_min,
        max_legs=args.max_legs,
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
