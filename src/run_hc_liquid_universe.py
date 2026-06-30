"""Generate a LIQUID-only universe for live trading — Fix 3 of the sim plan.

The full universe (configs/hc_universe_full.json) includes tokenized equities and
thin micro-caps whose real execution slippage the sim never modelled. On
2026-06-09 the only two live losers were exactly those: MRVL (equity) and H (thin,
1.6% bar-range). This drops them by a UNIT-FREE jumpiness gate (median 1-min
bar-range), keeping liquid names where candle prices ≈ fills.

  python -m src.run_hc_liquid_universe                 # writes configs/hc_universe_liquid.json
  python -m src.run_hc_liquid_universe --max-barrange 0.25 --drop-equities

Keep rule: NOT equity (unless --keep-equities) AND median bar-range% <= cap AND
fresh+long-enough candle history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config as C
from .database import CandleStore
from .hc.costs import barrange_pct
from .markets import is_equity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_full.json"))
    ap.add_argument("--out", type=Path, default=Path("configs/hc_universe_liquid.json"))
    ap.add_argument("--max-barrange", type=float, default=0.25,
                    help="max median 1-min bar-range%% to keep (liquid)")
    ap.add_argument("--lookback-min", type=int, default=240)
    ap.add_argument("--min-candles", type=int, default=240)
    ap.add_argument("--max-stale-min", type=float, default=1440.0,
                    help="drop symbols whose last candle is older than this")
    ap.add_argument("--keep-equities", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.universe.read_text(encoding="utf-8"))
    syms = data.get("symbols", data) if isinstance(data, dict) else data
    store = CandleStore(C.CANDLES_DIR)
    now = pd.Timestamp.now(tz="UTC")

    kept, dropped = [], []
    for s in sorted(str(x) for x in syms):
        c = store.load(s)
        reason = None
        if c is None or c.empty or len(c) < args.min_candles:
            reason = "no/short data"
        else:
            idx = c.index.tz_localize("UTC") if c.index.tz is None else c.index
            stale_min = (now - idx.max()).total_seconds() / 60.0
            br = barrange_pct(c, lookback_min=args.lookback_min)
            if is_equity(s) and not args.keep_equities:
                reason = "equity"
            elif stale_min > args.max_stale_min:
                reason = f"stale {stale_min:.0f}m"
            elif br is None:
                reason = "no barrange"
            elif br > args.max_barrange:
                reason = f"jumpy {br:.2f}%"
        if reason:
            dropped.append((s, reason))
        else:
            kept.append(s)

    out = {
        "symbols": kept,
        "meta": {
            "source": args.universe.name,
            "generated": now.isoformat(),
            "rule": f"not-equity & barrange<= {args.max_barrange}% & fresh<= {args.max_stale_min}m",
            "kept": len(kept), "dropped": len(dropped),
        },
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    eq_dropped = sum(1 for _, r in dropped if r == "equity")
    jumpy_dropped = sum(1 for _, r in dropped if r.startswith("jumpy"))
    print(f"wrote {args.out}: kept={len(kept)} dropped={len(dropped)} "
          f"(equity={eq_dropped}, jumpy={jumpy_dropped}, other={len(dropped)-eq_dropped-jumpy_dropped})")
    print("sample kept:", ", ".join(kept[:12]))
    for tgt in ("BTC_USDT_SWAP", "ETH_USDT_SWAP", "SOL_USDT_SWAP", "MRVL_USDT_SWAP", "H_USDT_SWAP"):
        print(f"  {tgt:22s} -> {'KEPT' if tgt in kept else 'dropped: ' + dict(dropped).get(tgt, '?')}")


if __name__ == "__main__":
    main()
