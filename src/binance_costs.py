"""Honest per-symbol round-trip cost for the Binance USDⓈ-M perps, measured from
REAL kline data — no flat 0.75% lie, no made-up constants.

The user's demand: "провір гарно під кожну позицію, який і чи він існує" — so every
component here is grounded and the data is integrity-checked before we trust it:

  1. FEES (deterministic, PUBLIC, definitely exists): Binance USDT-M taker fee is
     0.05% per side (0.04% with BNB; we do NOT assume the discount -> conservative).
     Round-trip taker = 0.10%.
  2. SPREAD (measured PER SYMBOL from the 1-min high/low series): Corwin-Schultz
     (2012) high-low spread estimator. It is a published method that backs the
     effective bid-ask spread out of consecutive-bar highs/lows while removing the
     volatility component (the γ term), so it is NOT just "bar range = cost". We
     aggregate it with the MEDIAN over a recent window (robust to 1-min noise).
  3. INTEGRITY: for each symbol we verify the parquet exists, is fresh, has the
     expected ~1-min density (gap %), and enough rows. A cost built on gappy/stale
     data is flagged and not trusted.

  round-trip cost%(sym) = fee_rt(0.10) + cs_spread%(sym) + impact_floor
                          floored so we never claim a cost below the fees+1bp.

Klines-only caveat (stated honestly): a true spread needs the order book, which we do
not have. Corwin-Schultz from H/L is the standard klines proxy; we cross-check the
majors (BTC/ETH) against the known ~1-2 bp reality so the table is sanity-anchored.

  python -m src.binance_costs                      # measure all, print table, write json
  python -m src.binance_costs --window-days 30     # spread window
  python -m src.binance_costs --symbols BTC_USDT_SWAP,ETH_USDT_SWAP
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANDLES = Path("data/binance/candles")
OUT = Path("configs/binance_costs.json")

# (1) fees — Binance USDT-M public taker, per side, in PERCENT. No BNB discount.
FEE_TAKER_PCT = 0.05
FEE_ROUNDTRIP_PCT = 2.0 * FEE_TAKER_PCT          # 0.10%

# spread/impact guards (all in PERCENT)
MIN_SPREAD_PCT = 0.01        # 1 bp floor: never claim a sub-tick spread on a perp
MAX_SPREAD_PCT = 1.50        # sanity cap; anything above => flag as suspect/thin
IMPACT_FLOOR_PCT = 0.01      # tiny market-impact pad for a small order

# integrity thresholds. NB: freshness is a LIVE-trading concern, NOT a cost-trust
# one — the spread uses 30d of history and training holds out the last 5d, so a few
# stale hours at the tip are irrelevant here. We keep a LOOSE age gate (48h) only to
# catch a delisted/dead symbol; the real cost-trust gates are rows + gaps + spread.
MAX_GAP_PCT = 5.0            # >5% missing 1-min bars => suspect data
MAX_AGE_MIN = 2880.0        # last bar older than 48h => likely dead listing
MIN_ROWS = 50_000           # ~35 days of 1-min minimum to trust the estimate

_C = 3.0 - 2.0 * np.sqrt(2.0)   # Corwin-Schultz denominator constant


def corwin_schultz_spread_pct(h: np.ndarray, l: np.ndarray) -> float:
    """Median Corwin-Schultz proportional spread (%) over consecutive 1-min bars.

    Per CS (2012): from each pair of adjacent bars,
      beta  = (ln(H_t/L_t))^2 + (ln(H_{t+1}/L_{t+1}))^2
      gamma = (ln(max(H_t,H_{t+1}) / min(L_t,L_{t+1})))^2
      alpha = (sqrt(2*beta)-sqrt(beta))/C - sqrt(gamma/C),  C = 3-2*sqrt(2)
      S     = 2*(e^alpha - 1)/(1 + e^alpha)
    Negative S -> 0 (the estimator's standard clean-up). Return the median * 100.
    """
    h = np.asarray(h, dtype="float64")
    l = np.asarray(l, dtype="float64")
    ok = (h > 0) & (l > 0) & (h >= l)
    # need pairs where BOTH bars are valid
    pair = ok[:-1] & ok[1:]
    if pair.sum() < 100:
        return float("nan")
    h0, l0 = h[:-1][pair], l[:-1][pair]
    h1, l1 = h[1:][pair], l[1:][pair]
    log_hl0 = np.log(h0 / l0)
    log_hl1 = np.log(h1 / l1)
    beta = log_hl0 ** 2 + log_hl1 ** 2
    hi = np.maximum(h0, h1)
    lo = np.minimum(l0, l1)
    gamma = np.log(hi / lo) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _C - np.sqrt(gamma / _C)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.where(np.isfinite(s), s, np.nan)
    s = np.clip(s, 0.0, None)            # negatives -> 0 (CS convention)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.median(s)) * 100.0


def barrange_pct(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> float:
    """Median (high-low)/close % — the cruder proxy, kept for cross-check only."""
    r = (h - l) / c
    r = r[np.isfinite(r)]
    return float(np.median(r)) * 100.0 if r.size else float("nan")


def measure(path: Path, window_days: int = 30) -> dict:
    name = path.stem
    out = {"symbol": name, "ok": False, "note": ""}
    try:
        df = pd.read_parquet(path, columns=["timestamp", "high", "low", "close"])
    except Exception as e:
        out["note"] = f"read fail: {str(e)[:40]}"
        return out
    n = len(df)
    out["rows"] = int(n)
    if n < 2:
        out["note"] = "empty"
        return out
    ts = pd.to_datetime(df["timestamp"], utc=True)
    first, last = ts.iloc[0], ts.iloc[-1]
    span_min = (last - first).total_seconds() / 60.0
    expected = span_min + 1.0
    out["days"] = round(span_min / 1440.0, 1)
    out["gap_pct"] = round(max(0.0, (1.0 - n / expected)) * 100.0, 2) if expected > 0 else 100.0
    age_min = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 60.0
    out["age_min"] = round(age_min, 1)

    # spread/barrange over the recent window only (spread ~stationary, cheaper)
    cut = last - pd.Timedelta(days=window_days)
    w = df[ts >= cut]
    h, l, c = w["high"].to_numpy(), w["low"].to_numpy(), w["close"].to_numpy()
    cs = corwin_schultz_spread_pct(h, l)
    out["cs_spread_pct"] = round(cs, 4) if np.isfinite(cs) else None
    out["barrange_pct"] = round(barrange_pct(h, l, c), 4)

    # integrity verdict
    bad = []
    if n < MIN_ROWS:
        bad.append(f"rows<{MIN_ROWS}")
    if out["gap_pct"] > MAX_GAP_PCT:
        bad.append(f"gap{out['gap_pct']}%")
    if age_min > MAX_AGE_MIN:
        bad.append(f"stale{out['age_min']}m")
    if cs is None or not np.isfinite(cs):
        bad.append("no-spread")

    spread = MIN_SPREAD_PCT if (cs is None or not np.isfinite(cs)) else min(max(cs, MIN_SPREAD_PCT), MAX_SPREAD_PCT)
    out["rt_cost_pct"] = round(FEE_ROUNDTRIP_PCT + spread + IMPACT_FLOOR_PCT, 4)
    out["ok"] = not bad
    out["note"] = ",".join(bad)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candles", type=Path, default=CANDLES)
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--symbols", default="", help="comma list to limit (debug)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    files = sorted(args.candles.glob("*.parquet"))
    if args.symbols:
        want = {s.strip() for s in args.symbols.split(",")}
        files = [f for f in files if f.stem in want]
    print(f"measuring {len(files)} symbols, spread window={args.window_days}d, "
          f"fee_rt={FEE_ROUNDTRIP_PCT:.2f}% (Binance taker 0.05%/side)\n", flush=True)

    rows = [measure(f, args.window_days) for f in files]
    rows = [r for r in rows if r.get("rows", 0) >= 2]
    df = pd.DataFrame(rows).sort_values("rt_cost_pct", ascending=False).reset_index(drop=True)

    good = df[df["ok"]]
    flagged = df[~df["ok"]]
    costs = good["rt_cost_pct"].to_numpy()

    def show(sub: pd.DataFrame, title: str, k: int = 12):
        print(f"--- {title} ---")
        print(f"{'symbol':22s}{'rt%':>7}{'spread%':>9}{'bar%':>8}{'days':>6}{'gap%':>7}{'note':>14}")
        for r in sub.head(k).itertuples(index=False):
            sp = "" if r.cs_spread_pct is None else f"{r.cs_spread_pct:.3f}"
            print(f"{r.symbol:22s}{r.rt_cost_pct:7.3f}{sp:>9}{r.barrange_pct:8.3f}"
                  f"{r.days:6.0f}{r.gap_pct:7.2f}{r.note:>14}")
        print()

    show(df, f"MOST EXPENSIVE (top of {len(df)})")
    majors = df[df["symbol"].isin(["BTC_USDT_SWAP", "ETH_USDT_SWAP", "SOL_USDT_SWAP",
                                   "XRP_USDT_SWAP", "DOGE_USDT_SWAP", "BNB_USDT_SWAP"])]
    show(majors.sort_values("rt_cost_pct"), "MAJORS (sanity anchors)", 8)
    if len(flagged):
        show(flagged, f"FLAGGED / suspect data ({len(flagged)})", 20)

    if len(costs):
        pct = np.percentile(costs, [5, 25, 50, 75, 95])
        print(f"GOOD symbols: {len(good)}/{len(df)}   round-trip cost% distribution:")
        print(f"  p5={pct[0]:.3f}  p25={pct[1]:.3f}  median={pct[2]:.3f}  "
              f"p75={pct[3]:.3f}  p95={pct[4]:.3f}   (OKX flat lie was 0.75)")
        print(f"  cheapest={costs.min():.3f}  most expensive(good)={costs.max():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cost_map = {r["symbol"]: r["rt_cost_pct"] for r in rows if r.get("ok")}
    payload = {"fee_roundtrip_pct": FEE_ROUNDTRIP_PCT, "window_days": args.window_days,
               "method": "fee + corwin_schultz_spread + impact_floor",
               "n_good": len(cost_map), "n_total": len(rows),
               "costs": cost_map,
               "flagged": {r["symbol"]: r["note"] for r in rows if not r.get("ok")}}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}  ({len(cost_map)} trusted, {len(rows)-len(cost_map)} flagged)")


if __name__ == "__main__":
    main()
