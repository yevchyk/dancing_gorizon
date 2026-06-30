"""Fetch 1 year of Binance USDT-M funding-rate history per symbol and measure it.

Funding is NOT in klines — it is a separate cash transfer every 8h (00/08/16 UTC).
A position held h minutes crosses ~h/480 funding events; the expected ADVERSE
funding over the hold is med|rate| * (h/480). Per the user's decision this is
ALWAYS folded into the label threshold for the Binance year-rebuild, so the model
only flags moves that clear fees + spread + funding ("ignore pointless risk").

Output configs/binance_funding.json:
  {"symbols": {SYM: {"med_abs_pct": ..., "p95_abs_pct": ..., "mean_pct": ..., "n": ...}},
   "universe_med_abs_pct": ...}

  python -m src.binance_funding
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .binance_fetcher import _get, norm_symbol

BASE = "https://fapi.binance.com/fapi/v1/fundingRate"
OUT = Path("configs/binance_funding.json")
SERIES_DIR = Path("data/binance/funding")   # per-symbol (timestamp, rate) parquet —
                                            # needed by the v5 funding_level feature
UNIVERSES = [Path("configs/binance_universe_liquid.json"),
             Path("configs/binance_universe_train_extra.json")]


def _syms(paths) -> list[str]:
    seen: list[str] = []
    for p in paths:
        data = json.loads(p.read_text())
        for s in (data.get("symbols", data) if isinstance(data, dict) else data):
            if s not in seen:
                seen.append(s)
    return seen


def fetch_rates(sym: str, days: int = 365):
    start = int((time.time() - days * 86400) * 1000)
    times: list[int] = []
    rates: list[float] = []
    cur = start
    while True:
        data = _get(f"{BASE}?symbol={sym}&startTime={cur}&limit=1000")
        if not data:
            break
        times += [int(d["fundingTime"]) for d in data]
        rates += [float(d["fundingRate"]) for d in data]
        last = int(data[-1]["fundingTime"])
        if len(data) < 1000:
            break
        cur = last + 1
    return np.asarray(times, dtype="int64"), np.asarray(rates, dtype="float64")


def main() -> None:
    syms = _syms(UNIVERSES)
    # only symbols we actually hold candles for; also drops any non-ascii meme
    # listing (e.g. Binance's chinese-named perp) that breaks URLs + cp1251 logs
    have = {p.stem for p in Path("data/binance/candles").glob("*.parquet")}
    before = len(syms)
    syms = [s for s in syms if s.isascii() and norm_symbol(s) in have]
    if len(syms) != before:
        print(f"skipped {before - len(syms)} symbols without local candles / ascii name", flush=True)
    print(f"funding history: {len(syms)} symbols x 365d (~1095 events each)", flush=True)
    out: dict[str, dict] = {}
    t0 = time.time()
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(syms, 1):
        try:
            ts, r = fetch_rates(s)
        except Exception as e:
            print(f"  {i}/{len(syms)} {s}: FAIL {str(e)[:60]}", flush=True)
            continue
        if r.size == 0:
            print(f"  {i}/{len(syms)} {s}: no data", flush=True)
            continue
        import pandas as pd
        pd.DataFrame({"timestamp": pd.to_datetime(ts, unit="ms", utc=True),
                      "rate": r}).to_parquet(SERIES_DIR / f"{norm_symbol(s)}.parquet", index=False)
        a = np.abs(r) * 100.0  # -> percent per 8h event
        out[norm_symbol(s)] = {
            "med_abs_pct": round(float(np.median(a)), 5),
            "p95_abs_pct": round(float(np.percentile(a, 95)), 5),
            "mean_pct": round(float(r.mean() * 100.0), 5),
            "n": int(r.size),
        }
        if i % 25 == 0 or i == len(syms):
            print(f"  {i}/{len(syms)} done ({(time.time()-t0)/60:.1f}m)", flush=True)

    med_all = round(float(np.median([v["med_abs_pct"] for v in out.values()])), 5) if out else None
    p95s = sorted(((k, v["p95_abs_pct"]) for k, v in out.items()), key=lambda kv: -kv[1])[:10]
    print(f"\nuniverse median |8h rate| = {med_all}%  (typical exchange default is 0.01%)")
    print("hottest p95 names:", ", ".join(f"{k.split('_')[0]}={v}" for k, v in p95s))
    OUT.write_text(json.dumps({"symbols": out, "universe_med_abs_pct": med_all,
                               "method": "median |rate| per 8h event; adverse funding over "
                                         "hold h_min = med_abs_pct * h/480"}, indent=2))
    print(f"wrote {OUT}  ({len(out)} symbols)")


if __name__ == "__main__":
    main()
