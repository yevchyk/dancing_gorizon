"""Timestamp-convention check: Binance vs OKX 1m BTC closes must correlate best
at LAG 0 (both stores are supposed to stamp bars with their OPEN time, UTC). An
off-by-one here would silently shift every feature by one bar, so run this after
any refetch and before building a dataset.

  python -m src.binance_okx_align
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _closes(path: str) -> pd.Series:
    df = pd.read_parquet(path, columns=["timestamp", "close"])
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return pd.Series(df["close"].to_numpy(), index=ts).groupby(level=0).last()


def main() -> None:
    b = _closes("data/binance/candles/BTC_USDT_SWAP.parquet")
    o = _closes("data/candles/BTC_USDT_SWAP.parquet")
    end = min(b.index.max(), o.index.max())
    start = end - pd.Timedelta(days=3)
    j = pd.concat([b.loc[start:end].rename("b"), o.loc[start:end].rename("o")],
                  axis=1, join="inner").dropna()
    print(f"common 1m bars over last 3d: {len(j)}  ({j.index.min()} .. {j.index.max()})")
    rb, ro = np.log(j["b"]).diff(), np.log(j["o"]).diff()
    corrs = {k: float(rb.shift(k).corr(ro)) for k in (-2, -1, 0, 1, 2)}
    for k in sorted(corrs):
        print(f"  lag {k:+d}: corr={corrs[k]:.4f}")
    best = max(corrs, key=corrs.get)
    print("VERDICT:", "ALIGNED (lag 0 wins)" if best == 0
          else f"MISALIGNED - best lag {best:+d}, FIX BEFORE TRAINING")


if __name__ == "__main__":
    main()
