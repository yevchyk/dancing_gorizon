"""Cross-market recon BEFORE any lead-lag test: when does tradfi actually trade?
Crypto is 24/7; OKX tokenized-stock perps have hours/gaps. Naive CCF on stale
tradfi prices = phantom lag. So map ACTIVE minutes (close changed) by hour-of-day
and weekday for both markets, and find the real overlap window.

  python -m src.run_xmkt_recon
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import markets

sys.stdout.reconfigure(encoding="utf-8")


def activity(df: pd.DataFrame, lo, hi):
    df = df[(df.index >= lo) & (df.index <= hi)]
    if df.empty:
        return None
    close = df["close"].to_numpy("float64")
    changed = np.r_[False, np.diff(close) != 0.0]
    g = pd.DataFrame({"hod": df.index.hour, "dow": df.index.dayofweek, "chg": changed})
    return df, g


def main():
    cr = markets.get("bluechip")
    tf = markets.get("tradfi_1m")
    print("crypto store:", cr.store_dir, "| tradfi store:", tf.store_dir)

    btc = cr.load("BTC_USDT_SWAP")
    # pick liquid tradfi proxies
    tf_syms = [s for s in ("QQQ_USDT_SWAP", "SPX_USDT_SWAP", "AAPL_USDT_SWAP",
                           "NVDA_USDT_SWAP", "QQQ", "SPX", "AAPL", "NVDA")
               if (tf.store_dir / f"{s}.parquet").exists()]
    print("tradfi probes found:", tf_syms[:6])

    # common overlap window = last 25 days of intersection
    tf0 = tf.load(tf_syms[0]) if tf_syms else None
    if tf0 is None or btc is None:
        print("MISSING DATA — btc:", btc is not None, "tf:", tf0 is not None)
        return
    lo = max(btc.index.min(), tf0.index.min())
    hi = min(btc.index.max(), tf0.index.max())
    print(f"\nBTC span:   {btc.index.min()} .. {btc.index.max()} ({len(btc):,} rows)")
    print(f"tradfi span:{tf0.index.min()} .. {tf0.index.max()} ({len(tf0):,} rows)")
    print(f"OVERLAP:    {lo} .. {hi}  ({(hi-lo).total_seconds()/86400:.1f}d)")
    lo = max(lo, hi - pd.Timedelta(days=25))

    print("\n=== ACTIVE minutes by HOUR-OF-DAY (UTC), last 25d of overlap ===")
    print("(active = close changed that minute; reveals real trading hours)")
    print(f"{'hod':>4}{'BTC_act':>9}{'BTC_chg%':>9} | {tf_syms[0][:10]:>10}{'_act':>5}{'_chg%':>7}")
    _, gb = activity(btc, lo, hi)
    _, gt = activity(tf0, lo, hi)
    bh = gb.groupby("hod")["chg"].agg(["size", "mean"])
    th = gt.groupby("hod")["chg"].agg(["size", "mean"])
    for h in range(24):
        bs = int(bh.loc[h, "size"]) if h in bh.index else 0
        bm = bh.loc[h, "mean"] if h in bh.index else 0
        ts = int(th.loc[h, "size"]) if h in th.index else 0
        tm = th.loc[h, "mean"] if h in th.index else 0
        print(f"{h:>4}{bs:>9}{bm*100:>8.0f}% | {ts:>13}{tm*100:>6.0f}%")

    print("\n=== ACTIVE minutes by WEEKDAY (0=Mon..6=Sun), last 25d ===")
    bd = gb.groupby("dow")["chg"].mean(); td = gt.groupby("dow")["chg"].mean()
    print(f"{'dow':>4}{'BTC_chg%':>10}{'tradfi_chg%':>13}")
    for d in range(7):
        print(f"{d:>4}{(bd.get(d,0))*100:>9.0f}%{(td.get(d,0))*100:>12.0f}%")

    # tradfi active fraction overall + how many tradfi syms have data
    on_disk = tf.symbols_on_disk()
    print(f"\ntradfi symbols on disk: {len(on_disk)}")
    print(f"tradfi overall active-minute fraction (last25d): {gt['chg'].mean()*100:.1f}%")
    print(f"BTC overall active-minute fraction:              {gb['chg'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
