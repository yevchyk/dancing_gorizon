"""Does crypto stress (24/7) LEAD tradfi? Maximally-correct lead-lag test.

Build market-aggregate 5m log-return series for crypto (120 bluechip) and tradfi
(69 OKX stock perps), align, then:
  1. CCF of returns:  corr(crypto[t], tradfi[t+lag])  -> peak at +lag = crypto leads
  2. CCF of |returns| (vol spillover) -- stress propagates clearer than direction
  3. Lead-lag asymmetry score: corr(C[t],T[t+k]) - corr(C[t+k],T[t]) for k>0
  4. Event study: avg tradfi cum-return around sharp crypto-down 5m moves
Split ALL by US cash hours (13:30-20:00 UTC) vs off-hours (perp-only).

  python -m src.run_xmkt_leadlag --bar 5 --maxlag 60
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import markets

sys.stdout.reconfigure(encoding="utf-8")


def market_ret(store, symbols, bar, lo, hi):
    cols = {}
    for s in symbols:
        df = store.load(s)
        if df is None or df.empty:
            continue
        c = df["close"]
        c = c[(c.index >= lo) & (c.index <= hi)]
        if len(c) < 100:
            continue
        r = np.log(c.resample(f"{bar}min").last()).diff()
        cols[s] = r
    M = pd.DataFrame(cols)
    return M.median(axis=1), M  # robust market move + matrix


def ccf(a, b, maxlag, step):
    out = {}
    for lag in range(-maxlag, maxlag + 1, step):
        bb = b.shift(-lag // step) if False else b.shift(-(lag // step))
        j = pd.concat([a, bb], axis=1).dropna()
        out[lag] = j.iloc[:, 0].corr(j.iloc[:, 1]) if len(j) > 30 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", type=int, default=5, help="bar minutes")
    ap.add_argument("--maxlag", type=int, default=60, help="max lag minutes")
    ap.add_argument("--days", type=int, default=91)
    args = ap.parse_args()
    bar = args.bar; step = bar

    cr = markets.get("bluechip"); tf = markets.get("tradfi_1m")
    cr_syms = cr.declared_symbols() or cr.symbols_on_disk()
    tf_syms = tf.symbols_on_disk()
    btc = cr.load("BTC_USDT_SWAP")
    # use a LIQUID, LONG-history tradfi ref for the window (not an alphabetical newbie)
    ref = next((r for r in ("QQQ_USDT_SWAP", "SPX_USDT_SWAP", "AAPL_USDT_SWAP") if r in tf_syms), tf_syms[0])
    tfr = tf.load(ref)
    hi = min(btc.index.max(), tfr.index.max())
    lo = max(btc.index.min(), tfr.index.min(), hi - pd.Timedelta(days=args.days))
    print(f"tradfi window ref={ref}")
    print(f"overlap window {lo} .. {hi}  bar={bar}m maxlag={args.maxlag}m")
    print(f"crypto syms={len(cr_syms)} tradfi syms={len(tf_syms)}")

    C, _ = market_ret(cr, cr_syms, bar, lo, hi)
    T, _ = market_ret(tf, tf_syms, bar, lo, hi)
    J = pd.concat([C.rename("crypto"), T.rename("tradfi")], axis=1).dropna()
    print(f"aligned {bar}m bars: {len(J)}  "
          f"contemporaneous corr(ret)={J['crypto'].corr(J['tradfi']):+.3f}  "
          f"corr(|ret|)={J['crypto'].abs().corr(J['tradfi'].abs()):+.3f}")

    def block(label, j):
        if len(j) < 100:
            print(f"\n[{label}] too few bars ({len(j)})"); return
        cc = ccf(j["crypto"], j["tradfi"], args.maxlag, step)
        ca = ccf(j["crypto"].abs(), j["tradfi"].abs(), args.maxlag, step)
        print(f"\n=== {label}  (n={len(j)}) ===")
        print("  lag(min)   corr(ret)   corr(|ret|)   [+lag => crypto LEADS tradfi]")
        for lag in range(-args.maxlag, args.maxlag + 1, step):
            star = "  <-- lag0" if lag == 0 else ""
            print(f"  {lag:>+6}     {cc[lag]:>+7.3f}      {ca[lag]:>+7.3f}{star}")
        # asymmetry score (k>0): positive => crypto leads
        ks = [l for l in cc if l > 0]
        asym_r = np.nanmean([cc[k] - cc[-k] for k in ks])
        asym_a = np.nanmean([ca[k] - ca[-k] for k in ks])
        argr = max(cc, key=lambda l: (cc[l] if not np.isnan(cc[l]) else -9))
        arga = max(ca, key=lambda l: (ca[l] if not np.isnan(ca[l]) else -9))
        print(f"  --> argmax corr(ret) at lag={argr:+d}m ; corr(|ret|) at lag={arga:+d}m")
        print(f"  --> lead-lag asymmetry (>0=crypto leads): ret={asym_r:+.3f}  vol={asym_a:+.3f}")

    block("ALL hours (24/7)", J)
    us = J[(J.index.hour * 60 + J.index.minute >= 13 * 60 + 30) &
           (J.index.hour * 60 + J.index.minute < 20 * 60) & (J.index.dayofweek < 5)]
    off = J[~J.index.isin(us.index)]
    block("US cash hours (13:30-20:00 UTC, Mon-Fri)", us)
    block("OFF-hours (perp-only)", off)

    # --- event study: sharp crypto-down bars -> tradfi forward path ---
    thr = J["crypto"].quantile(0.01)
    ev = J.index[J["crypto"] <= thr]
    pre, post = 60 // bar, 120 // bar
    print(f"\n=== EVENT STUDY: {len(ev)} sharp crypto-down {bar}m bars (<= {thr*100:.2f}%) ===")
    print(f"avg cum-return (%) around event; t=0 is the crypto-down bar")
    paths_c, paths_t = [], []
    idx = J.index
    pos = {ts: i for i, ts in enumerate(idx)}
    arr_c = J["crypto"].to_numpy(); arr_t = J["tradfi"].to_numpy()
    for ts in ev:
        i = pos[ts]
        if i - pre < 0 or i + post >= len(idx):
            continue
        paths_c.append(arr_c[i - pre:i + post + 1])
        paths_t.append(arr_t[i - pre:i + post + 1])
    if paths_c:
        mc = np.nanmean(np.vstack(paths_c), axis=0).cumsum() * 100
        mt = np.nanmean(np.vstack(paths_t), axis=0).cumsum() * 100
        mc -= mc[pre]; mt -= mt[pre]  # zero at t=0
        print(f"{'t(min)':>7}{'crypto_cum%':>13}{'tradfi_cum%':>13}")
        for k in range(-pre, post + 1):
            tag = "  <- event" if k == 0 else ""
            print(f"{k*bar:>+7}{mc[k+pre]:>+13.3f}{mt[k+pre]:>+13.3f}{tag}")
        print("\nIf tradfi keeps falling AFTER t=0 while it was flat before -> crypto leads.")


if __name__ == "__main__":
    main()
