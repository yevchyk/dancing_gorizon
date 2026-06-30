"""Trailing-3h regime test: does market breadth + BTC volatility over the LAST 3
hours predict the engine's performance in the NEXT hour? (Causal: features from
the past only.) If quiet+red over 3h -> skip the next hour.

Verified by temporal split + checking we don't just skip everything.

Usage:
  python -m src.run_regime3h --win-h 3
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
FLOOR, OPP, EXCL = 0.82, 0.30, {"down_1h", "down_2h"}
COST = FEE + 0.0005


def engine_signals(g):
    raw = {}
    for h in C.HORIZONS:
        lab = h.label
        ex, en = g[f"exit_{lab}"].to_numpy(), g["entry_price"].to_numpy()
        ok = np.isfinite(ex); ret = ex / en - 1.0
        for kind, sgn in (("up", 1), ("down", -1)):
            if f"{kind}_{lab}" in EXCL:
                continue
            p = g[f"p_{kind}_{lab}"].to_numpy()
            opp = (g[f"p_down_{lab}"] if kind == "up" else g[f"p_up_{lab}"]).to_numpy()
            m = ok & (p >= FLOOR) & (opp <= OPP)
            for i in np.where(m)[0]:
                raw.setdefault((g["symbol"].iat[i], g["time"].iat[i], sgn), []).append(
                    (p[i] - opp[i], sgn * ret[i] - COST))
    rows = []
    for (s, t, sgn), lst in raw.items():
        sp, pn = max(lst, key=lambda x: x[0])
        rows.append({"time": t, "pnl": pn})
    return pd.DataFrame(rows).sort_values("time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win-h", type=int, default=3)
    args = ap.parse_args()
    g = pd.read_parquet(C.DATASETS_DIR / "sim_grid.parquet")
    g["time"] = pd.to_datetime(g["time"], utc=True)

    # wide price matrix (time x symbol) on the 5-min grid
    px = g.pivot_table(index="time", columns="symbol", values="entry_price").sort_index()
    btc = px["BTC_USDT_SWAP"]
    sig = engine_signals(g)

    hours = pd.date_range(px.index.min().ceil("h"), px.index.max().floor("h"), freq="h", tz="UTC")
    W = pd.Timedelta(hours=args.win_h)
    rows = []
    for H in hours:
        win = px.loc[(px.index >= H - W) & (px.index < H)]
        if len(win) < 12:
            continue
        rets = win.iloc[-1] / win.iloc[0] - 1.0
        breadth = (rets > 0).mean()
        bvol = btc.loc[(btc.index >= H - W) & (btc.index < H)].pct_change().std() * 100
        fwd = sig[(sig.time >= H) & (sig.time < H + pd.Timedelta(hours=1))]
        if len(fwd) < 5:
            continue
        rows.append({"H": H, "breadth": breadth, "btcvol": bvol,
                     "fwd_n": len(fwd), "fwd_win": (fwd.pnl > 0).mean(),
                     "fwd_pnl": fwd.pnl.mean()})
    df = pd.DataFrame(rows).dropna()
    print(f"hours tested: {len(df)}\n")
    print("=== corr of TRAILING-3h features with NEXT-hour engine pnl ===")
    print(f"  breadth(3h)  corr={df['fwd_pnl'].corr(df['breadth']):+.2f}")
    print(f"  btcvol(3h)   corr={df['fwd_pnl'].corr(df['btcvol']):+.2f}")
    # combined 'bad' flag: quiet AND red over 3h
    df = df.sort_values("H")
    cut = int(len(df) * 0.6)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    bth, vth = tr.breadth.quantile(0.4), tr.btcvol.quantile(0.4)
    print(f"\n=== RULE (fit on train): SKIP next hour if breadth<{bth:.2f} AND btcvol<{vth:.3f} ===")
    for name, d in (("TRAIN", tr), ("TEST", te)):
        bad = (d.breadth < bth) & (d.btcvol < vth)
        keep, skip = d[~bad], d[bad]
        aw = (d.fwd_pnl * d.fwd_n).sum() / d.fwd_n.sum() * 100
        kw = (keep.fwd_pnl * keep.fwd_n).sum() / keep.fwd_n.sum() * 100 if len(keep) else 0
        sw = (skip.fwd_pnl * skip.fwd_n).sum() / skip.fwd_n.sum() * 100 if len(skip) else 0
        print(f"  {name}: all={aw:+.4f}%  filtered={kw:+.4f}%  | skipped {len(skip)}/{len(d)}h "
              f"(their avg={sw:+.4f}%, want NEG)  good-hrs-skipped={int((skip.fwd_pnl>0).sum())}")


if __name__ == "__main__":
    main()
