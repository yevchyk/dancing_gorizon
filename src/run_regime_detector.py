"""Per-hour regime detector: does the PREVIOUS hour's market state (breadth,
volatility) + the engine's own recent behaviour predict whether THIS hour will
be profitable for the engine? If yes, skip the bad hours.

Strictly causal: features come from the prior hour only. Verified by a temporal
split (fit on early hours, test on late) and by checking it does not just skip
everything (how many good hours wrongly dropped).

Usage:
  python -m src.run_regime_detector
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
FLOOR, OPP, AGREE, EXCL = 0.82, 0.30, 2, {"down_1h", "down_2h"}
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
            idx = np.where(m)[0]
            for i in idx:
                key = (g["symbol"].iat[i], g["time"].iat[i], sgn)
                raw.setdefault(key, []).append((lab, p[i] - opp[i], sgn * ret[i] - COST))
    rows = []
    for (sym, t, sgn), lst in raw.items():
        if len(lst) < AGREE:
            continue
        lab, spread, pnl = max(lst, key=lambda x: x[1])
        rows.append({"time": t, "side": sgn, "spread": spread, "pnl": pnl})
    return pd.DataFrame(rows)


def main():
    g = pd.read_parquet(C.DATASETS_DIR / "sim_grid.parquet")
    g["time"] = pd.to_datetime(g["time"], utc=True)
    g["hr"] = g["time"].dt.floor("h")

    # market features per hour from the grid itself
    mkt = []
    for hr, gh in g.groupby("hr"):
        rets = gh.groupby("symbol")["entry_price"].agg(lambda s: s.iloc[-1] / s.iloc[0] - 1.0)
        btc = rets.get("BTC_USDT_SWAP", np.nan)
        mkt.append({"hr": hr, "breadth": (rets > 0).mean(), "disp": rets.std(),
                    "btc_ret": btc, "n_coins": len(rets)})
    M = pd.DataFrame(mkt).set_index("hr")

    # engine outcome per hour
    sig = engine_signals(g)
    sig["hr"] = sig["time"].dt.floor("h")
    eng = sig.groupby("hr").agg(n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()),
                                pnl=("pnl", "mean"), long_frac=("side", lambda x: (x > 0).mean()),
                                spread=("spread", "mean"))
    df = eng.join(M, how="inner")
    # LAG market+engine features by one hour (decision uses prior hour)
    for c in ["breadth", "disp", "btc_ret", "win", "long_frac", "spread", "n"]:
        df[f"prev_{c}"] = df[c].shift(1)
    df = df.dropna(subset=["prev_breadth", "prev_win"])
    df = df[df.n >= 5]
    print(f"usable hours: {len(df)}\n")

    print("=== correlation of PREV-hour features with THIS-hour engine pnl ===")
    for c in ["prev_breadth", "prev_disp", "prev_btc_ret", "prev_win", "prev_long_frac", "prev_spread"]:
        print(f"  {c:<16} corr={df['pnl'].corr(df[c]):+.2f}")

    # temporal split: fit threshold on first 60%, test on last 40%
    df = df.sort_index()
    cut = int(len(df) * 0.6)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    # simple rule: skip hour if prev_breadth below the train median AND prev_disp below median
    bth = tr["prev_breadth"].median()
    print(f"\n=== RULE: trade only if prev_breadth >= {bth:.2f} (fit on train) ===")
    for name, d in (("TRAIN", tr), ("TEST", te)):
        keep = d[d.prev_breadth >= bth]
        skip = d[d.prev_breadth < bth]
        allp = (d.pnl * d.n).sum() / d.n.sum() * 100
        keepp = (keep.pnl * keep.n).sum() / keep.n.sum() * 100 if len(keep) else 0
        print(f"  {name}: all_hours avg_pnl={allp:+.4f}%  ->  filtered avg_pnl={keepp:+.4f}%  "
              f"(kept {len(keep)}/{len(d)} hrs, skipped {len(skip)})")
        if len(skip):
            print(f"       skipped hours avg_pnl={(skip.pnl*skip.n).sum()/skip.n.sum()*100:+.4f}% "
                  f"(want this NEGATIVE), good hours wrongly skipped: {(skip.pnl>0).sum()}")


if __name__ == "__main__":
    main()
