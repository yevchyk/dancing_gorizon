"""Comprehensive engine analysis from engine_stats.parquet:
  1. probability threshold that stays positive in EVERY fold (robust, with slippage)
  2. win-rate gradient by confidence (does it rise - to +)
  3. liquidity effect (thin coins = worse?)
  4. harmful coins to drop
  5. assemble an efficient config (threshold + clean liquid pool) and backtest it

A 'trade' per anchor x horizon: side = argmax(p_up, p_down), conf = that prob;
taken if conf >= threshold. pnl = signed real_ret - fee - slippage.

Usage:
  python -m src.run_engine_analysis --slip 0.05 --pool 60
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0


def make_trades(s: pd.DataFrame, cost: float) -> pd.DataFrame:
    side = np.where(s["p_up"].to_numpy() >= s["p_down"].to_numpy(), 1, -1)
    conf = np.maximum(s["p_up"].to_numpy(), s["p_down"].to_numpy())
    pnl = side * s["real_ret"].to_numpy() - cost
    return pd.DataFrame({"fold": s["fold"].to_numpy(), "symbol": s["symbol"].to_numpy(),
                         "day": s["day"].to_numpy(), "horizon": s["horizon"].to_numpy(),
                         "side": side, "conf": conf, "pnl": pnl, "won": (pnl > 0).astype(int)})


def liquidity_rank(store: CandleStore, symbols) -> dict:
    liq = {}
    for sym in symbols:
        c = store.load(sym)
        if c is None or c.empty:
            continue
        tail = c.iloc[-43200:]   # ~last 30d of 1m-equivalent
        liq[sym] = float((tail["close"] * tail["volume"]).sum())
    return liq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip", type=float, default=0.05, help="extra slippage %% per round-trip")
    ap.add_argument("--pool", type=int, default=60, help="top-N liquid coins for final config")
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0
    print(f"cost/trade = fee+slip = {cost*100:.3f}%\n")

    s = pd.read_parquet(C.OUTPUTS_DIR / "analysis" / "engine_stats.parquet")
    trades = make_trades(s, cost)
    nfolds = trades["fold"].nunique()

    # 1. threshold robustness (positive in EVERY fold)
    print("=== 1. THRESHOLD sweep (avg pnl overall + worst fold) ===")
    best_thr = None
    for thr in [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]:
        g = trades[trades.conf >= thr]
        if len(g) < 50:
            continue
        perfold = g.groupby("fold")["pnl"].mean() * 100
        worst = perfold.min()
        print(f"  conf>={thr:.2f}: n={len(g):>6} win={g.won.mean():.3f} "
              f"avg={g.pnl.mean()*100:+.4f}% worst_fold={worst:+.4f}% "
              f"({'+all folds' if (perfold>0).all() else 'NEG in a fold'})")
        if (perfold > 0).all() and best_thr is None:
            best_thr = thr
    best_thr = best_thr or 0.65
    print(f"  -> robust threshold = {best_thr}")

    # 2. win-rate gradient by confidence decile
    print("\n=== 2. WIN-RATE gradient (confidence decile) ===")
    q = pd.qcut(trades["conf"], 10, labels=False, duplicates="drop")
    grad = trades.assign(q=q).groupby("q").agg(conf=("conf", "mean"),
            win=("won", "mean"), pnl=("pnl", lambda x: x.mean() * 100), n=("pnl", "size"))
    for i, r in grad.iterrows():
        print(f"  d{int(i)}: conf={r.conf:.3f} win={r.win:.3f} pnl={r.pnl:+.3f}% n={int(r.n)}")

    sel = trades[trades.conf >= best_thr]

    # 3. liquidity effect
    print("\n=== 3. LIQUIDITY effect (quartiles, at robust threshold) ===")
    store = CandleStore(C.CANDLES_DIR)
    liq = liquidity_rank(store, sel["symbol"].unique())
    sel = sel.assign(liq=sel["symbol"].map(liq)).dropna(subset=["liq"])
    sel["liq_q"] = pd.qcut(sel["liq"].rank(method="first"), 4, labels=["Q1 thin", "Q2", "Q3", "Q4 liquid"])
    for lab, g in sel.groupby("liq_q", observed=True):
        print(f"  {lab:<10} n={len(g):>5} win={g.won.mean():.3f} avg_pnl={g.pnl.mean()*100:+.4f}%")

    # 4. harmful coins
    print("\n=== 4. HARMFUL coins (>=15 trades, worst avg pnl) ===")
    per = sel.groupby("symbol").agg(n=("pnl", "size"), win=("won", "mean"),
                                    avg=("pnl", lambda x: x.mean() * 100),
                                    tot=("pnl", lambda x: x.sum() * 100))
    harmful = per[(per.n >= 15) & (per.avg < 0)].sort_values("tot")
    for sym, r in harmful.head(15).iterrows():
        print(f"  {sym:<16} n={int(r.n):>4} win={r.win:.2f} avg={r.avg:+.3f}% tot={r.tot:+.2f}%")
    blacklist = set(harmful.index)

    # 5. assemble efficient config: robust threshold + drop harmful + top liquid pool
    print(f"\n=== 5. EFFICIENT CONFIG: conf>={best_thr}, drop {len(blacklist)} harmful, "
          f"top-{args.pool} liquid pool ===")
    top_pool = set(pd.Series(liq).sort_values(ascending=False).head(args.pool).index)
    final = sel[(~sel.symbol.isin(blacklist)) & (sel.symbol.isin(top_pool))]
    if len(final):
        daily = final.groupby("day")["pnl"].mean() * 100
        perfold = final.groupby("fold")["pnl"].mean() * 100
        print(f"  trades={len(final)}  win={final.won.mean():.3f}  "
              f"avg_pnl={final.pnl.mean()*100:+.4f}%")
        print(f"  per-fold pnl: " + " ".join(f"{v:+.3f}%" for v in perfold))
        print(f"  daily: green={int((daily>0).sum())}/{len(daily)}  "
              f"worst={daily.min():+.3f}% mean={daily.mean():+.4f}%")
        print(f"  coins traded: {final.symbol.nunique()}  trades/day~{len(final)/len(daily):.1f}")
        comp = final[final.conf >= 0.70]
        if len(comp):
            print(f"  high-conf subset (>=0.70): n={len(comp)} win={comp.won.mean():.3f} "
                  f"avg={comp.pnl.mean()*100:+.4f}%")

        # strict freshest slice: last 10 days only (what we trust most)
        mx = pd.to_datetime(final["day"]).max()
        cut = (mx - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        last = final[final["day"] > cut]
        if len(last):
            ld = last.groupby("day")["pnl"].mean() * 100
            print(f"\n  [STRICT last-10-days only, > {cut}] "
                  f"n={len(last)} win={last.won.mean():.3f} "
                  f"avg_pnl={last.pnl.mean()*100:+.4f}%  "
                  f"green_days={int((ld>0).sum())}/{len(ld)}")


if __name__ == "__main__":
    main()
