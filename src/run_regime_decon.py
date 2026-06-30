"""Deep regime deconstruction.

Layer market-state metrics on top of the engine's trades and ask, honestly:
  1. BASELINE (yardstick): what is our average winrate with NO regime filter?
     (classical = take the engine's top-K/day longs blindly.)
  2. SINGLE METRICS: bucket the SAME trades by each market metric -> winrate lift
     over baseline. A metric that doesn't move winrate is noise.
  3. PAIRS: 3x3 grid of the two strongest metrics -> which COMBO lifts most.

Market metrics (computed real-time from the 120-coin universe, not BTC alone):
  breadth_60   = frac of universe up over last 60m          (direction up/down)
  togetherness = max(breadth, 1-breadth)                    (herd / stress)
  dispersion   = cross-sectional std of 60m returns         (inverse stress)
  btc_ret_60   = BTC 60m return                             (single-asset trend)
  btc_vol_pct  = BTC 30m realized-vol percentile            (instability)

  python -m src.run_regime_decon --tag bluechip_short --horizon 8m --days 10 --k 50
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .trading.timeutil import index_to_ns

sys.stdout.reconfigure(encoding="utf-8")
COST = 0.0012
STORE = C.DATA_DIR / "bluechip" / "candles_1m"
SYMS = C.ROOT / "configs" / "bluechip_symbols.json"


def market_metrics(anchor_ns: np.ndarray) -> pd.DataFrame:
    """Per-anchor market-state from the whole universe."""
    syms = json.loads(SYMS.read_text(encoding="utf-8"))
    syms = syms.get("symbols", syms) if isinstance(syms, dict) else syms
    store = CandleStore(STORE)
    n = len(anchor_ns)
    ret60 = np.full((n, len(syms)), np.nan)
    btc_vol = np.full(n, np.nan); btc_ret = np.full(n, np.nan)
    for j, sym in enumerate(syms):
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index()
        ts = index_to_ns(c.index); close = c["close"].to_numpy("float64")
        ei = np.searchsorted(ts, anchor_ns, side="right") - 1
        ok = ei >= 60
        idx = ei[ok]
        ret60[ok, j] = close[idx] / close[idx - 60] - 1.0
        if sym.startswith("BTC"):
            lg = np.diff(np.log(close), prepend=np.log(close[0]))
            for k in np.where(ei >= 30)[0]:
                e = ei[k]
                btc_vol[k] = lg[e - 30:e + 1].std()
            btc_ret[ok] = ret60[ok, j]
    valid = ~np.isnan(ret60)
    nvalid = valid.sum(axis=1).clip(min=1)
    pos = np.nansum((ret60 > 0).astype(float), axis=1)
    breadth = pos / nvalid
    disp = np.nanstd(ret60, axis=1)
    df = pd.DataFrame({
        "anchor_ns": anchor_ns,
        "breadth_60": breadth,
        "togetherness": np.maximum(breadth, 1 - breadth),
        "dispersion": disp * 100,
        "btc_ret_60": btc_ret * 100,
        "btc_vol_pct": pd.Series(btc_vol).rank(pct=True).to_numpy(),
    })
    return df


def qbucket(x, q=5):
    try:
        return pd.qcut(x, q, labels=False, duplicates="drop")
    except Exception:
        return pd.Series(np.zeros(len(x)), index=x.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bluechip_short")
    ap.add_argument("--experiment", default="fast_bluechip")
    ap.add_argument("--horizon", default="8m")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--notional", type=float, default=30.0)
    args = ap.parse_args()
    h = args.horizon; N = args.notional

    path = f"outputs/analysis/{args.experiment}/{args.tag}/holdout_scores.parquet"
    s = pd.read_parquet(path, columns=["symbol", "anchor_time", "day",
                                       f"p_up_{h}", f"real_ret_{h}"])
    days = sorted(s["day"].unique())[-args.days:]
    s = s[s["day"].isin(days)].copy()
    s = s.rename(columns={f"p_up_{h}": "p", f"real_ret_{h}": "ret"})
    print(f"file={path}\nhorizon={h}  window={days[0]}..{days[-1]} ({len(days)}d)  rows={len(s):,}")

    # --- selection = engine top-K/day longs ---
    s["rk"] = s.groupby("day")["p"].rank(ascending=False, method="first")
    sel = s[s.rk <= args.k].copy()
    sel["pnl"] = sel["ret"] - COST
    sel["winb"] = (sel["pnl"] > 0).astype(int)

    # --- BASELINE yardstick ---
    base_win = sel["winb"].mean()
    base_avg = sel["pnl"].mean() * 100
    dpd = sel.groupby("day")["pnl"].sum() * N
    print("\n" + "=" * 70)
    print(f"BASELINE (yardstick): top-{args.k}/day longs, NO regime filter")
    print(f"  trades={len(sel):,} ({len(sel)//len(days)}/day)  WINRATE={base_win:.4f}"
          f"  avg%={base_avg:+.4f}  $/day={dpd.mean():+.2f}  posos={dpd.min():+.2f}")
    # classical momentum reference: of ALL candidates, buy if up-prob>0.5
    allc = s.copy(); allc["pnl"] = allc["ret"] - COST
    print(f"  [ref] all p>0.5 candidates winrate={(allc[allc.p>0.5]['pnl']>0).mean():.4f}  "
          f"all candidates winrate={(allc['pnl']>0).mean():.4f}")

    # --- market metrics per unique anchor ---
    uniq = sel["anchor_time"].drop_duplicates().reset_index(drop=True)
    uns = pd.DatetimeIndex(pd.to_datetime(uniq, utc=True)).as_unit("ns").asi8
    mm = market_metrics(uns)
    mm["anchor_time"] = pd.to_datetime(uniq.values, utc=True)
    metrics = ["breadth_60", "togetherness", "dispersion", "btc_ret_60", "btc_vol_pct"]
    sel = sel.merge(mm[["anchor_time"] + metrics], on="anchor_time", how="left")

    # --- SINGLE METRIC predictive power ---
    print("\n" + "=" * 70)
    print(f"SINGLE-METRIC lift (trades bucketed into quintiles by metric)")
    print(f"baseline winrate = {base_win:.4f}\n")
    print(f"{'metric':<14}{'Q1':>7}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'Q5':>7}{'spread':>8}{'mono':>6}")
    rank = []
    for mcol in metrics:
        sel["b"] = qbucket(sel[mcol])
        wins = sel.groupby("b")["winb"].mean()
        if len(wins) < 2:
            continue
        row = [wins.get(i, np.nan) for i in range(5)]
        spread = wins.iloc[-1] - wins.iloc[0]
        mono = np.all(np.diff(wins.values) > 0) or np.all(np.diff(wins.values) < 0)
        rank.append((mcol, spread, row, mono))
        cells = "".join(f"{(v if not np.isnan(v) else 0):>7.3f}" for v in row)
        print(f"{mcol:<14}{cells}{spread:>+8.3f}{('yes' if mono else ''):>6}")
    rank.sort(key=lambda r: -abs(r[1]))
    print(f"\n-> strongest single metrics by |spread|: "
          f"{', '.join(f'{m}({sp:+.3f})' for m,sp,_,_ in rank[:3])}")

    # --- GATE PAYOFF: baseline vs stress-gated ---
    print("\n" + "=" * 70)
    print("GATE PAYOFF: does filtering to stress windows beat the baseline?")
    tog_hi = sel["togetherness"].quantile(0.80)
    brd_lo = sel["breadth_60"].quantile(0.20)

    def report(name, mask):
        g = sel[mask]
        if len(g) == 0:
            print(f"  {name:<26} (no trades)"); return
        dpd_ = g.groupby("day")["pnl"].sum() * N
        print(f"  {name:<26} n={len(g):<5} n/day={len(g)/len(days):>4.1f} "
              f"days={g['day'].nunique():>2} win={g['winb'].mean():.3f} "
              f"avg%={g['pnl'].mean()*100:+.4f} $/day={dpd_.sum()/len(days):+.2f} "
              f"posos={dpd_.min():+.2f} green={ (dpd_>0).mean():.2f}")

    report("baseline (all top-50)", np.ones(len(sel), bool))
    report("stress: togeth>=p80", sel["togetherness"] >= tog_hi)
    report("stress: breadth<=p20", sel["breadth_60"] <= brd_lo)
    report("stress: togeth&breadth", (sel["togetherness"] >= tog_hi) & (sel["breadth_60"] <= brd_lo))
    report("calm: togeth<p80", sel["togetherness"] < tog_hi)

    # --- detail for top metric ---
    top_m = rank[0][0]
    sel["b"] = qbucket(sel[top_m])
    print(f"\n--- detail: {top_m} quintiles ---")
    print(f"{'bucket':<8}{'range':>20}{'n':>7}{'n/day':>7}{'win':>8}{'avg%':>9}{'lift':>8}")
    for b, g in sel.groupby("b"):
        lo, hi = g[top_m].min(), g[top_m].max()
        pnl = g["pnl"]
        print(f"Q{int(b)+1:<7}{f'[{lo:.3f},{hi:.3f}]':>20}{len(g):>7}{len(g)//len(days):>7}"
              f"{g['winb'].mean():>8.3f}{pnl.mean()*100:>+9.4f}{g['winb'].mean()-base_win:>+8.3f}")

    # --- BEST PAIR: 3x3 grid of top-2 metrics ---
    if len(rank) >= 2:
        m1, m2 = rank[0][0], rank[1][0]
        sel["b1"] = qbucket(sel[m1], 3); sel["b2"] = qbucket(sel[m2], 3)
        print(f"\n" + "=" * 70)
        print(f"BEST PAIR grid: rows={m1} (low->high), cols={m2} (low->high)")
        print(f"cell = winrate (n) ; baseline={base_win:.3f}")
        grid_w = sel.pivot_table(index="b1", columns="b2", values="winb", aggfunc="mean")
        grid_n = sel.pivot_table(index="b1", columns="b2", values="winb", aggfunc="size")
        hdr = "       " + "".join(f"{m2[:8]+'_'+str(c):>16}" for c in grid_w.columns)
        print(hdr)
        best = (-1, None)
        for r in grid_w.index:
            cells = ""
            for c in grid_w.columns:
                w = grid_w.loc[r, c]; nn = grid_n.loc[r, c]
                if pd.isna(w) or pd.isna(nn):
                    cells += f"{'--':>16}"; continue
                nn = int(nn)
                cells += f"{f'{w:.3f}({nn})':>16}"
                if nn >= 30 and w > best[0]:
                    best = (w, (r, c, nn))
            print(f"{m1[:6]+str(r):<7}{cells}")
        if best[1]:
            r, c, nn = best[1]
            print(f"\n-> best cell: {m1}=Q{r+1} & {m2}=Q{c+1}  winrate={best[0]:.3f} "
                  f"(n={nn}, lift={best[0]-base_win:+.3f} over baseline)")

    # --- per-day: winrate vs regime ---
    print(f"\n" + "=" * 70)
    print(f"PER-DAY: engine winrate vs market regime")
    print(f"{'day':<12}{'win':>7}{'$day':>9}{'breadth':>9}{'togeth':>8}{'disp':>7}{'btcret':>8}{'btcvol%':>9}")
    dd = sel.groupby("day")
    for d, g in dd:
        print(f"{d:<12}{g['winb'].mean():>7.3f}{g['pnl'].sum()*N:>+9.2f}"
              f"{g['breadth_60'].mean():>9.3f}{g['togetherness'].mean():>8.3f}"
              f"{g['dispersion'].mean():>7.3f}{g['btc_ret_60'].mean():>+8.3f}{g['btc_vol_pct'].mean():>9.3f}")


if __name__ == "__main__":
    main()
