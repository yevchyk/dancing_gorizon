"""MAXIMALLY-CORRECT validation of the conviction ladder (L1 model -> L2 herd gate
-> L3 clean-knife wick filter).

The leak we are killing: previously the gate threshold (togetherness p80) and the
wick filter (median) were chosen by looking at the SAME holdout we then scored.
Here every threshold is CAUSAL: at day d it is computed ONLY from anchors in days
< d (expanding past). Metrics are reported on a VALIDATION span (after a warmup)
that never participated in choosing any threshold. The per-symbol model is already
out-of-sample (trained <= 05-10, holdout 05-11..06-04).

  python -m src.run_ladder_validate --horizon 8m --warmup 7
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


def market_state(anchor_ns: np.ndarray) -> pd.DataFrame:
    """Fully vectorized per-anchor market state (no look-ahead, no python loops)."""
    syms = json.loads(SYMS.read_text(encoding="utf-8"))
    syms = syms.get("symbols", syms) if isinstance(syms, dict) else syms
    store = CandleStore(STORE)
    n = len(anchor_ns)
    r60 = np.full((n, len(syms)), np.nan)
    wick = np.full((n, len(syms)), np.nan)
    btc_r15 = np.full(n, np.nan)
    for j, sym in enumerate(syms):
        c = store.load(sym)
        if c is None or c.empty:
            continue
        c = c.sort_index(); ts = index_to_ns(c.index)
        close = c["close"].to_numpy("float64"); high = c["high"].to_numpy("float64")
        low = c["low"].to_numpy("float64"); op = c["open"].to_numpy("float64")
        ei = np.searchsorted(ts, anchor_ns, side="right") - 1
        ok = ei >= 60
        idx = ei[ok]
        r60[ok, j] = close[idx] / close[idx - 60] - 1
        rng = high[idx] - low[idx]
        lw = np.minimum(op[idx], close[idx]) - low[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            wick[ok, j] = np.where(rng > 0, lw / rng, np.nan)
        if sym.startswith("BTC"):
            ok15 = ei >= 15
            btc_r15[ok15] = (close[ei[ok15]] / close[ei[ok15] - 15] - 1) * 100
    nv = (~np.isnan(r60)).sum(1).clip(min=1)
    breadth = np.nansum((r60 > 0), 1) / nv
    return pd.DataFrame({
        "anchor_ns": anchor_ns,
        "breadth": breadth,
        "togetherness": np.maximum(breadth, 1 - breadth),
        "wick_frac": np.nanmean(wick, 1),
        "btc_r15": btc_r15,
    })


def report(name, x, ndays, N):
    if len(x) == 0:
        print(f"  {name:<34} (no trades)"); return
    dpd = x.groupby("day")["pnl"].sum() * N
    print(f"  {name:<34} n={len(x):<5} n/day={len(x)/ndays:>4.1f} days={x['day'].nunique():>2} "
          f"win={x['winb'].mean():.3f} avg%={x['pnl'].mean()*100:+.4f} "
          f"$/day={dpd.sum()/ndays:+.2f} posos={dpd.min():+.2f} green={(dpd>0).mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bluechip_short")
    ap.add_argument("--experiment", default="fast_bluechip")
    ap.add_argument("--horizon", default="8m")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=7, help="days used only to seed causal thresholds")
    ap.add_argument("--gate-pct", type=float, default=80.0)
    ap.add_argument("--notional", type=float, default=30.0)
    args = ap.parse_args()
    h = args.horizon; N = args.notional

    path = f"outputs/analysis/{args.experiment}/{args.tag}/holdout_scores.parquet"
    s = pd.read_parquet(path, columns=["symbol", "anchor_time", "day", f"p_up_{h}", f"real_ret_{h}"])
    s = s.rename(columns={f"p_up_{h}": "p", f"real_ret_{h}": "ret"})
    days = sorted(s["day"].unique())
    print(f"file={path}\nhorizon={h}  holdout {days[0]}..{days[-1]} ({len(days)}d)  "
          f"warmup={args.warmup}d  validation={len(days)-args.warmup}d")

    # selection = top-K/day (intraday ranking caveat noted; the LEAK we fix is thresholds)
    s["rk"] = s.groupby("day")["p"].rank(ascending=False, method="first")
    sel = s[s.rk <= args.k].copy()
    sel["pnl"] = sel["ret"] - COST
    sel["winb"] = (sel["pnl"] > 0).astype(int)

    # market state at ALL unique holdout anchors (reference distribution for causal thresholds)
    uniq = s["anchor_time"].drop_duplicates().reset_index(drop=True)
    uns = pd.DatetimeIndex(pd.to_datetime(uniq, utc=True)).as_unit("ns").asi8
    ms = market_state(uns)
    ms["anchor_time"] = pd.to_datetime(uniq.values, utc=True)
    ms["day"] = ms["anchor_time"].dt.strftime("%Y-%m-%d")
    sel = sel.merge(ms[["anchor_time", "togetherness", "wick_frac", "btc_r15"]],
                    on="anchor_time", how="left")

    # CAUSAL thresholds: at day d use only anchors with day < d (expanding past)
    val_days = days[args.warmup:]
    tog_thr = {}; wick_thr = {}
    for d in val_days:
        past = ms[ms["day"] < d]
        tog_thr[d] = np.nanpercentile(past["togetherness"], args.gate_pct)
        wick_thr[d] = np.nanmedian(past["wick_frac"])
    sel_v = sel[sel["day"].isin(val_days)].copy()
    sel_v["tog_thr"] = sel_v["day"].map(tog_thr)
    sel_v["wick_thr"] = sel_v["day"].map(wick_thr)
    gate = sel_v["togetherness"] >= sel_v["tog_thr"]
    clean = gate & (sel_v["wick_frac"] <= sel_v["wick_thr"])
    knife = clean & (sel_v["btc_r15"] <= -0.3)
    nd = len(val_days)

    print(f"\n=== CAUSAL ladder on VALIDATION span ({val_days[0]}..{val_days[-1]}, {nd}d) ===")
    print("  thresholds derived from PAST ONLY (expanding); model already OOS")
    report("L1 baseline (all top-50)", sel_v, nd, N)
    report("L2 + causal herd gate", sel_v[gate], nd, N)
    report("L3 + causal gate + clean wick", sel_v[clean], nd, N)
    report("L3b + clean knife (wick & r15)", sel_v[knife], nd, N)

    # honesty: compare to the IN-SAMPLE (snooped full-quantile) version on the SAME val span
    tog_is = sel_v["togetherness"].quantile(args.gate_pct / 100)
    wick_is = sel_v["wick_frac"].median()
    g_is = sel_v["togetherness"] >= tog_is
    c_is = g_is & (sel_v["wick_frac"] <= wick_is)
    print(f"\n  [in-sample snooped thresholds, same span — for degradation check]")
    report("  L2 in-sample gate", sel_v[g_is], nd, N)
    report("  L3 in-sample gate+wick", sel_v[c_is], nd, N)

    # per-day on validation span
    print(f"\n=== per-day (validation): L3 causal clean-knife vs baseline ===")
    print(f"{'day':<12}{'base_n':>7}{'base_$':>8}{'L3_n':>6}{'L3_win':>8}{'L3_$':>8}{'tog_thr':>9}{'wick_thr':>9}")
    base_g = sel_v.groupby("day")
    l3 = sel_v[clean]
    for d in val_days:
        b = base_g.get_group(d) if d in base_g.groups else sel_v.iloc[0:0]
        x = l3[l3["day"] == d]
        print(f"{d:<12}{len(b):>7}{b['pnl'].sum()*N:>+8.2f}{len(x):>6}"
              f"{(x['winb'].mean() if len(x) else float('nan')):>8.3f}"
              f"{x['pnl'].sum()*N:>+8.2f}{tog_thr[d]:>9.3f}{wick_thr[d]:>9.3f}")


if __name__ == "__main__":
    main()
