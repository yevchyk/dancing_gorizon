"""Event-driven equity simulation over the 5-min sim grid (the 'classic' run):
start with $100, $10 per trade, scan every 5 min, trust engine picks, positions
close at their horizon (fixed-horizon close), capital + concurrency enforced,
equity compounds. Reports the final balance.

Usage:
  python -m src.run_sim --equity 100 --size 10 --global-trust 0.0 --top 3
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from .trading.timeutil import anchors_to_ns

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
HMIN = {h.label: h.minutes for h in C.HORIZONS}


def build_candidates(g: pd.DataFrame, weights: dict, floor: float, gthr: float,
                     cost: float, per_model_thr: dict | None = None) -> pd.DataFrame:
    out = []
    for h in C.HORIZONS:
        lab = h.label
        for kind, side in (("up", 1), ("down", -1)):
            name = f"{kind}_{lab}"
            w = weights.get(name, 0.0)
            if w <= 0 or w < gthr:
                continue
            mfloor = floor
            if per_model_thr is not None:
                t = per_model_thr.get(name)
                if t is None:
                    continue
                mfloor = t
            prob = (g[f"p_up_{lab}"] if kind == "up" else g[f"p_down_{lab}"]).to_numpy()
            opp = (g[f"p_down_{lab}"] if kind == "up" else g[f"p_up_{lab}"]).to_numpy()
            mfe, mae = g[f"mfe_{lab}"].to_numpy(), g[f"mae_{lab}"].to_numpy()
            fav = mfe if side == 1 else -mae
            adv = np.abs(mae) if side == 1 else mfe
            rr = np.clip(fav / (np.abs(adv) + 1e-4), 0, 5)
            ex = g[f"exit_{lab}"].to_numpy()
            d = pd.DataFrame({
                "symbol": g["symbol"].to_numpy(), "time": g["time"].to_numpy(),
                "model": f"{kind}_{lab}", "side": side, "horizon_min": h.minutes,
                "prob": prob, "opp": opp, "spread": prob - opp, "rr": rr,
                "score": (prob - opp) * w,   # rank by directional spread (research win)
                "entry": g["entry_price"].to_numpy(), "exit": ex})
            d = d[(d.prob >= mfloor) & np.isfinite(d.exit)]
            out.append(d)
    c = pd.concat(out, ignore_index=True)
    c["pnl_pct"] = c.side * (c.exit / c.entry - 1.0) - cost
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, default=100.0)
    ap.add_argument("--size", type=float, default=10.0)
    ap.add_argument("--floor", type=float, default=0.60)
    ap.add_argument("--global-trust", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=3, help="max new trades per 5-min scan")
    ap.add_argument("--cooldown", type=int, default=C.COOLDOWN_MIN)
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--no-trust", action="store_true", help="all models equal (ignore trust)")
    ap.add_argument("--exclude", default="", help="comma list of models to drop")
    ap.add_argument("--per-model-thr", action="store_true",
                    help="use models/prob_thresholds.json per-model activation")
    ap.add_argument("--after", default="", help="only sim days strictly after this UTC date")
    ap.add_argument("--clean-opp", type=float, default=1.01,
                    help="drop signals where opposite side prob > this")
    ap.add_argument("--min-agree", type=int, default=1,
                    help="#4: require N horizons agreeing (same symbol/time/side)")
    ap.add_argument("--size-by-spread", action="store_true",
                    help="#3: size position proportional to spread (0.5x..2x)")
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0

    grid = pd.read_parquet(C.DATASETS_DIR / "sim_grid.parquet")
    grid["time"] = pd.to_datetime(grid["time"], utc=True)
    if args.after:
        grid = grid[grid["time"] > pd.Timestamp(args.after, tz="UTC")]
    weights = json.loads((C.MODELS_DIR / "trust_weights.json").read_text())["weights"]
    if args.no_trust:
        weights = {m: 1.0 for m in weights}
    for m in [x for x in args.exclude.split(",") if x]:
        weights[m] = 0.0
    pmt = None
    if args.per_model_thr:
        pmt = json.loads((C.MODELS_DIR / "prob_thresholds.json").read_text())
        weights = {m: 1.0 for m in weights}   # per-model threshold replaces trust
    cand = build_candidates(grid, weights, args.floor, args.global_trust, cost, pmt)
    cand = cand[cand.opp <= args.clean_opp]
    if args.min_agree > 1:   # #4: keep only symbol/time/side with >=N horizons agreeing
        cand["agree"] = cand.groupby(["symbol", "time", "side"])["score"].transform("size")
        cand = cand[cand["agree"] >= args.min_agree]
    cand["t_ns"] = anchors_to_ns(cand["time"])
    by_time = {tns: d for tns, d in cand.groupby("t_ns")}
    times = sorted(np.unique(anchors_to_ns(grid["time"])))
    # price lookup (symbol, t_ns) -> price, for mark-to-market drawdown
    grid_ns = anchors_to_ns(grid["time"])
    price_at = {(s, n): p for s, n, p in
                zip(grid["symbol"].to_numpy(), grid_ns, grid["entry_price"].to_numpy())}
    print(f"grid: {grid.symbol.nunique()} coins, {len(times)} 5-min steps "
          f"({pd.Timestamp(times[0]).date()}..{pd.Timestamp(times[-1]).date()})")
    print(f"trusted models in play: "
          f"{[m for m,w in weights.items() if w>0 and w>=args.global_trust]}\n")

    cash = args.equity
    open_pos = {}          # symbol -> dict(exit_time, exit_pnl_dollars)
    last_trade = {}        # symbol -> time
    trades = []
    peak = args.equity
    max_dd = 0.0
    max_concurrent = 0
    equity_curve = []      # (timestamp, mark-to-market equity)

    for t_ns in times:
        t = pd.Timestamp(t_ns, tz="UTC")
        # 1) close due positions
        for sym in [s for s, p in open_pos.items() if p["exit_time"] <= t]:
            p = open_pos.pop(sym)
            cash += p["size"] + p["pnl_dollars"]
            p["rec"]["close_time"] = t
            trades.append(p["rec"])
        # 2) MARK-TO-MARKET equity & drawdown (include unrealized PnL of open pos)
        mtm = cash
        for sym, p in open_pos.items():
            cur = price_at.get((sym, t_ns))
            if cur is not None and p["entry"] > 0:
                ur = p["side"] * (cur / p["entry"] - 1.0)
                mtm += p["size"] * (1.0 + ur)
            else:
                mtm += p["size"]
        peak = max(peak, mtm)
        max_dd = max(max_dd, (peak - mtm) / peak)
        max_concurrent = max(max_concurrent, len(open_pos))
        equity_curve.append((t, mtm))
        # 3) open new trades
        cands = by_time.get(t_ns)
        if cands is None or cands.empty:
            continue
        opened_this_scan = 0
        for r in cands.sort_values("score", ascending=False).itertuples():
            if opened_this_scan >= args.top:
                break
            # #3: size proportional to spread (0.5x..2x), else flat
            size = args.size * min(2.0, max(0.5, r.spread / 0.5)) if args.size_by_spread else args.size
            if cash < size:
                continue   # can't afford this one -> try the next (cheaper) by priority
            if r.symbol in open_pos:
                continue
            lt = last_trade.get(r.symbol)
            if lt is not None and (t - lt).total_seconds() < args.cooldown * 60:
                continue
            cash -= size
            pnl_dollars = size * r.pnl_pct
            open_pos[r.symbol] = {
                "exit_time": t + pd.Timedelta(minutes=r.horizon_min),
                "pnl_dollars": pnl_dollars, "size": size,
                "entry": r.entry, "side": r.side,
                "rec": {"symbol": r.symbol, "model": r.model, "side": r.side,
                        "time": t, "pnl_pct": r.pnl_pct, "pnl_$": pnl_dollars,
                        "won": int(r.pnl_pct > 0)}}
            last_trade[r.symbol] = t
            opened_this_scan += 1
    # close any still-open at their pnl (already known)
    for sym, p in list(open_pos.items()):
        cash += p["size"] + p["pnl_dollars"]
        p["rec"]["close_time"] = p["exit_time"]
        trades.append(p["rec"])
    final = cash

    T = pd.DataFrame(trades)
    print("=== SIMULATION RESULT ===")
    print(f"  start equity : ${args.equity:.2f}")
    print(f"  FINAL equity : ${final:.2f}   ({(final/args.equity-1)*100:+.2f}%)")
    print(f"  trades       : {len(T)}   win-rate {T.won.mean():.3f}" if len(T) else "  no trades")
    if len(T):
        print(f"  avg pnl/trade: {T.pnl_pct.mean()*100:+.4f}%   total $ {T['pnl_$'].sum():+.2f}")
        print(f"  max concurrent positions: {max_concurrent}   max drawdown: {max_dd*100:.1f}%")
        print(f"  by model: ")
        for m, gg in T.groupby("model"):
            print(f"    {m:<9} n={len(gg):>4} win={gg.won.mean():.3f} $ {gg['pnl_$'].sum():+.2f}")

        # per-day breakdown (P&L attributed to CLOSE day; equity = end-of-day mtm)
        eq = pd.DataFrame(equity_curve, columns=["t", "equity"])
        eq["day"] = eq["t"].dt.strftime("%m-%d")
        eod = eq.groupby("day")["equity"].last()
        T["cday"] = pd.to_datetime(T["close_time"]).dt.strftime("%m-%d")
        print("\n  === PER-DAY ===")
        print(f"  {'day':>6} {'trades':>7} {'win':>6} {'P&L $':>9} {'end equity':>12}")
        prev = args.equity
        for day in sorted(set(T["cday"]) | set(eod.index)):
            g = T[T["cday"] == day]
            end_eq = eod.get(day, prev)
            n = len(g)
            win = f"{g.won.mean():.3f}" if n else "   -"
            pl = g["pnl_$"].sum() if n else 0.0
            print(f"  {day:>6} {n:>7} {win:>6} {pl:>+8.2f} {end_eq:>11.2f}  "
                  f"{'UP' if end_eq>=prev else 'DOWN'}")
            prev = end_eq
        ups = (eod.diff().dropna() > 0).sum() + (1 if eod.iloc[0] >= args.equity else 0)
        print(f"  green days: {ups}/{len(eod)}")


if __name__ == "__main__":
    main()
