"""Unicorn simulation on okx_liquid new symbols, last 3 days.

Uses the existing fixed candle-replay simulator (simulate_engine).
Stake: $10 x 5 leverage = $50 notional per trade.
Account: $100 starting (for context).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from pathlib import Path

from . import config as C
from .fast import config as FC
from .trading.fast_combo_engine import FastComboEngine, WORTHY
from .trading.timeutil import index_to_ns
from .run_test_engine_harvest_sim import simulate_engine

STAKE      = 10.0          # $ per trade
LEVERAGE   = 5             # x leverage
NOTIONAL   = STAKE * LEVERAGE   # $50 notional per trade
ACCOUNT    = 100.0         # starting $
EVAL       = FC.EVAL_COST
DAYS       = 3
TOP_SCAN   = 3
MAX_OPEN   = 5             # max concurrent (50*5=$250 margin from $100)
COOLDOWN   = 10
EXIT_H     = "10m"

OKX_DIR = Path("data/okx_liquid/candles_mixed")


# ---- custom PriceBook that reads from okx_liquid/candles_mixed ----
class OkxLiquidPriceBook:
    def __init__(self) -> None:
        self._cache: dict = {}

    def _load(self, symbol: str):
        if symbol in self._cache:
            return self._cache[symbol]
        path = OKX_DIR / f"{symbol}.parquet"
        if not path.exists():
            self._cache[symbol] = None
            return None
        try:
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            df = df.sort_index()
            ts_ns = index_to_ns(df.index)
            cl = df["close"].to_numpy("float64")
            self._cache[symbol] = (ts_ns, cl)
            return self._cache[symbol]
        except Exception:
            self._cache[symbol] = None
            return None

    def at(self, symbol: str, t: pd.Timestamp):
        data = self._load(symbol)
        if data is None:
            return None
        ts_ns, cl = data
        t_ns = int(pd.Timestamp(t).value)
        idx = int(np.searchsorted(ts_ns, t_ns, side="right")) - 1
        if idx < 0:
            return None
        px = float(cl[idx])
        return px if np.isfinite(px) and px > 0 else None


def main() -> None:
    t0 = time.time()
    eng = FastComboEngine("pulse00")

    # symbols: new (not trained) from okx_liquid
    trained = {p.stem for p in FC.FAST_CHUNKS_DIR.glob("*.parquet")}
    blacklist = set(C.BLACKLIST_SYMBOLS)
    all_okx = {p.stem for p in OKX_DIR.glob("*.parquet")}
    new_syms = sorted(all_okx - trained - blacklist)
    # also include known symbols that are in okx_liquid (broader test)
    known_syms = sorted((all_okx & trained) - blacklist)
    syms = new_syms + known_syms
    print(f"okx_liquid: {len(new_syms)} new + {len(known_syms)} known = {len(syms)} total")

    now = pd.Timestamp.now(tz="UTC").floor("1min")
    end = now - pd.Timedelta(minutes=12)
    start = now - pd.Timedelta(days=DAYS)
    anchors = pd.date_range(start.ceil("2min"), end.floor("2min"), freq="2min")
    ans_ns = anchors.as_unit("ns").asi8
    print(f"window {start:%m-%d %H:%M} -> {end:%m-%d %H:%M} UTC  anchors={len(anchors)}")

    # worthy model name map
    worthy_map = {
        "fast_v2_up_10m": "up_10m", "fast_v2_up_8m": "up_8m",
        "fast_v2_up_2m": "up_2m",   "fast_v2_down_10m": "down_10m",
        "fast_v2_down_8m": "down_8m", "fast_v2_down_2m": "down_2m",
    }

    # build candidates
    cand_rows = []
    scored = 0
    for sym in syms:
        path = OKX_DIR / f"{sym}.parquet"
        try:
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            df = df.sort_index()
            if len(df) < 300:
                continue
        except Exception:
            continue

        ff, fv = eng.curve.build_matrix(
            index_to_ns(df.index), df["close"].to_numpy("float64"), ans_ns
        )
        if fv.sum() == 0:
            continue
        idx = np.where(fv)[0]
        X = pd.DataFrame(ff[idx], columns=eng.columns)

        up = np.zeros(len(idx)); dn = np.zeros(len(idx))
        sc_up = np.zeros(len(idx)); sc_dn = np.zeros(len(idx))
        for nm, (mname, _sn, side, base) in WORTHY.items():
            key = worthy_map[nm]
            m, cols = eng._models[key]
            p = m.predict_proba(X[cols])[:, 1]
            hr = np.clip((p - base) / max(1e-9, 1.0 - base), 0, None)
            if side == 1:
                up += (p >= base); sc_up += hr
            else:
                dn += (p >= base); sc_dn += hr

        long_ok = (up >= 3) & (dn == 0)
        short_ok = (dn >= 3) & (up == 0)
        fire = long_ok | short_ok
        scored += 1

        for i in np.where(fire)[0]:
            side_v = 1 if long_ok[i] else -1
            score = float(sc_up[i] if long_ok[i] else sc_dn[i])
            cand_rows.append({
                "engine": "unicorn_okx",
                "family": "new" if sym in new_syms else "known",
                "source": "unicorn",
                "signal_model": "PulseClean3",
                "symbol": sym,
                "anchor_time": anchors[idx[i]],
                "day": anchors[idx[i]].strftime("%m-%d"),
                "side": side_v,
                "exit": EXIT_H,
                "threshold": np.nan,
                "leverage": float(LEVERAGE),
                "score": score,
            })

    print(f"Scored {scored}/{len(syms)} symbols in {time.time()-t0:.0f}s")

    if not cand_rows:
        print("No Unicorn signals found in this window.")
        return

    cand = pd.DataFrame(cand_rows)
    scan_times = sorted(pd.Timestamp(t) for t in cand["anchor_time"].drop_duplicates())
    book = OkxLiquidPriceBook()

    trades, blocks = simulate_engine(
        "unicorn_okx", cand, scan_times, book,
        harvest=False, top_per_scan=TOP_SCAN,
        max_open=MAX_OPEN, cooldown_min=COOLDOWN,
    )

    print(f"\ncandidates: {len(cand)}  scan_times: {len(scan_times)}")
    print(f"blocks: max_open={blocks['block_max_open']} "
          f"same_sym={blocks['block_already_open']} "
          f"cooldown={blocks['block_cooldown']} "
          f"no_price={blocks['block_no_price']}")

    if trades.empty:
        print("Simulator: 0 trades (all blocked or no fills).")
        return

    # dollar P&L
    trades["dollar_pnl"] = NOTIONAL * trades["net_pnl_pct"] / 100.0
    trades["is_new"] = trades["symbol"].isin(new_syms)

    # ---- summary ----
    n = len(trades)
    win = float(trades["won"].mean())
    avg_pct = float(trades["net_pnl_pct"].mean())
    total_pct = float(trades["net_pnl_pct"].sum())
    total_usd = float(trades["dollar_pnl"].sum())
    final_acc = ACCOUNT + total_usd

    print(f"\n{'='*65}")
    print(f"  UNICORN on OKX-LIQUID  |  last {DAYS} days")
    print(f"  Stake ${STAKE} x {LEVERAGE}x = ${NOTIONAL} notional/trade")
    print(f"  Account: ${ACCOUNT} start  ->  ${final_acc:.2f} finish")
    print(f"{'='*65}")
    print(f"  Trades:        {n}")
    print(f"  Win rate:      {win:.3f}  ({int(win*n)}/{n})")
    print(f"  Avg per trade: {avg_pct:+.4f}%  /  ${NOTIONAL*avg_pct/100:+.3f}")
    print(f"  Total %:       {total_pct:+.2f}%")
    print(f"  Total $:       ${total_usd:+.2f}")
    print(f"  PnL/account:   {total_usd/ACCOUNT*100:+.1f}%")
    print(f"{'='*65}")

    # ---- per day ----
    print("\n  Per day:")
    print(f"  {'day':>5}  {'n':>3}  {'win':>5}  {'total%':>8}  {'total$':>8}")
    for day, g in trades.groupby("open_day"):
        dpct = g.net_pnl_pct.sum(); dusd = g.dollar_pnl.sum()
        print(f"  {day:>5}  {len(g):>3}  {g.won.mean():>5.3f}  {dpct:>+8.2f}%  ${dusd:>+7.2f}")

    # ---- new vs known ----
    print("\n  New symbols vs known:")
    for label, mask in [("new (never seen)", trades.is_new), ("known (trained)", ~trades.is_new)]:
        g = trades[mask]
        if len(g) == 0:
            continue
        print(f"  {label:25s}: n={len(g):3d} win={g.won.mean():.3f} "
              f"avg={g.net_pnl_pct.mean():+.4f}% total${g.dollar_pnl.sum():+.2f}")

    # ---- per symbol top ----
    print("\n  Top symbols by total $:")
    sym_g = trades.groupby("symbol").agg(
        n=("won","count"), win=("won","mean"),
        avg_pct=("net_pnl_pct","mean"), total_usd=("dollar_pnl","sum")
    ).sort_values("total_usd", ascending=False).head(10)
    for sym, r in sym_g.iterrows():
        tag = " [NEW]" if sym in new_syms else ""
        print(f"    {sym:22s}{tag:7s} n={int(r.n):2d} win={r.win:.2f} "
              f"avg={r.avg_pct:+.3f}% total=${r.total_usd:+.2f}")

    # ---- all trades table ----
    print(f"\n  All {n} trades:")
    print(f"  {'time':>5}  {'symbol':20}  {'side':>5}  {'held':>5}  "
          f"{'pnl%':>7}  {'$pnl':>7}  {'W/L':>4}  {'src':>4}")
    print("  " + "-"*72)
    for _, r in trades.sort_values("opened_at").iterrows():
        tag = "[N]" if r.symbol in new_syms else "   "
        wl = "WIN" if r.won else "loss"
        print(f"  {str(r.opened_at)[11:16]:>5}  {r.symbol:20}  "
              f"{'LONG' if r.side_int==1 else 'SHORT':>5}  "
              f"{r.held_min:>5.0f}m  "
              f"{r.net_pnl_pct:>+7.3f}%  ${r.dollar_pnl:>+6.2f}  {wl:>4}  {tag}")


if __name__ == "__main__":
    main()
