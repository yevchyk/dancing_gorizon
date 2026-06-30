"""Compare recent live Unicorn trades with a same-anchor candle simulation.

This is a read-only forensic helper. It parses the live event/trade logs,
rebuilds pulse00 decisions on the same scan anchors, runs the live-like
simulator, and prints hourly differences.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .fast import config as FC
from .database import CandleStore
from .run_okx_liquid_sim import OkxLiquidPriceBook
from .run_test_engine_harvest_sim import simulate_engine
from .trading.fast_combo_engine import FastComboEngine


DEFAULT_STORE = Path("data/okx_liquid/candles_mixed")
DEFAULT_RUN = Path("outputs/trading_logs/live_20260603_004444")
SCAN_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"scan @ (?P<anchor>[^:]+?:\d{2}:\d{2}\+00:00): "
    r"(?P<symbols>\d+) symbols, (?P<opened>\d+) opened, "
    r"(?P<harvested>\d+) harvested, (?P<open>\d+) open"
)


def load_symbols(path: str | None, store: CandleStore) -> list[str]:
    if not path:
        return sorted(store.symbols())
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("symbols", data) if isinstance(data, dict) else data
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        sym = str(item).strip().upper().replace("-", "_")
        if sym and sym not in seen:
            out.append(sym)
            seen.add(sym)
    return out


def parse_scans(run_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    path = run_dir / "events.log"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SCAN_RE.match(line)
        if not m:
            continue
        rows.append({
            "log_ts": pd.Timestamp(m.group("ts"), tz="UTC"),
            "anchor": pd.Timestamp(m.group("anchor")),
            "symbols": int(m.group("symbols")),
            "opened": int(m.group("opened")),
            "open_after": int(m.group("open")),
        })
    if not rows:
        raise SystemExit(f"no scan rows in {path}")
    return pd.DataFrame(rows)


def parse_live_trades(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "trades.csv"
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    for col in ("entry_price", "exit_price", "size_usd", "pnl_pct"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def pair_live_trades(raw: pd.DataFrame, scans: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    opens = raw[raw["event"] == "open"].sort_values("ts").copy()
    closes = raw[raw["event"].str.contains("close", na=False)].sort_values("ts").copy()
    scan_rows = scans.sort_values("log_ts").to_dict("records")

    paired: list[dict] = []
    used_close: set[int] = set()
    for open_idx, op in opens.iterrows():
        match_anchor = pd.NaT
        prior = [s for s in scan_rows if s["log_ts"] <= op["ts"] + pd.Timedelta(seconds=2)]
        if prior:
            match_anchor = prior[-1]["anchor"]

        cl_idx = None
        for idx, cl in closes.iterrows():
            if idx in used_close:
                continue
            if cl["ts"] < op["ts"]:
                continue
            if str(cl["symbol"]) == str(op["symbol"]) and str(cl["side"]) == str(op["side"]):
                cl_idx = idx
                break
        cl = closes.loc[cl_idx] if cl_idx is not None else None
        if cl_idx is not None:
            used_close.add(cl_idx)
        gross = float(cl["pnl_pct"]) if cl is not None and pd.notna(cl["pnl_pct"]) else np.nan
        paired.append({
            "anchor": match_anchor,
            "opened_wall": op["ts"],
            "closed_wall": cl["ts"] if cl is not None else pd.NaT,
            "symbol": op["symbol"],
            "side": op["side"],
            "model": op["model"],
            "horizon": op["horizon"],
            "entry_price": float(op["entry_price"]) if pd.notna(op["entry_price"]) else np.nan,
            "exit_price": float(cl["exit_price"]) if cl is not None and pd.notna(cl["exit_price"]) else np.nan,
            "size_usd": float(op["size_usd"]) if pd.notna(op["size_usd"]) else np.nan,
            "gross_pnl_pct": gross,
            "est_net_pnl_pct": gross - FC.EVAL_COST * 100.0 if np.isfinite(gross) else np.nan,
        })
    return pd.DataFrame(paired)


def build_sim_candidates(
    *,
    eng: FastComboEngine,
    store: CandleStore,
    symbols: list[str],
    anchors: list[pd.Timestamp],
    top_per_scan: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    snap_rows: list[dict] = []
    for anchor in anchors:
        snap = eng.snapshot(store, symbols, anchor)
        snap_rows.append({"anchor": anchor, "sim_symbols": int(len(snap))})
        if snap.empty:
            continue
        for sig in eng.decide(snap, top_n=top_per_scan):
            rows.append({
                "engine": "sim_pulse00",
                "family": "okx_liquid",
                "source": getattr(sig, "source", ""),
                "signal_model": sig.model,
                "symbol": sig.symbol,
                "anchor_time": anchor,
                "day": anchor.strftime("%m-%d"),
                "side": 1 if sig.side == "long" else -1,
                "exit": sig.horizon,
                "threshold": np.nan,
                "leverage": 1.0,
                "score": float(sig.score),
            })
    return pd.DataFrame(rows), pd.DataFrame(snap_rows)


def summarize_live(live: pd.DataFrame) -> pd.DataFrame:
    if live.empty:
        return pd.DataFrame(columns=["utc", "kyiv", "live_n", "live_win", "live_gross%", "live_net%"])
    x = live.copy()
    x["utc"] = pd.to_datetime(x["anchor"], utc=True).dt.hour
    x["kyiv"] = (x["utc"] + 3) % 24
    rows = []
    for (utc, kyiv), g in x.groupby(["utc", "kyiv"], sort=True):
        net = g["est_net_pnl_pct"].dropna()
        gross = g["gross_pnl_pct"].dropna()
        rows.append({
            "utc": int(utc),
            "kyiv": int(kyiv),
            "live_n": int(len(g)),
            "live_win": float((net > 0).mean()) if len(net) else np.nan,
            "live_gross%": float(gross.sum()) if len(gross) else 0.0,
            "live_net%": float(net.sum()) if len(net) else 0.0,
        })
    return pd.DataFrame(rows)


def summarize_sim(sim: pd.DataFrame) -> pd.DataFrame:
    if sim.empty:
        return pd.DataFrame(columns=["utc", "kyiv", "sim_n", "sim_win", "sim_net%"])
    x = sim.copy()
    x["utc"] = pd.to_datetime(x["opened_at"], utc=True).dt.hour
    x["kyiv"] = (x["utc"] + 3) % 24
    rows = []
    for (utc, kyiv), g in x.groupby(["utc", "kyiv"], sort=True):
        rows.append({
            "utc": int(utc),
            "kyiv": int(kyiv),
            "sim_n": int(len(g)),
            "sim_win": float(g["won"].mean()),
            "sim_net%": float(g["net_pnl_pct"].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    ap.add_argument("--store-dir", default=str(DEFAULT_STORE))
    ap.add_argument("--symbols-file", default="configs/okx_liquid_symbols_100.json")
    ap.add_argument("--top-per-scan", type=int, default=3)
    ap.add_argument("--max-open", type=int, default=3)
    ap.add_argument("--cooldown-min", type=int, default=10)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    store = CandleStore(Path(args.store_dir))
    symbols = load_symbols(args.symbols_file, store)

    scans_all = parse_scans(run_dir)
    end = scans_all["anchor"].max()
    start = end - pd.Timedelta(hours=args.hours)
    scans = scans_all[(scans_all["anchor"] >= start) & (scans_all["anchor"] <= end)].copy()
    anchors = [pd.Timestamp(t) for t in scans["anchor"].drop_duplicates().sort_values()]
    print(f"run_dir={run_dir}")
    print(f"window={start} -> {end} UTC  scans={len(scans)} anchors={len(anchors)} symbols_file={len(symbols)}")
    if scans.empty:
        raise SystemExit("no scans in requested window")

    live_raw = parse_live_trades(run_dir)
    live = pair_live_trades(live_raw, scans_all)
    live = live[(live["anchor"] >= start) & (live["anchor"] <= end)].copy() if not live.empty else live

    eng = FastComboEngine("pulse00")
    cand, snap = build_sim_candidates(
        eng=eng,
        store=store,
        symbols=symbols,
        anchors=anchors,
        top_per_scan=args.top_per_scan,
    )
    if cand.empty:
        sim = pd.DataFrame()
        blocks = {}
    else:
        sim, blocks = simulate_engine(
            "sim_pulse00",
            cand,
            anchors,
            OkxLiquidPriceBook(),
            harvest=False,
            top_per_scan=args.top_per_scan,
            max_open=args.max_open,
            cooldown_min=args.cooldown_min,
        )

    live_sum = summarize_live(live)
    sim_sum = summarize_sim(sim)
    hours = sorted(set(live_sum.get("utc", pd.Series(dtype=int))).union(set(sim_sum.get("utc", pd.Series(dtype=int)))))
    if not hours:
        hours = sorted(set(pd.to_datetime(scans["anchor"], utc=True).dt.hour))
    base = pd.DataFrame({"utc": hours})
    base["kyiv"] = (base["utc"] + 3) % 24
    hourly = base.merge(sim_sum, on=["utc", "kyiv"], how="left").merge(live_sum, on=["utc", "kyiv"], how="left")
    for col in ("sim_n", "live_n"):
        hourly[col] = hourly[col].fillna(0).astype(int)
    for col in ("sim_net%", "live_gross%", "live_net%"):
        hourly[col] = hourly[col].fillna(0.0)
    hourly["delta_net%"] = hourly["live_net%"] - hourly["sim_net%"]

    print("\nOVERALL")
    sim_net = float(sim["net_pnl_pct"].sum()) if not sim.empty else 0.0
    sim_n = int(len(sim))
    sim_win = float(sim["won"].mean()) if sim_n else np.nan
    live_net = float(live["est_net_pnl_pct"].dropna().sum()) if not live.empty else 0.0
    live_gross = float(live["gross_pnl_pct"].dropna().sum()) if not live.empty else 0.0
    live_n = int(len(live))
    live_win = float((live["est_net_pnl_pct"].dropna() > 0).mean()) if live_n else np.nan
    print(f"sim:  trades={sim_n} win={sim_win:.3f} net={sim_net:+.2f}% candidates={len(cand)} blocks={blocks}")
    print(f"live: trades={live_n} win={live_win:.3f} gross={live_gross:+.2f}% est_net={live_net:+.2f}%")
    print(f"delta live_net - sim_net = {live_net - sim_net:+.2f}%")
    print(f"snapshot symbols: live avg={scans['symbols'].mean():.1f}, sim avg={snap['sim_symbols'].mean():.1f}, "
          f"live range={scans['symbols'].min()}-{scans['symbols'].max()}, "
          f"sim range={snap['sim_symbols'].min()}-{snap['sim_symbols'].max()}")

    print("\nHOURLY")
    print(hourly.to_string(index=False, formatters={
        "sim_win": lambda v: "" if pd.isna(v) else f"{v:.3f}",
        "live_win": lambda v: "" if pd.isna(v) else f"{v:.3f}",
        "sim_net%": "{:+.2f}".format,
        "live_gross%": "{:+.2f}".format,
        "live_net%": "{:+.2f}".format,
        "delta_net%": "{:+.2f}".format,
    }))

    print("\nSIM TRADES")
    if sim.empty:
        print("(none)")
    else:
        show = sim.sort_values("opened_at")[["opened_at", "symbol", "side", "exit", "net_pnl_pct", "won"]]
        print(show.to_string(index=False, formatters={"net_pnl_pct": "{:+.3f}".format}))

    print("\nLIVE TRADES")
    if live.empty:
        print("(none)")
    else:
        show = live.sort_values("anchor")[["anchor", "opened_wall", "symbol", "side", "horizon", "gross_pnl_pct", "est_net_pnl_pct"]]
        print(show.to_string(index=False, formatters={
            "gross_pnl_pct": "{:+.3f}".format,
            "est_net_pnl_pct": "{:+.3f}".format,
        }))


if __name__ == "__main__":
    main()
