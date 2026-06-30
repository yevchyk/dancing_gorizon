"""Sweep exit/management policies for ONE pocket -> find the best EDGE multiplier,
confirmed A->B.

The single-spec exit-test (run_binance_exittest) answers "is THIS exit good?".
This tool answers "which exit is BEST for this pocket, and does it hold out of
sample?" — it scores the model once, selects the pocket's entries (1 pos per
(symbol, scan), highest p_dir, like live), splits the live pool into two time
halves A (older) / B (newer), then evaluates a GRID of price-path exit policies
(trail / take-profit / crash-stop / combos) against the HOLD-to-horizon baseline.

We judge by EDGE = avg net % per position (NOT winrate), and a policy only
"passes" if it beats/holds HOLD on BOTH halves (anti-overfit: selecting the exit
on the same window we report on would inflate it, exactly the §11/§6.2 trap).

Reuses run_binance_exittest / run_binance_engine_exittest helpers verbatim, so
costs (honest per-symbol thr_pct), the 1-min price path, the dedup and the exit
math stay bit-identical to the explorer tools.

  python -m src.run_binance_exit_sweep --sim "binance d8" --pmin 0.80 \
      --hmin 95 --hmax 240 --side long --out outputs/exit_sweep_d8_flagship.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .binance_fetcher import norm_symbol
from .hc_model_registry import SIM_TO_DIR
from .run_binance_exittest import (
    COSTS, LIQUID, FLOOR, ENTRY_DELAY, NOTIONAL, _close_series, _apply,
)
from .run_binance_engine_exittest import (
    DATASET_NOW, DATASET_Y1, DATASET_V5, _score_model, _legs,
)


def _levels_from_cli(side: str, pmin: float, hmin: int, hmax: int) -> list:
    """Build explorer-style entry levels from CLI knobs (consumed by _apply/_cond)."""
    lv = []
    if side != "both":
        lv.append({"conditions": [{"regulator": "side", "value": side}]})
    lv.append({"conditions": [{"regulator": "p_dir", "value": pmin}]})
    lv.append({"conditions": [{"regulator": "hmin", "value": hmin}]})
    lv.append({"conditions": [{"regulator": "hmax", "value": hmax}]})
    return lv


def _grid() -> list[tuple[str, dict]]:
    """HOLD baseline + the price-path 'exit multiplier' levers. trail = let the
    winner run then give back; tp = fixed take; crash = disaster stop; combos add
    a wide stop under a trail. Kept ~24 policies so a sweep stays fast."""
    pols: list[tuple[str, dict]] = [("HOLD", {})]
    for arm in (0.6, 1.0, 1.5, 2.0):
        for give in (0.3, 0.5, 0.8):
            pols.append((f"trail arm{arm} give{give}", {"trail": {"arm": arm, "give": give}}))
    for tp in (1.0, 1.5, 2.0, 3.0):
        pols.append((f"tp {tp}", {"tp": {"pct": tp}}))
    for cp in (1.0, 1.5, 2.0):
        pols.append((f"crash {cp}", {"crash": {"pct": cp}}))
    for arm, give in ((1.0, 0.5), (1.5, 0.5)):
        for cp in (2.0, 3.0):
            pols.append((f"trail arm{arm} give{give} + crash {cp}",
                         {"trail": {"arm": arm, "give": give}, "crash": {"pct": cp}}))
    return pols


def _policy_net(fav: np.ndarray, tmins: np.ndarray, cost: float, ep: dict) -> float:
    """Net % of one position under exit policy ep on a precomputed sgn-adjusted
    favourable-return path. Mirrors run_binance_engine_exittest._exit_net exactly
    (tp/crash/trail, earliest trigger wins; HOLD = close at horizon)."""
    trig: dict[str, tuple[float, float]] = {}  # name -> (minute, net)
    tp = ep.get("tp")
    if tp:
        idx = np.where(fav >= tp["pct"])[0]
        if len(idx):
            trig["tp"] = (tmins[idx[0]], tp["pct"] - cost)
    crash = ep.get("crash")
    if crash:
        idx = np.where(-fav >= crash["pct"])[0]
        if len(idx):
            trig["crash"] = (tmins[idx[0]], -crash["pct"] - cost)
    trail = ep.get("trail")
    if trail:
        arm, give = trail["arm"], trail["give"]
        peak = -1e9
        for j in range(len(fav)):
            if fav[j] > peak:
                peak = fav[j]
            if peak >= arm and fav[j] <= peak - give:
                trig["trail"] = (tmins[j], (peak - give) - cost)
                break
    if trig:
        return min(trig.values(), key=lambda x: x[0])[1]
    return float(fav[-1]) - cost  # HOLD to horizon


def _agg(nets: list[float]) -> dict:
    if not nets:
        return {"n": 0, "avg_net": 0.0, "win": 0.0, "usd": 0.0, "p90": 0.0}
    a = np.asarray(nets)
    return {"n": len(a), "avg_net": round(float(a.mean()), 4),
            "win": round(100.0 * float((a > 0).mean()), 1),
            "usd": round(NOTIONAL * float(a.sum()) / 100.0, 2),
            "p90": round(float(np.percentile(a, 90)), 3)}  # tail-capture diagnostic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="binance d8")
    ap.add_argument("--spec", type=Path, help="optional explorer build json {sim,levels}; overrides CLI levels")
    ap.add_argument("--pmin", type=float, default=0.80)
    ap.add_argument("--hmin", type=int, default=95)
    ap.add_argument("--hmax", type=int, default=240)
    ap.add_argument("--side", default="long", choices=["long", "short", "both"])
    ap.add_argument("--days", type=int, default=12, help="window length back from --asof/now")
    ap.add_argument("--asof", default=None, help="treat this UTC time as 'now' (e.g. 2026-06-10); "
                    "lets the sweep target a historical window for large-n verdicts")
    ap.add_argument("--dataset", default="auto", choices=["auto", "now", "y1", "v5"],
                    help="force source: auto=now-if-present-else-y1 (v5 sims always v5); "
                         "use y1 for a wide multi-month window (large n)")
    ap.add_argument("--out", type=Path, default=C.OUTPUTS_DIR / "exit_sweep.json")
    args = ap.parse_args()

    if args.spec:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        sim = spec["sim"]; levels = spec.get("levels", [])
        pocket = f"spec:{args.spec.name}"
    else:
        sim = args.sim
        levels = _levels_from_cli(args.side, args.pmin, args.hmin, args.hmax)
        pocket = f"{args.side} h{args.hmin}-{args.hmax} p_dir>={args.pmin}"

    mdir = Path(SIM_TO_DIR.get(sim, sim))
    if args.dataset == "y1":
        dset = DATASET_V5 if "v5" in sim else DATASET_Y1
    elif args.dataset == "now":
        dset = DATASET_NOW
    elif args.dataset == "v5":
        dset = DATASET_V5
    else:  # auto
        dset = DATASET_V5 if "v5" in sim else (DATASET_NOW if DATASET_NOW.exists() else DATASET_Y1)

    ref = (pd.Timestamp(args.asof, tz="UTC") if args.asof else pd.Timestamp.now(tz="UTC")).floor("min")
    start = ref - pd.Timedelta(days=args.days)
    print(f"exit-sweep sim={sim} pocket=[{pocket}] -> {mdir}  dataset={dset.parent.name}  "
          f"window {str(start)[:16]} .. {str(ref)[:16]}", flush=True)
    costs = json.loads(COSTS.read_text(encoding="utf-8"))["costs"]
    liquid = [norm_symbol(s) for s in json.loads(LIQUID.read_text(encoding="utf-8"))["symbols"]]
    trade = [s for s in liquid if s in costs]

    df = _score_model(mdir, dset, trade, start)
    df = df[df["base_time"] < ref].reset_index(drop=True)  # cap at asof (score helper has no upper bound)
    legs = _legs(df)                       # bt = epoch minutes; both sides; FLOOR pre-cut
    sel = _apply(legs, levels)
    best: dict[tuple, dict] = {}           # 1 position per (symbol, scan): highest p_dir
    for l in sel:
        k = (l["sym"], l["bt"])
        if k not in best or l["pd"] > best[k]["pd"]:
            best[k] = l
    entries = list(best.values())
    print(f"entries after pocket levels (1/sym/scan): {len(entries)}", flush=True)
    if not entries:
        args.out.write_text(json.dumps({"sim": sim, "pocket": pocket, "entries": 0}, indent=2), encoding="utf-8")
        print("no entries -> wrote empty result"); return

    # precompute the sgn-adjusted favourable path ONCE per entry (the heavy part)
    closes: dict[str, object] = {}
    def cs_for(s):
        if s not in closes:
            closes[s] = _close_series(s)
        return closes[s]

    rows = []  # (bt_min, fav, tmins, cost)
    for e in entries:
        cs = cs_for(e["sym"])
        if cs is None:
            continue
        et = pd.Timestamp(e["bt"] * 60_000_000_000, tz="UTC") + pd.Timedelta(minutes=ENTRY_DELAY)
        xt = et + pd.Timedelta(minutes=e["h"])
        ei = cs.index.searchsorted(et, side="right") - 1
        if ei < 0:
            continue
        entry_px = float(cs.iloc[ei])
        path = cs[(cs.index >= et) & (cs.index <= xt)]
        if not len(path):
            continue
        sgn = 1.0 if e["side"] > 0 else -1.0
        fav = sgn * (path.to_numpy() / entry_px - 1.0) * 100.0
        tmins = ((path.index.astype("int64") // 60_000_000_000) - (et.value // 60_000_000_000)).to_numpy()
        rows.append((e["bt"], fav, tmins, e["thr"]))

    n = len(rows)
    bts = sorted(r[0] for r in rows)
    mid = bts[len(bts) // 2]                # split entries into A (older) / B (newer)
    win_start = pd.Timestamp(min(bts) * 60_000_000_000, tz="UTC")
    win_end = pd.Timestamp(max(bts) * 60_000_000_000, tz="UTC")
    print(f"resolved {n} positions  window {str(win_start)[:16]} .. {str(win_end)[:16]}  "
          f"A/B split @ {str(pd.Timestamp(mid*60_000_000_000, tz='UTC'))[:16]}", flush=True)

    results = []
    for name, ep in _grid():
        all_n, a_n, b_n = [], [], []
        for bt, fav, tmins, cost in rows:
            net = _policy_net(fav, tmins, cost, ep)
            all_n.append(net)
            (a_n if bt <= mid else b_n).append(net)
        results.append({"policy": name, "ep": ep,
                        "all": _agg(all_n), "A": _agg(a_n), "B": _agg(b_n)})

    hold = next(r for r in results if r["policy"] == "HOLD")
    hA, hB, hAll = hold["A"]["avg_net"], hold["B"]["avg_net"], hold["all"]["avg_net"]
    for r in results:
        r["d_all"] = round(r["all"]["avg_net"] - hAll, 4)
        # anti-overfit gate: must beat/hold HOLD on BOTH halves AND overall
        r["ab_pass"] = bool(r["all"]["avg_net"] > hAll and r["A"]["avg_net"] >= hA
                            and r["B"]["avg_net"] >= hB)

    ranked = sorted(results, key=lambda r: -r["all"]["avg_net"])
    winner = next((r for r in ranked if r["policy"] != "HOLD" and r["ab_pass"]), None)

    out = {
        "sim": sim, "pocket": pocket, "dataset": dset.parent.name, "notional": NOTIONAL,
        "window_start": str(win_start)[:16], "window_end": str(win_end)[:16],
        "positions": n, "ab_split": str(pd.Timestamp(mid * 60_000_000_000, tz="UTC"))[:16],
        "hold": hold, "winner": winner, "ranked": ranked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # readable console table (edge-first)
    print(f"\n=== EXIT SWEEP -- {sim} -- [{pocket}] -- {n} pos -- judge=EDGE(avg net %), A->B gated ===")
    print(f"{'policy':32s} {'n':>4} {'avg_net':>8} {'p90':>7} {'win%':>6} {'d_HOLD':>9} {'A':>7} {'B':>7}  A->B")
    for r in ranked:
        tag = "  <<<" if r is winner else ("  pass" if r["ab_pass"] else "")
        print(f"{r['policy']:32s} {r['all']['n']:>4} {r['all']['avg_net']:>8.3f} "
              f"{r['all']['p90']:>7.2f} {r['all']['win']:>6.1f} {r['d_all']:>+9.3f} "
              f"{r['A']['avg_net']:>7.3f} {r['B']['avg_net']:>7.3f}{tag}")
    if winner:
        print(f"\nBEST multiplier: {winner['policy']}  edge {winner['all']['avg_net']:+.3f}%/pos "
              f"(HOLD {hAll:+.3f}), A {winner['A']['avg_net']:+.3f} / B {winner['B']['avg_net']:+.3f} -- holds A->B")
    else:
        print("\nNo exit beat HOLD on BOTH halves — HOLD-to-horizon stays the policy for this pocket.")
    print("wrote " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
