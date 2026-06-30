"""My own research on top of the trust engine, from the OOF table (no retrain):

  A. TRUST TRANSFER: is the trust-engine's lift over baseline consistent on
     EVERY fold transition, or a fold2->3 fluke?
  B. MFE/MAE EXIT: we only use excursions to SELECT. Does using them to EXIT
     (take-profit / stop) beat the fixed-horizon close, on the engine's picks?

Usage:
  python -m src.run_research --slip 0.05
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
MODELS = [f"{k}_{h.label}" for h in C.HORIZONS for k in ("up", "down")]


def explode(s: pd.DataFrame, cost: float) -> pd.DataFrame:
    """One row per model-direction candidate with raw excursions + reward/risk."""
    rows = []
    for h in C.HORIZONS:
        sub = s[s.horizon == h.label]
        fav_l, adv_l = sub.pred_mfe.to_numpy(), np.abs(sub.pred_mae.to_numpy())
        for k, prob, side, fav, adv in (
            ("up", sub.p_up.to_numpy(), 1, sub.pred_mfe.to_numpy(), np.abs(sub.pred_mae.to_numpy())),
            ("down", sub.p_down.to_numpy(), -1, -sub.pred_mae.to_numpy(), sub.pred_mfe.to_numpy())):
            rr = np.clip(fav / (np.abs(adv) + 1e-4), 0, 5)
            rows.append(pd.DataFrame({
                "model": f"{k}_{h.label}", "day": sub.day.to_numpy(),
                "fold": sub.fold.to_numpy(), "prob": prob, "side": side, "rr": rr,
                "ret": sub.real_ret.to_numpy(), "mfe": sub.real_mfe.to_numpy(),
                "mae": sub.real_mae.to_numpy()}))
    c = pd.concat(rows, ignore_index=True)
    c["pnl"] = c.side * c.ret - cost
    return c


def trust_weights(tr: pd.DataFrame, floor: float) -> dict:
    fire = tr[tr.prob >= floor]
    rc = pd.to_datetime(fire.day).max() - pd.Timedelta(days=14)
    w = {}
    for m in MODELS:
        g = fire[fire.model == m]
        gr = g[pd.to_datetime(g.day) > rc]
        t_all = g.pnl.mean() if len(g) else -1
        t_rec = gr.pnl.mean() if len(gr) >= 8 else -1
        w[m] = min(t_all, t_rec) if (t_all > 0 and t_rec > 0) else 0.0
    return w


def engine_pick(te: pd.DataFrame, w: dict, floor: float, k: int) -> pd.DataFrame:
    c = te[te.prob >= floor].copy()
    c["w"] = c.model.map(w)
    c = c[c.w > 0]
    c["score"] = c.prob * c.rr * c.w
    c["rk"] = c.groupby("day")["score"].rank(ascending=False, method="first")
    return c[c.rk <= k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--floor", type=float, default=0.60)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0

    s = pd.read_parquet(C.OUTPUTS_DIR / "analysis" / "engine_stats.parquet")
    cand = explode(s, cost)

    print("=== A. TRUST TRANSFER (each fold transition) ===")
    print(f"  {'test_fold':>9} | {'TRUST win/pnl':>20} | {'BASELINE win/pnl':>20}")
    for tf in sorted(cand.fold.unique())[1:]:
        tr, te = cand[cand.fold < tf], cand[cand.fold == tf]
        p = engine_pick(te, trust_weights(tr, args.floor), args.floor, args.k)
        b = te[te.prob >= args.floor].copy()
        b["score"] = b.prob * b.rr
        b["rk"] = b.groupby("day")["score"].rank(ascending=False, method="first")
        b = b[b.rk <= args.k]
        flag = "  <<trust better" if p.pnl.mean() > b.pnl.mean() else ""
        print(f"  {tf:>9} | win={(p.pnl>0).mean():.3f} pnl={p.pnl.mean()*100:+.4f}% | "
              f"win={(b.pnl>0).mean():.3f} pnl={b.pnl.mean()*100:+.4f}%{flag}")

    print("\n=== B. MFE/MAE EXIT vs fixed-horizon close (trust picks, last fold) ===")
    tf = cand.fold.max()
    picks = engine_pick(cand[cand.fold < tf], None, args.floor, args.k) if False else \
        engine_pick(cand[cand.fold == tf], trust_weights(cand[cand.fold < tf], args.floor),
                    args.floor, args.k)
    print(f"  fixed close   : win={(picks.pnl>0).mean():.3f} avg_pnl={picks.pnl.mean()*100:+.4f}%")
    ret, mfe, mae, side = (picks.ret.to_numpy(), picks.mfe.to_numpy(),
                           picks.mae.to_numpy(), picks.side.to_numpy())
    fav = np.where(side == 1, mfe, -mae)
    adv = np.where(side == 1, -mae, mfe)
    print(f"  {'TP%':>5} {'SL%':>5} | {'win':>5} {'avg_pnl':>9}")
    for tp, sl in [(0.5, 0.5), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.5)]:
        TP, SL = tp / 100, sl / 100
        htp, hsl = fav >= TP, adv >= SL
        out = side * ret
        out = np.where(htp & ~hsl, TP, out)
        out = np.where(hsl & ~htp, -SL, out)
        out = np.where(htp & hsl, -SL, out)   # conservative: stop first
        pnl = out - cost
        print(f"  {tp:>5} {sl:>5} | {(pnl>0).mean():>5.3f} {pnl.mean()*100:>+8.4f}%")


if __name__ == "__main__":
    main()
