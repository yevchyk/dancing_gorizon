"""Trust-layer engine: each model earns a TRUST INDEX from its realized PnL, and
only trusted models vote. Low-trust models drop out automatically; retraining +
recomputing trust lets them rejoin. A global trust knob trades frequency vs
quality.

Trust is computed leak-free on the TRAIN folds, on TWO windows (overall ~50d and
recent ~14d); a model is trusted only if positive on BOTH (stability). The engine
is then evaluated on the held-out latest fold (and its strict last-10-days).

Decision: among trusted, firing (prob>=floor) model-directions, rank by the
risk-adjusted score (prob * MFE/|MAE|) and take top-K/day.

Usage:
  python -m src.run_trust_engine --slip 0.05 --k 20 --global-trust 0.0
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
MODELS = [f"{k}_{h.label}" for h in C.HORIZONS for k in ("up", "down")]


def candidates(s: pd.DataFrame, cost: float) -> pd.DataFrame:
    """Explode each anchor-horizon into its two model-direction candidates."""
    rows = []
    for h in C.HORIZONS:
        sub = s[s.horizon == h.label]
        fav_l = sub.pred_mfe.to_numpy(); adv_l = np.abs(sub.pred_mae.to_numpy())
        fav_s = -sub.pred_mae.to_numpy(); adv_s = sub.pred_mfe.to_numpy()
        for k, prob, side, fav, adv in (
            ("up", sub.p_up.to_numpy(), 1, fav_l, adv_l),
            ("down", sub.p_down.to_numpy(), -1, fav_s, adv_s)):
            rr = np.clip(fav / (np.abs(adv) + 1e-4), 0, 5)
            rows.append(pd.DataFrame({
                "model": f"{k}_{h.label}", "day": sub.day.to_numpy(),
                "fold": sub.fold.to_numpy(), "prob": prob, "side": side,
                "rr": rr, "pnl": side * sub.real_ret.to_numpy() - cost}))
    return pd.concat(rows, ignore_index=True)


def trust_table(cand: pd.DataFrame, floor: float) -> pd.DataFrame:
    """Per-model trust = realized avg PnL when firing, on overall + recent train."""
    fire = cand[cand.prob >= floor]
    days = pd.to_datetime(fire.day)
    recent_cut = days.max() - pd.Timedelta(days=14)
    out = []
    for m in MODELS:
        g = fire[fire.model == m]
        gr = g[pd.to_datetime(g.day) > recent_cut]
        t_all = g.pnl.mean() * 100 if len(g) else np.nan
        t_rec = gr.pnl.mean() * 100 if len(gr) >= 8 else np.nan
        trusted = (t_all > 0) and (t_rec > 0)
        out.append({"model": m, "n_all": len(g), "trust_50d": round(t_all, 4),
                    "n_rec": len(gr), "trust_10d": round(t_rec, 4) if not np.isnan(t_rec) else None,
                    "trusted": bool(trusted),
                    "weight": round(min(t_all, t_rec), 4) if trusted else 0.0})
    return pd.DataFrame(out)


def run_engine(cand_test: pd.DataFrame, trust: pd.DataFrame, floor: float,
               k_per_day: int, gthr: float) -> pd.DataFrame:
    tmap = trust.set_index("model")
    allowed = set(tmap[(tmap.trusted) & (tmap.weight >= gthr)].index)
    c = cand_test[(cand_test.model.isin(allowed)) & (cand_test.prob >= floor)].copy()
    if c.empty:
        return c
    c["w"] = c.model.map(tmap.weight)
    c["score"] = c.prob * c.rr * c.w          # prob x reward/risk x trust
    c["rk"] = c.groupby("day")["score"].rank(ascending=False, method="first")
    return c[c.rk <= k_per_day]


def stats(df, tag):
    if len(df) < 3:
        print(f"  {tag:<22} n={len(df)} (too few)"); return
    dg = df.groupby("day")["pnl"].mean() * 100
    print(f"  {tag:<22} n={len(df):>4} win={(df.pnl>0).mean():.3f} "
          f"avg_pnl={df.pnl.mean()*100:+.4f}%  green={int((dg>0).sum())}/{len(dg)}  "
          f"total={df.pnl.sum()*100:+.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip", type=float, default=0.05)
    ap.add_argument("--floor", type=float, default=0.60)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--global-trust", type=float, default=0.0)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0

    s = pd.read_parquet(C.OUTPUTS_DIR / "analysis" / "engine_stats.parquet")
    cand = candidates(s, cost)
    last_fold = s.fold.max()
    tr = cand[cand.fold < last_fold]
    te = cand[cand.fold == last_fold]
    cut = (pd.to_datetime(te.day).max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    trust = trust_table(tr, args.floor)
    print("=== PER-MODEL TRUST (train: 50d overall vs 14d recent) ===")
    print(trust.to_string(index=False))
    print(f"\ntrusted models: {trust.trusted.sum()}/{len(trust)}  "
          f"(global_trust>={args.global_trust})\n")

    picks = run_engine(te, trust, args.floor, args.k, args.global_trust)
    last = picks[pd.to_datetime(picks.day) > cut] if len(picks) else picks
    print(f"=== TRUST ENGINE (k={args.k}/day) ===")
    stats(picks, "fold3 (~14d)")
    stats(last, f"last-10d (>{cut})")

    # baseline: all models equal (no trust gate), same risk-adj top-k/day
    base = te[te.prob >= args.floor].copy()
    base["score"] = base.prob * base.rr
    base["rk"] = base.groupby("day")["score"].rank(ascending=False, method="first")
    bpick = base[base.rk <= args.k]
    bl = bpick[pd.to_datetime(bpick.day) > cut]
    print("\n=== BASELINE (all models, no trust gate, same top-k) ===")
    stats(bpick, "fold3 (~14d)")
    stats(bl, f"last-10d (>{cut})")


if __name__ == "__main__":
    main()
