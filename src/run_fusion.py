"""Fusion sandbox for the 3-view crypto ensemble:
   A = bluechipmag (magnitude screamer), B = bluechip (direction), C = listener (market regime).

Loads the three holdout score sets, aligns on (symbol, anchor_time), and lets us
test arbitrary fusion configs fast: per-model thresholds, weights, methods
(avg / product / logodds / rank / min / max), and a market-regime GATE by C.

  python -m src.run_fusion --horizon 32m            # default battery at 32m
  python -m src.run_fusion --horizon 18m --gate 0.55
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
COST = 0.0012
BASE = "outputs/analysis/fast_bluechip"
SETS = {"A": "bluechipmag", "B": "bluechip", "C": "listener"}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _rank(p):
    return pd.Series(p).rank(pct=True).to_numpy()


def load(horizon: str):
    import os
    frames = {}
    for k, tag in SETS.items():
        p = f"{BASE}/{tag}/holdout_scores.parquet"
        if not os.path.exists(p):
            print(f"[warn] {k} ({tag}) not found yet -> skipped")
            continue
        d = pd.read_parquet(p, columns=["symbol", "anchor_time", "day",
                                        f"p_up_{horizon}", f"real_ret_{horizon}"])
        frames[k] = d.rename(columns={f"p_up_{horizon}": f"p_{k}",
                                      f"real_ret_{horizon}": "real_ret"})
    if "B" not in frames:
        raise SystemExit("need at least B (bluechip)")
    m = frames["B"]
    for k in ("A", "C"):
        if k in frames:
            m = m.merge(frames[k][["symbol", "anchor_time", f"p_{k}"]],
                        on=["symbol", "anchor_time"], how="inner")
    return m


def evaluate(name, fire, real, day, days):
    n = int(fire.sum())
    if n < days:
        return None
    pnl = real[fire] - COST
    dd = pd.DataFrame({"d": day[fire], "p": pnl}).groupby("d")["p"].sum() * 30
    return dict(name=name, n=n, nday=n // days, win=float((pnl > 0).mean()),
                avg=float(pnl.mean() * 100), dpd=float(pnl.sum() / days * 30),
                worst=float(dd.min()), grn=float((dd > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="32m")
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--gate", type=float, default=0.55, help="C regime-gate threshold")
    ap.add_argument("--wA", type=float, default=1.0)
    ap.add_argument("--wB", type=float, default=1.0)
    ap.add_argument("--wC", type=float, default=1.0)
    args = ap.parse_args()

    m = load(args.horizon)
    have = [k for k in ("A", "B", "C") if f"p_{k}" in m.columns]
    days = m["day"].nunique()
    real = m["real_ret"].to_numpy(); day = m["day"].to_numpy()
    P = {k: m[f"p_{k}"].to_numpy() for k in have}
    print(f"horizon={args.horizon} rows={len(m)} days={days} models={have}")
    if len(have) > 1:
        import itertools
        for a, b in itertools.combinations(have, 2):
            print(f"  corr({a},{b})={np.corrcoef(P[a],P[b])[0,1]:.3f}")
    print()

    res = []
    thr = args.thr
    # singles
    for k in have:
        res.append(evaluate(f"{k} alone @{thr}", P[k] >= thr, real, day, days))
    if len(have) >= 2:
        ks = have
        w = {"A": args.wA, "B": args.wB, "C": args.wC}
        avg = sum(w[k] * P[k] for k in ks) / sum(w[k] for k in ks)
        prod = np.prod([P[k] for k in ks], axis=0)
        logodds = 1 / (1 + np.exp(-np.mean([_logit(P[k]) for k in ks], axis=0)))
        rankf = np.mean([_rank(P[k]) for k in ks], axis=0)
        mn = np.min([P[k] for k in ks], axis=0)
        res.append(evaluate(f"AVG(w) @{thr}", avg >= thr, real, day, days))
        res.append(evaluate("PRODUCT @{:.2f}^n".format(thr), prod >= thr ** len(ks), real, day, days))
        res.append(evaluate(f"LOGODDS @{thr}", logodds >= thr, real, day, days))
        res.append(evaluate("RANK top", rankf >= 0.97, real, day, days))
        res.append(evaluate(f"MIN(all) @{thr}", mn >= thr, real, day, days))
        res.append(evaluate(f"CONFLUENCE all>= {thr}", np.all([P[k] >= thr for k in ks], axis=0), real, day, days))
    # regime GATE by C (trade B/A only when market favorable)
    if "C" in have and "B" in have:
        gate = P["C"] >= args.gate
        base = (P.get("A", P["B"]) >= thr) & (P["B"] >= thr) if "A" in have else (P["B"] >= thr)
        res.append(evaluate(f"GATE C>={args.gate} & B>={thr}", gate & (P["B"] >= thr), real, day, days))
        if "A" in have:
            res.append(evaluate(f"GATE C>={args.gate} & A&B>={thr}", gate & (P["A"] >= thr) & (P["B"] >= thr), real, day, days))

    res = [r for r in res if r]
    res.sort(key=lambda r: -r["win"])
    print(f"{'method':<26}{'n/day':>6}{'win':>8}{'avg%':>9}{'$/day':>8}{'worst':>7}{'grn':>6}")
    for r in res:
        print(f"{r['name']:<26}{r['nday']:>6}{r['win']:>8.3f}{r['avg']:>+9.4f}{r['dpd']:>+8.1f}{r['worst']:>+7.0f}{r['grn']:>6.2f}")

    # --- STACKING: honest time-split (first 4 days fit meta, last 3 eval) ---
    if len(have) >= 2:
        from sklearn.linear_model import LogisticRegression
        ud = sorted(m["day"].unique())
        sp = max(1, len(ud) * 4 // 7)
        tr_days = set(ud[:sp]); trm = m["day"].isin(tr_days).to_numpy(); evm = ~trm
        ed = m["day"].to_numpy()[evm]; er = real[evm]; edays = len(set(ed))
        y = (real > COST).astype(int)
        Xc = [f"p_{k}" for k in have]
        Xall = m[Xc].to_numpy()
        meta = LogisticRegression(max_iter=600, C=1.0).fit(Xall[trm], y[trm])
        ps = meta.predict_proba(Xall)[:, 1][evm]
        print(f"\n--- STACKING (meta on {Xc}, fit {len(tr_days)}d / eval {edays}d) ---")
        print(f"   meta coefs: {dict(zip(have, np.round(meta.coef_[0],2)))}")
        print(f"   {'cfg':<22}{'n/day':>6}{'win':>8}{'avg%':>9}")
        # eval stacking vs B-alone on the SAME eval days, matched-ish thresholds
        for q in (0.90, 0.95, 0.98):
            thr_s = np.quantile(ps, q)
            f = ps >= thr_s; pnl = er[f] - COST
            if f.sum() >= edays:
                print(f"   STACK top{int((1-q)*100)}%        {int(f.sum())//edays:>6}{(pnl>0).mean():>8.3f}{pnl.mean()*100:>+9.4f}")
        for thr_b in (0.85, 0.90):
            f = (m['p_B'].to_numpy()[evm] >= thr_b); pnl = er[f] - COST
            if f.sum() >= edays:
                print(f"   B alone @{thr_b}          {int(f.sum())//edays:>6}{(pnl>0).mean():>8.3f}{pnl.mean()*100:>+9.4f}")


if __name__ == "__main__":
    main()
