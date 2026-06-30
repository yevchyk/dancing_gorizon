"""Find the balance for the risk-adjusted (#2) strategy: across selectivity
levels show trades/day, the PROBABILITY inside the picks (mean/median conf),
win-rate and avg PnL -- on the strict last-10-days.

Also a variant with a probability floor (rank by reward/risk but only among
trades whose direction prob >= floor), so it can't pick low-prob setups.

Usage:
  python -m src.run_riskadj_balance --slip 0.05
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0


def score_table(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["conf"] = np.maximum(s.p_up, s.p_down)
    s["side"] = np.where(s.p_up >= s.p_down, 1, -1)
    fav = np.where(s.side == 1, s.pred_mfe, -s.pred_mae)
    adv = np.where(s.side == 1, np.abs(s.pred_mae), s.pred_mfe)
    s["rr"] = np.clip(fav / (np.abs(adv) + 1e-4), 0, 5)
    s["score"] = s["conf"] * s["rr"]
    return s


def report(df: pd.DataFrame, n: int, cost: float, tag: str):
    g = df.nlargest(n, "score")
    pnl = g.side * g.real_ret - cost
    ndays = g.day.nunique()
    print(f"  {tag:<16} n={n:>4} ~{n/11:>4.1f}/day  conf:med={g.conf.median():.2f} "
          f"mean={g.conf.mean():.2f}  win={(pnl>0).mean():.3f}  "
          f"avg_pnl={pnl.mean()*100:+.4f}%  rr_med={g.rr.median():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip", type=float, default=0.05)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0

    s = score_table(pd.read_parquet(C.OUTPUTS_DIR / "analysis" / "engine_stats.parquet"))
    last = s[s.fold == s.fold.max()]
    last = last[pd.to_datetime(last.day) > pd.Timestamp("2026-05-20")]
    print(f"last-10-days: {len(last)} candidates, {last.day.nunique()} days, "
          f"cost={cost*100:.3f}%\n")

    print("=== RISK-ADJ #2: balance by selectivity (no prob floor) ===")
    for n in (55, 110, 220, 440, 880):
        report(last, n, cost, "top-score")

    print("\n=== RISK-ADJ + prob floor (rank by RR among conf>=floor) ===")
    for floor in (0.60, 0.65, 0.70):
        sub = last[last.conf >= floor]
        print(f"  -- floor {floor:.2f} ({len(sub)} candidates, ~{len(sub)/11:.0f}/day) --")
        for n in (55, 110, 220):
            if len(sub) >= n:
                report(sub, n, cost, f"floor{floor:.2f}")


if __name__ == "__main__":
    main()
