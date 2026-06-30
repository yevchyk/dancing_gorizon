"""Process the new (regression) data broadly: walk-forward OOS stats, daily
consistency, the predicted-value -> win-rate relationship, and a side-by-side
with the old classification models' probability -> win-rate.

Usage:
  python -m src.run_reg_walkforward --folds 4 --train-days 90 --test-days 14
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from .features import CurveBuilder
from .training.horizon_slicer import HorizonSlicer
from .training.reg_trainer import RegTrainer
from .walkforward.reg_walk_forward import RegWalkForward

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
HZ = [h.label for h in C.HORIZONS]


def ev_trades(stats: pd.DataFrame) -> pd.DataFrame:
    """One EV trade per (anchor, horizon) where |pred_ret| > fee."""
    rows = []
    for lab in HZ:
        p = stats[f"pred_ret_{lab}"].to_numpy()
        r = stats[f"real_ret_{lab}"].to_numpy()
        side = np.where(p > FEE, 1, np.where(p < -FEE, -1, 0))
        take = side != 0
        pnl = side[take] * r[take] - FEE
        sub = pd.DataFrame({"day": stats["day"].to_numpy()[take],
                            "horizon": lab, "pred": p[take], "side": side[take],
                            "pnl": pnl, "won": (pnl > 0).astype(int)})
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--test-days", type=int, default=14)
    args = ap.parse_args()

    master = pd.read_parquet(C.DATASETS_DIR / "master_reg.parquet")
    trainer = RegTrainer(HorizonSlicer(CurveBuilder(
        C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)))
    wf = RegWalkForward(trainer, train_days=args.train_days,
                        test_days=args.test_days, n_folds=args.folds)
    stats = wf.run(master)
    out = C.OUTPUTS_DIR / "analysis" / "reg_walkforward_stats.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(out, index=False)
    print(f"stats: {len(stats)} test anchors -> {out}\n")

    # 1. OOS correlation pred_ret vs real_ret
    print("=== 1. OOS corr (pred_ret vs real_ret) per horizon ===")
    for lab in HZ:
        c = np.corrcoef(stats[f"pred_ret_{lab}"], stats[f"real_ret_{lab}"])[0, 1]
        print(f"  {lab:>3}: corr={c:+.3f}")

    trades = ev_trades(stats)
    # 2. EV strategy per horizon + overall
    print("\n=== 2. EV-entry per horizon (long if pred>fee, short if <-fee) ===")
    for lab in HZ:
        g = trades[trades.horizon == lab]
        if len(g):
            print(f"  {lab:>3}: n={len(g):>5} win={g.won.mean():.3f} "
                  f"avg_pnl={g.pnl.mean()*100:+.4f}%")
    print(f"  ALL: n={len(trades)} win={trades.won.mean():.3f} "
          f"avg_pnl={trades.pnl.mean()*100:+.4f}%")

    # 3. daily consistency
    print("\n=== 3. DAILY consistency (EV strategy, all horizons) ===")
    daily = trades.groupby("day")["pnl"].mean() * 100
    green = int((daily > 0).sum())
    print(f"  days={len(daily)}  green={green}  red={len(daily)-green}  "
          f"worst={daily.min():+.3f}%  best={daily.max():+.3f}%  mean={daily.mean():+.4f}%")
    daily.round(4).to_csv(C.OUTPUTS_DIR / "analysis" / "reg_daily_pnl.csv")

    # 4. NEW: predicted-decile -> realized win-rate & pnl (1h as the workhorse)
    print("\n=== 4. NEW prediction->winrate (pred_ret decile -> realized) ===")
    for lab in ("5m", "1h"):
        p = stats[f"pred_ret_{lab}"]; r = stats[f"real_ret_{lab}"].to_numpy()
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        d = pd.DataFrame({"q": q, "r": r})
        agg = d.groupby("q")["r"].agg(winrate=lambda x: (x > 0).mean(), avg=lambda x: x.mean())
        print(f"  ret_{lab}: decile winrate  " +
              " ".join(f"{int(i)}:{row.winrate:.2f}" for i, row in agg.iterrows()))
        print(f"  ret_{lab}: decile avg_ret% " +
              " ".join(f"{int(i)}:{row.avg*100:+.2f}" for i, row in agg.iterrows()))

    # 5. OLD comparison: prob decile -> win-rate
    old_path = C.OUTPUTS_DIR / "analysis" / "walkforward_stats.parquet"
    if old_path.exists():
        old = pd.read_parquet(old_path)
        print("\n=== 5. OLD prob->winrate (classification, prob decile -> won) ===")
        for name in ("up_15m", "down_5m"):
            g = old[old.model == name]
            if g.empty:
                continue
            q = pd.qcut(g["prob"], 10, labels=False, duplicates="drop")
            agg = pd.DataFrame({"q": q, "won": g["won"].to_numpy()}).groupby("q")["won"].mean()
            print(f"  {name}: decile winrate  " +
                  " ".join(f"{int(i)}:{v:.2f}" for i, v in agg.items()))

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
