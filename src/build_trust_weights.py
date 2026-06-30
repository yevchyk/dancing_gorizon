"""Compute per-model trust weights from the OOF engine stats and save them for
the live engine. Trust = realized avg PnL when the model fires (prob>=floor),
required positive on BOTH the overall window and the recent ~14d (stability).
Low/unstable models get weight 0 -> they drop out automatically. Re-running this
after a retrain refreshes the weights, so models rejoin on their own.

Usage:
  python -m src.build_trust_weights --floor 0.60
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C

FEE = 2 * C.OKX_FEE_PER_SIDE / 100.0
MODELS = [f"{k}_{h.label}" for h in C.HORIZONS for k in ("up", "down")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.60)
    ap.add_argument("--slip", type=float, default=0.05)
    args = ap.parse_args()
    cost = FEE + args.slip / 100.0

    s = pd.read_parquet(C.OUTPUTS_DIR / "analysis" / "engine_stats.parquet")
    rows = []
    for h in C.HORIZONS:
        sub = s[s.horizon == h.label]
        for k, prob, side in (("up", sub.p_up, 1), ("down", sub.p_down, -1)):
            pnl = side * sub.real_ret.to_numpy() - cost
            rows.append(pd.DataFrame({"model": f"{k}_{h.label}", "day": sub.day.to_numpy(),
                                      "prob": prob.to_numpy(), "pnl": pnl}))
    cand = pd.concat(rows, ignore_index=True)
    fire = cand[cand.prob >= args.floor]
    rc = pd.to_datetime(fire.day).max() - pd.Timedelta(days=14)

    weights, table = {}, []
    for m in MODELS:
        g = fire[fire.model == m]
        gr = g[pd.to_datetime(g.day) > rc]
        t_all = g.pnl.mean() if len(g) else -1
        t_rec = gr.pnl.mean() if len(gr) >= 8 else -1
        w = float(min(t_all, t_rec)) if (t_all > 0 and t_rec > 0) else 0.0
        weights[m] = round(w, 6)
        table.append({"model": m, "trust_all": round(t_all * 100, 4),
                      "trust_recent": round(t_rec * 100, 4), "weight": round(w, 6)})

    out = {"floor": args.floor, "cost": cost, "weights": weights}
    (C.MODELS_DIR / "trust_weights.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(table).to_string(index=False))
    print(f"\ntrusted: {sum(w>0 for w in weights.values())}/{len(weights)} "
          f"-> {C.MODELS_DIR/'trust_weights.json'}")


if __name__ == "__main__":
    main()
