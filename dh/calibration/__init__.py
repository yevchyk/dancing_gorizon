"""Named calibration methods (see reports/HC_CALIBRATION_METHODS.md).

Each returns a tidy DataFrame: bin | n | win% | avg_net% .  "win" = net>0 after cost.
The validated edge lives in the high-conviction tail (prob >= ~0.90); below ~0.80 loses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROB_BINS = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.001]
SPREAD_BINS = [0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]


def _bin(df, score_col, net, bins):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df[score_col] >= lo) & (df[score_col] < hi)
        n = int(m.sum())
        rows.append({
            "bin": f"{lo:.2f}-{hi:.2f}", "n": n,
            "win%": round(float((net[m] > 0).mean() * 100), 1) if n else np.nan,
            "avg_net%": round(float(net[m].mean()), 2) if n else np.nan,
        })
    return pd.DataFrame(rows)


def c1_raw(df, side: str, cost: float):
    """C1 RAW: winrate vs raw up_prob (long) / down_prob (short), per row."""
    col = "up_prob" if side == "long" else "down_prob"
    sign = 1.0 if side == "long" else -1.0
    return _bin(df, col, sign * df["gross_move"] - cost, PROB_BINS)


def c3_spread(df, cost: float):
    """C3 SPREAD: winrate vs (up_prob - down_prob), long."""
    d = df.assign(_sp=df["up_prob"] - df["down_prob"])
    return _bin(d, "_sp", d["gross_move"] - cost, SPREAD_BINS)


def horizon_mean(df):
    """Collapse horizons -> one row per (symbol, base_time): mean prob/spread/move."""
    d = df.assign(_sp=df["up_prob"] - df["down_prob"])
    g = d.groupby(["symbol", "base_time"]).agg(
        mean_up=("up_prob", "mean"), mean_dn=("down_prob", "mean"),
        mean_spread=("_sp", "mean"), mean_move=("gross_move", "mean"),
        nH=("gross_move", "size")).reset_index()
    return g


def c6_horizon_mean(g, side: str, cost: float):
    """C6 HORIZON_MEAN: bin by cross-horizon mean conviction (denoised)."""
    col = "mean_up" if side == "long" else "mean_dn"
    sign = 1.0 if side == "long" else -1.0
    return _bin(g, col, sign * g["mean_move"] - cost, PROB_BINS)


def c7_spread_mean(g, cost: float):
    """C7 SPREAD_MEAN: bin by cross-horizon mean spread (cleanest, rare)."""
    return _bin(g, "mean_spread", g["mean_move"] - cost, SPREAD_BINS)
