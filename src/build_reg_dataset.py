"""Attach the regression targets (ret/mfe/mae per horizon) to the existing
independent-anchor master, computed fast with numpy. Reuses the curve features
already in master_independent.parquet (no curve rebuild).

Usage:
  python -m src.build_reg_dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .database import CandleStore
from .trading.timeutil import index_to_ns, anchors_to_ns, NS_PER_MIN

HZ = [(h.label, h.minutes) for h in C.HORIZONS]


def _targets_for_symbol(ts, high, low, close, anchors_ns):
    """Return dict col -> array aligned with anchors_ns (NaN where no fwd data)."""
    n = len(anchors_ns)
    out = {f"{k}_{lab}": np.full(n, np.nan)
           for lab, _ in HZ for k in ("ret", "mfe", "mae")}
    for i, a_ns in enumerate(anchors_ns):
        ei = int(np.searchsorted(ts, a_ns, side="right")) - 1
        if ei < 0:
            continue
        entry = close[ei]
        if entry <= 0:
            continue
        for lab, mins in HZ:
            end = a_ns + mins * NS_PER_MIN
            fj = int(np.searchsorted(ts, end, side="right"))
            if fj <= ei + 1:
                continue
            h = high[ei + 1:fj]
            l = low[ei + 1:fj]
            c_end = close[fj - 1]
            out[f"ret_{lab}"][i] = c_end / entry - 1.0
            out[f"mfe_{lab}"][i] = h.max() / entry - 1.0
            out[f"mae_{lab}"][i] = l.min() / entry - 1.0
    return out


def main() -> None:
    master = pd.read_parquet(C.DATASETS_DIR / "master_independent.parquet")
    store = CandleStore(C.CANDLES_DIR)
    cols = [f"{k}_{lab}" for lab, _ in HZ for k in ("ret", "mfe", "mae")]
    for c in cols:
        master[c] = np.nan

    for symbol, g in master.groupby("symbol"):
        candles = store.load(symbol)
        if candles is None:
            continue
        ts = index_to_ns(candles.index)
        high, low, close = (candles[c].to_numpy(float) for c in ("high", "low", "close"))
        anchors_ns = anchors_to_ns(g["anchor_time"])
        tgt = _targets_for_symbol(ts, high, low, close, anchors_ns)
        for c in cols:
            master.loc[g.index, c] = tgt[c]

    before = len(master)
    master = master.dropna(subset=cols).reset_index(drop=True)
    out = C.DATASETS_DIR / "master_reg.parquet"
    master.to_parquet(out, index=False)
    print(f"master_reg: {len(master)}/{before} rows with targets -> {out}")
    # sanity: mean ret/mfe/mae per horizon
    for lab, _ in HZ:
        print(f"  {lab:>3}  ret={master[f'ret_{lab}'].mean():+.4f}  "
              f"mfe={master[f'mfe_{lab}'].mean():+.4f}  "
              f"mae={master[f'mae_{lab}'].mean():+.4f}")


if __name__ == "__main__":
    main()
