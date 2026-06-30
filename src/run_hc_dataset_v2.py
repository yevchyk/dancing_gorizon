"""Build a V2 band-schema dataset (no BTC + time-of-day features).

One dataset with the UNION of all band horizons; train each band by filtering
--horizon-min/--horizon-max in run_hc_prod_train.

  python -m src.run_hc_dataset_v2 --out-dir data/hc_bands_v2/dataset \
      --universe configs/hc_universe_full.json --days 21 --fresh
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .hc import config as HC
from .hc import schema_v2 as S2
from .hc.data_v2 import build_dataset_shards_v2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_full.json"))
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--stride-min", type=int, default=HC.SAMPLE_STRIDE_MIN)
    ap.add_argument("--entry-delay-min", type=int, default=HC.EXEC_ENTRY_DELAY_MIN)
    ap.add_argument("--horizons", default="union", help="'union' or comma-separated minutes")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    horizons = S2.union_horizons() if a.horizons == "union" else [int(x) for x in a.horizons.split(",")]
    print(f"v2 dataset -> {a.out_dir} universe={a.universe.name} days={a.days} "
          f"stride={a.stride_min}m horizons={len(horizons)} ({horizons[0]}..{horizons[-1]}) "
          f"feat_cols={len(S2.FEATURE_COLUMNS_V2)}", flush=True)
    s = build_dataset_shards_v2(out_dir=a.out_dir, universe_path=a.universe, horizons=horizons,
                                stride_min=a.stride_min, days=a.days,
                                entry_delay_min=a.entry_delay_min, fresh=a.fresh)
    print(f"DONE rows={s['rows']} shards={s['shards']} "
          f"window {s['valid_base_time_min']}..{s['valid_base_time_max']}", flush=True)


if __name__ == "__main__":
    main()
