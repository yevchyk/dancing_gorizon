"""Build a V3 dataset (adds 1-minute timeframe) — for the fast band-A scalper.

  python -m src.run_hc_dataset_v3 --out-dir data/hc_bandA_v3/dataset \
      --universe configs/hc_universe_full.json --days 21 --band A --fresh
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .hc import config as HC
from .hc import schema_v2 as S2
from .hc import schema_v3 as S3
from .hc.data_v3 import build_dataset_shards_v3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_full.json"))
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--stride-min", type=int, default=HC.SAMPLE_STRIDE_MIN)
    ap.add_argument("--entry-delay-min", type=int, default=HC.EXEC_ENTRY_DELAY_MIN)
    ap.add_argument("--band", default="A", help="A/B/C (default A) or 'union'")
    ap.add_argument("--horizons", default=None, help="comma list overrides --band")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    if a.horizons:
        horizons = [int(x) for x in a.horizons.split(",")]
    elif a.band == "union":
        horizons = S2.union_horizons()
    else:
        horizons = S2.band_horizons(a.band)

    print(f"v3 dataset -> {a.out_dir} universe={a.universe.name} days={a.days} stride={a.stride_min}m "
          f"band={a.band} horizons={len(horizons)} ({horizons[0]}..{horizons[-1]}) "
          f"feat_cols={len(S3.FEATURE_COLUMNS_V3)}", flush=True)
    s = build_dataset_shards_v3(out_dir=a.out_dir, universe_path=a.universe, horizons=horizons,
                                stride_min=a.stride_min, days=a.days,
                                entry_delay_min=a.entry_delay_min, fresh=a.fresh)
    print(f"DONE rows={s['rows']} shards={s['shards']} window {s['valid_base_time_min']}..{s['valid_base_time_max']}",
          flush=True)


if __name__ == "__main__":
    main()
