"""Build the V4 dataset: ONE model, 1-minute horizons 2..120 (1-min targets).

  python -m src.run_hc_dataset_v4 --out-dir data/hc_min1_2to120/dataset \
      --universe configs/hc_universe_full.json --days 92 --stride-min 180 \
      --random-count 25 --fresh
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .hc import config as HC
from .hc import schema_v3 as S3
from .hc.data_v4 import DEFAULT_ANCHORS, build_dataset_shards_v4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--universe", type=Path, default=Path("configs/hc_universe_full.json"))
    ap.add_argument("--days", type=int, default=92)
    ap.add_argument("--stride-min", type=int, default=HC.SAMPLE_STRIDE_MIN)
    ap.add_argument("--entry-delay-min", type=int, default=HC.EXEC_ENTRY_DELAY_MIN)
    ap.add_argument("--hmin", type=int, default=2)
    ap.add_argument("--hmax", type=int, default=120)
    ap.add_argument("--random-count", type=int, default=25)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    candidates = tuple(range(a.hmin, a.hmax + 1))
    anchors = tuple(x for x in DEFAULT_ANCHORS if a.hmin <= x <= a.hmax)
    print(f"v4 dataset -> {a.out_dir} universe={a.universe.name} days={a.days} stride={a.stride_min}m "
          f"horizons={a.hmin}..{a.hmax}/1min (rand {a.random_count}/snap) feat_cols={len(S3.FEATURE_COLUMNS_V3)}",
          flush=True)
    s = build_dataset_shards_v4(out_dir=a.out_dir, universe_path=a.universe, anchors=anchors,
                                candidates=candidates, random_count=a.random_count,
                                stride_min=a.stride_min, days=a.days,
                                entry_delay_min=a.entry_delay_min, fresh=a.fresh)
    print(f"DONE rows={s['rows']} shards={s['shards']} window {s['valid_base_time_min']}..{s['valid_base_time_max']}",
          flush=True)


if __name__ == "__main__":
    main()
