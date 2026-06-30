"""Build horizon-conditioned per-symbol dataset shards.

Smoke first:
    python -m src.run_hc_dataset --smoke --fresh

Full:
    python -m src.run_hc_dataset --stage all --fresh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hc import config as HC
from .hc.data import build_dataset_shards, generate_universe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["universe", "dataset", "all"], default="all")
    ap.add_argument("--smoke", action="store_true", help="5 symbols, last 20 days, anchors only")
    ap.add_argument("--exec", action="store_true", help="leak-free executable target under data/hc_exec")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--universe", type=Path, default=HC.UNIVERSE_PATH)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--stride-min", type=int, default=HC.SAMPLE_STRIDE_MIN)
    ap.add_argument("--anchors-only", action="store_true")
    ap.add_argument("--random-count", type=int, default=HC.RANDOM_HORIZONS_PER_SNAPSHOT)
    ap.add_argument("--random-step-min", type=int, default=HC.RANDOM_HORIZON_STEP_MIN)
    ap.add_argument("--entry-delay-min", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.stage in {"universe", "all"}:
        payload = generate_universe(args.universe)
        print(f"universe -> {args.universe} symbols={payload['actual_count']}")

    if args.stage in {"dataset", "all"}:
        smoke = args.smoke
        out_dir = args.out_dir or (
            HC.SMOKE_DATASET_DIR if smoke else HC.EXEC_DATASET_DIR if args.exec else HC.DATASET_DIR
        )
        max_symbols = args.max_symbols
        days = args.days
        anchors_only = args.anchors_only
        random_count = args.random_count
        entry_delay_min = args.entry_delay_min
        if args.exec and entry_delay_min is None:
            entry_delay_min = HC.EXEC_ENTRY_DELAY_MIN
        if entry_delay_min is None:
            entry_delay_min = 0
        if smoke:
            max_symbols = 5 if max_symbols is None else max_symbols
            days = 20 if days is None else days
            anchors_only = True
            random_count = 0
        summary = build_dataset_shards(
            out_dir=out_dir,
            universe_path=args.universe,
            symbols=args.symbols,
            max_symbols=max_symbols,
            stride_min=args.stride_min,
            days=days,
            anchors_only=anchors_only,
            random_count=random_count,
            random_step_min=args.random_step_min,
            entry_delay_min=entry_delay_min,
            seed=args.seed,
            fresh=args.fresh,
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "stats"}, indent=2))


if __name__ == "__main__":
    main()
