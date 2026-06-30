"""TT Phase 1 CLI — train the curve model (MultiRMSE).

  # smoke (CPU, few iters):
  python -m src.run_tt_train --dataset-dir data/tt_smoke/dataset \
      --model-dir models/tt_smoke_curve --iterations 200 --depth 6 --task-type CPU --seeds 42
  # real:
  python -m src.run_tt_train --dataset-dir data/tt_curve/dataset \
      --model-dir models/tt_curve --iterations 4000 --depth 7 --seeds 42,7,123
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .tt.train_tt import train_curve


def _seeds(raw: str) -> list[int]:
    return [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/tt_curve/dataset"))
    ap.add_argument("--model-dir", type=Path, default=Path("models/tt_curve"))
    ap.add_argument("--seeds", default="42,7,123")
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--random-val", action="store_true")
    ap.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU",
                    help="GPU default (MultiRMSE is GPU-supported; 240-dim leaves are tiny). "
                         "Fall back to CPU only if the 4070 OOMs (lower --gpu-ram-part first).")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--learning-rate", type=float, default=0.045)
    ap.add_argument("--l2-leaf-reg", type=float, default=4.0)
    ap.add_argument("--border-count", type=int, default=32)
    ap.add_argument("--gpu-ram-part", type=float, default=0.85)
    ap.add_argument("--od-wait", type=int, default=300)
    ap.add_argument("--verbose", type=int, default=200)
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--no-scale", action="store_true",
                    help="center target per-node but DON'T divide by std (raw-ratio magnitude — TT v2)")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--embargo-min", type=int, default=None,
                    help="train/val gap; default = h_max + entry_delay (one row's target span)")
    ap.add_argument("--continue-from", type=Path, default=None,
                    help="continue training: existing model dir; adds --iterations MORE trees per seed on top "
                         "(reuses that model's standardizer). Run on the same/refreshed dataset.")
    a = ap.parse_args()

    train_curve(dataset_dir=a.dataset_dir, model_dir=a.model_dir, seeds=_seeds(a.seeds),
                val_fraction=a.val_fraction, random_val=a.random_val, task_type=a.task_type,
                devices=a.devices, iterations=a.iterations, depth=a.depth,
                learning_rate=a.learning_rate, l2_leaf_reg=a.l2_leaf_reg, border_count=a.border_count,
                gpu_ram_part=a.gpu_ram_part, od_wait=a.od_wait, verbose=a.verbose,
                no_early_stop=a.no_early_stop, sample_frac=a.sample_frac, embargo_min=a.embargo_min,
                no_scale=a.no_scale, continue_from=a.continue_from)


if __name__ == "__main__":
    main()
