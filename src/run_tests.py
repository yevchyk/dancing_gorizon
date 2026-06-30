"""Run the full test suite on the holdout window.

Usage:
  python -m src.run_tests
  python -m src.run_tests --coin-threshold 0.85 --anchors 60
"""

from __future__ import annotations

import argparse

from .training import ModelRegistry
from .testing import ModelTester


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coin-threshold", type=float, default=0.80)
    p.add_argument("--anchors", type=int, default=60, help="holdout anchors per symbol")
    args = p.parse_args()

    registry = ModelRegistry.load_default()
    print(f"loaded {len(registry.names)} models")
    ModelTester(registry, anchors_per_symbol=args.anchors).run(args.coin_threshold)


if __name__ == "__main__":
    main()
