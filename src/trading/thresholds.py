"""Load per-model signal thresholds from the latest block-5 percentile run.

Picks each model's abs_threshold at a chosen top-percentile slice (default top
1%, the most confident calls). Falls back to an empty dict if no run exists, so
the SignalFilter simply uses its DEFAULT_SIGNAL_THRESHOLD.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config as C


def _latest_run(test_results_dir: Path) -> Path | None:
    # latest run that actually has a percentile summary (build-only holdout
    # runs don't), so threshold loading never silently falls back to default
    runs = sorted((p for p in test_results_dir.glob("run_*") if p.is_dir()),
                  reverse=True)
    for r in runs:
        if (r / "percentile_summary.csv").exists():
            return r
    return runs[0] if runs else None


def load_signal_thresholds(top_pct: float = 1.0,
                           test_results_dir: Path = C.TEST_RESULTS_DIR
                           ) -> dict[str, float]:
    run = _latest_run(test_results_dir)
    if run is None:
        return {}
    csv = run / "percentile_summary.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    sub = df[(df["top_pct"] == top_pct) & (df["kind"] == "direction")]
    return {r["model"]: float(r["abs_threshold"]) for _, r in sub.iterrows()}


def load_optimal_thresholds(path: Path) -> dict[str, float]:
    """Per-model PnL-optimal thresholds from a ThresholdOptimizer run
    (optimal_thresholds.csv)."""
    df = pd.read_csv(path)
    return {r["model"]: float(r["threshold"]) for _, r in df.iterrows()}
