"""Orchestrates a full test run on the held-out window (last 10 days the models
never saw): build holdout -> score once -> threshold/top/deceptive -> report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config as C
from ..database import CandleStore
from ..features import CurveBuilder
from ..dataset import AnchorSampler, TargetBuilder, DatasetCollector
from ..training import ModelRegistry
from .threshold_analyzer import ThresholdAnalyzer
from .percentile_analyzer import PercentileThresholdAnalyzer
from .top_coins import TopCoinsAnalyzer
from .deceptive_coins import DeceptiveCoinsAnalyzer
from .winrate_report import WinRateReport


class ModelTester:
    def __init__(self, registry: ModelRegistry, holdout_days: int = None,
                 anchors_per_symbol: int = 60):
        self.registry = registry
        self.holdout_days = holdout_days or C.HOLDOUT_DAYS
        self.anchors_per_symbol = anchors_per_symbol

    def build_holdout(self, out_path: Path) -> pd.DataFrame:
        """Anchors strictly inside the holdout window [now-10d, now-2h]."""
        store = CandleStore(C.CANDLES_DIR)
        curve = CurveBuilder(C.CURVE_POINTS, C.CURVE_MIN_STEP_MIN, C.CURVE_MAX_DEPTH_MIN)
        sampler = AnchorSampler(self.anchors_per_symbol,
                                start_offset_days=self.holdout_days,
                                end_offset_days=0)
        collector = DatasetCollector(store, curve, sampler, TargetBuilder(),
                                     C.CHUNKS_DIR / "_holdout")
        collector.collect(store.symbols(), out_path)
        return pd.read_parquet(out_path)

    def run(self, threshold_for_coins: float = 0.80) -> Path:
        holdout_path = C.DATASETS_DIR / "holdout.parquet"
        holdout = (pd.read_parquet(holdout_path) if holdout_path.exists()
                   else self.build_holdout(holdout_path))

        scored = self.registry.score(holdout)
        summary = ThresholdAnalyzer(self.registry).analyze(scored)
        pct_summary = PercentileThresholdAnalyzer(self.registry).analyze(scored)
        top = TopCoinsAnalyzer(self.registry).analyze(scored, threshold_for_coins)
        deceptive = DeceptiveCoinsAnalyzer(self.registry).analyze(scored, threshold_for_coins)

        report = WinRateReport(C.TEST_RESULTS_DIR)
        run_dir = report.new_run_dir()
        report.write(run_dir, scored=scored, threshold_summary=summary,
                     percentile_summary=pct_summary,
                     top_coins=top, deceptive_coins=deceptive)
        print(f"test run -> {run_dir}  ({len(holdout)} holdout rows)")
        return run_dir
