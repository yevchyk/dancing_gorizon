"""Adapter to score the OLD ml_predictor models on the v2 holdout.

The old models use a different feature set (340 log windows of price+volume +
30 BTC windows = 710 cols) and different horizons (30/90/180/240m). This package
rebuilds those exact features from the v2 candle store so the legacy models can
be benchmarked through the same ExitSimulator PnL as the new ones.
"""

from .old_features import OldFeatureBuilder
from .old_scorer import LegacyModelGroup

__all__ = ["OldFeatureBuilder", "LegacyModelGroup"]
