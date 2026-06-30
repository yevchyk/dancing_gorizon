"""Model registry: load + ensemble-score the OLD and NEW HC models."""
from __future__ import annotations

from src.run_hc_classic_sim import score_ensemble as _score_ensemble

from dh import config as cfg


def score(features, which: str = "new"):
    """Return `features` with up_prob/down_prob columns from the chosen model.

    which: 'new' (trained to 2026-06-04) or 'old' (trained to 2026-05-26).
    """
    if which not in cfg.MODELS:
        raise KeyError(f"unknown model '{which}', known: {list(cfg.MODELS)}")
    return _score_ensemble(features.copy(), cfg.MODELS[which])
