"""Walk-forward statistics collection: honest out-of-sample validation by
retraining models on rolling cutoffs and testing on independent anchors.

Builds the next, properly-validated version of the model/strategy stats without
touching the production models the live loop is using.
"""

from .independent_sampler import IndependentAnchorSampler
from .walk_forward import WalkForward

__all__ = ["IndependentAnchorSampler", "WalkForward"]
