"""Dancing Horizon - horizon-conditioned crypto signal system (clean layer).

Public modules:
  dh.data        - candle store (race-safe) + feature building + universe
  dh.models      - load/score the OLD and NEW HC models
  dh.sim         - executable outcomes + mature window helpers
  dh.calibration - named calibration methods (C1..C7)
  dh.report      - `python -m dh.report.stats --date YYYY-MM-DD` : full stats for a period
"""
from . import config  # noqa: F401
