"""Hybrid curve builder for short-horizon experiments."""

from __future__ import annotations

import numpy as np


class FastCurve:
    def __init__(
        self,
        points: int,
        min_step_min: float,
        max_depth_min: float,
        segments: tuple[tuple[float, float, int], ...] | None = None,
        offsets_min: tuple[float, ...] | None = None,
    ):
        self.points = points
        self.min_step_min = min_step_min
        self.max_depth_min = max_depth_min
        self.segments = segments
        if offsets_min is not None:               # explicit lookbacks (e.g. BTC context)
            self.offsets_min = np.asarray(offsets_min, dtype="float64")
            self.points = len(self.offsets_min)
        else:
            self.offsets_min = self._build_offsets()
        self.offsets_ns = np.round(self.offsets_min * 60_000_000_000).astype("int64")

    def _build_offsets(self) -> np.ndarray:
        if not self.segments:
            ratio = self.max_depth_min / self.min_step_min
            exponents = np.arange(self.points) / (self.points - 1)
            return self.min_step_min * (ratio ** exponents)

        pieces = []
        for i, (start, end, count) in enumerate(self.segments):
            if count <= 0:
                continue
            raw_count = count if i == 0 else count + 1
            part = np.geomspace(float(start), float(end), raw_count)
            if i > 0:
                part = part[1:]
            pieces.append(part)
        offsets = np.concatenate(pieces) if pieces else np.array([], dtype="float64")
        if len(offsets) != self.points:
            raise ValueError(f"hybrid curve produced {len(offsets)} points, expected {self.points}")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("hybrid curve offsets must be strictly increasing")
        return offsets

    def columns(self) -> list[str]:
        return [f"p_{i:03d}" for i in range(self.points)]

    def columns_for_lookback(self, lookback_min: float) -> list[str]:
        cols = self.columns()
        return [cols[i] for i, off in enumerate(self.offsets_min) if off <= lookback_min]

    def build_matrix(self, ts_ns: np.ndarray, close: np.ndarray,
                     anchors_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (features, valid_mask) for anchors.

        Features are close(anchor-offset)/close(anchor), with every sampled point
        strictly at or before the requested historical timestamp.
        """
        entry_idx = np.searchsorted(ts_ns, anchors_ns, side="right") - 1
        valid = (entry_idx >= 0) & np.isfinite(close[np.clip(entry_idx, 0, len(close) - 1)])
        entry = close[np.clip(entry_idx, 0, len(close) - 1)]
        valid &= entry > 0

        sample_times = anchors_ns[:, None] - self.offsets_ns[None, :]
        sample_idx = np.searchsorted(ts_ns, sample_times, side="right") - 1
        valid &= (sample_idx.min(axis=1) >= 0)
        sample_idx = np.clip(sample_idx, 0, len(close) - 1)
        feats = close[sample_idx] / entry[:, None]
        valid &= np.isfinite(feats).all(axis=1)
        return feats.astype("float32"), valid
