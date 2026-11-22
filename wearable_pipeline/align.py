# SPDX-License-Identifier: GPL-3.0-only
"""Sync offsets, overlap and resampling on the shared clock."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyncFit:
    offset_s: float  # add to the stream's timestamps
    residual_s: np.ndarray
    n: int


def fit_sync(stream_s, reference_s, max_residual_s: float = 1.0) -> SyncFit:
    """A constant offset, the median over paired sync checks. No drift is fitted."""
    s, r = np.asarray(stream_s, float), np.asarray(reference_s, float)
    if s.shape != r.shape or not s.size:
        raise ValueError("need the same non-zero number of sync times in both streams")
    offset = float(np.median(r - s))
    residual = r - (s + offset)
    if not np.all(np.abs(residual) <= max_residual_s):
        raise ValueError("sync checks disagree: wrong pairing, or a clock jumped")
    return SyncFit(offset, residual, s.size)


def overlap_window(*streams) -> tuple[float, float]:
    """The span every stream covers. Raises when there is none."""
    spans = []
    for t in streams:
        t = np.asarray(t, float)
        t = t[np.isfinite(t)]
        if not t.size:
            raise ValueError("a stream has no valid timestamps")
        spans.append((t.min(), t.max()))
    if not spans:
        raise ValueError("need at least one stream")
    lo, hi = max(a for a, _ in spans), min(b for _, b in spans)
    if hi < lo:
        raise ValueError("streams do not overlap")
    return float(lo), float(hi)


def bin_mean(t, values, grid, step: float) -> np.ndarray:
    """Mean in [grid, grid + step). A gap stays NaN and is never interpolated."""
    t, v, grid = (np.asarray(a, float) for a in (t, values, grid))
    if np.any(np.diff(t) < 0):
        raise ValueError("timestamps must be sorted")
    ok = np.isfinite(v)
    n = np.r_[0, np.cumsum(ok)]
    total = np.r_[0.0, np.cumsum(np.where(ok, v, 0.0))]
    lo, hi = np.searchsorted(t, grid), np.searchsorted(t, grid + step)
    count = n[hi] - n[lo]
    return np.where(count > 0, (total[hi] - total[lo]) / np.maximum(count, 1), np.nan)
