# SPDX-License-Identifier: GPL-3.0-only
# Developed with reference to https://github.com/ElioS-S/pupil-size (GPL-3.0). Modified 2026: see NOTICE.
"""Pupil preprocessing after Kret & Sjak-Shie (2019), each eye on its own.

Per eye, in this order:
  1. range gate: 1.5-9 mm (1-9 mm as a sensitivity analysis)
  2. blinks plus 500 ms of recovery, and off-head samples, become missing
  3. dilation-speed filter on raw samples: median + 16 x MAD
  4. gaps over 75 ms are padded by 50 ms on each side
  5. gaps up to 250 ms are interpolated linearly
  6. zero-phase 4 Hz low pass within each continuous stretch
Then the eyes are combined: the mean where both are valid, otherwise the one
eye, shifted by half the mean between-eye difference onto the mean's scale.

Kret, M. E., & Sjak-Shie, E. E. (2019). Preprocessing pupil size data:
Guidelines and code. Behavior Research Methods, 51, 1336-1342.
"""
from __future__ import annotations

import numpy as np

GATE_MM = (1.5, 9.0)
GATE_SENSITIVITY_MM = (1.0, 9.0)
POST_BLINK_MS = 500.0
MAD_N = 16
GAP_MIN_MS, GAP_PAD_MS = 75.0, 50.0
INTERPOLATE_MAX_MS = 250.0
LOWPASS_HZ, LOWPASS_ORDER = 4.0, 4


def _runs(mask) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, np.int8), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def clean_eye(t_ns, diameter_mm, blinks_ns=(), worn=None, gate=GATE_MM,
              post_blink_ms: float = POST_BLINK_MS) -> tuple[np.ndarray, dict]:
    """(cleaned signal, what each step removed). Times are integer nanoseconds."""
    from scipy.signal import butter, filtfilt

    t = np.asarray(t_ns, np.int64)
    d = np.array(diameter_mm, float)
    n = d.size
    if t.shape != d.shape or n < 2 or np.any(np.diff(t) < 0):
        raise ValueError("need sorted timestamps, one per sample")
    fs = 1e9 / np.median(np.diff(t))
    if fs <= 2 * LOWPASS_HZ:
        raise ValueError("sampling rate is too low for the 4 Hz filter")
    report: dict = dict(n_samples=n)

    out_of_range = np.isfinite(d) & ((d < gate[0]) | (d > gate[1]))
    d[out_of_range] = np.nan
    report["removed_gate"] = int(out_of_range.sum())

    removed = 0
    recovery = int(round(post_blink_ms * 1e6))
    for start, end in blinks_ns:
        a, b = np.searchsorted(t, [int(start), int(end) + recovery])
        removed += int(np.isfinite(d[a:b]).sum())
        d[a:b] = np.nan
    report["removed_blink"] = removed

    removed = 0
    if worn is not None:
        worn = np.asarray(worn, bool)
        if worn.shape != t.shape:
            raise ValueError("worn must be sampled on the pupil timestamps")
        off = ~worn & np.isfinite(d)
        d[off] = np.nan
        removed = int(off.sum())
    report["removed_not_worn"] = removed

    # On raw samples: a low pass first would hide the jumps this looks for.
    dt = np.diff(t) / 1e9
    with np.errstate(invalid="ignore"):
        step = np.abs(np.diff(d)) / np.where(dt > 0, dt, np.nan)
    speed = np.full(n, np.nan)
    speed[1:] = step
    speed[:-1] = np.fmax(speed[:-1], step)
    known = np.isfinite(speed)
    report["removed_speed"], report["speed_threshold_mm_s"] = 0, None
    if known.sum() > 10:
        median = float(np.median(speed[known]))
        threshold = median + MAD_N * float(np.median(np.abs(speed[known] - median)))
        fast = known & (speed > threshold) & np.isfinite(d)
        d[fast] = np.nan
        report["removed_speed"], report["speed_threshold_mm_s"] = int(fast.sum()), threshold

    removed = 0
    pad = int(GAP_PAD_MS * 1e6)
    for a, b in _runs(~np.isfinite(d)):
        last = min(b, n - 1)
        if (t[last] - t[a]) / 1e6 <= GAP_MIN_MS:
            continue
        low, high = np.searchsorted(t, t[a] - pad), np.searchsorted(t, t[last] + pad)
        removed += int(np.isfinite(d[low:a]).sum() + np.isfinite(d[b:high]).sum())
        d[low:a] = np.nan
        d[b:high] = np.nan
    report["removed_gap_padding"] = removed

    filled = 0
    if np.isfinite(d).sum() > 2:
        for a, b in _runs(~np.isfinite(d)):
            if a and b < n and (t[b] - t[a - 1]) / 1e6 <= INTERPOLATE_MAX_MS:
                d[a:b] = np.interp(t[a:b], [t[a - 1], t[b]], [d[a - 1], d[b]])
                filled += b - a
    report["interpolated"] = filled

    num, den = butter(LOWPASS_ORDER, LOWPASS_HZ / (fs / 2), btype="low")
    for a, b in _runs(np.isfinite(d)):
        if b - a > 3 * (LOWPASS_ORDER + 1) + 10:  # shorter than this, left unsmoothed
            d[a:b] = filtfilt(num, den, d[a:b])
    report["valid_frac"] = float(np.isfinite(d).mean())
    return d, report


def combine_eyes(left, right) -> tuple[np.ndarray, dict]:
    left, right = np.asarray(left, float), np.asarray(right, float)
    has_left, has_right = np.isfinite(left), np.isfinite(right)
    both = has_left & has_right
    difference = float(np.mean(left[both] - right[both])) if both.any() else 0.0
    out = np.full(left.shape, np.nan)
    out[both] = (left[both] + right[both]) / 2
    only_left, only_right = has_left & ~has_right, has_right & ~has_left
    out[only_left] = left[only_left] - difference / 2
    out[only_right] = right[only_right] + difference / 2
    return out, dict(between_eye_difference_mm=difference, n_both=int(both.sum()),
                     n_left_only=int(only_left.sum()), n_right_only=int(only_right.sum()))


def clean(t, left_mm, right_mm, blinks=(), worn=None, gate=GATE_MM,
          post_blink_ms: float = POST_BLINK_MS) -> tuple[np.ndarray, dict]:
    """Both eyes, then combined. t and the blink (start, end) pairs are seconds."""
    t_ns = np.round(np.asarray(t, float) * 1e9).astype(np.int64)
    spans = [(round(a * 1e9), round(b * 1e9)) for a, b in blinks]
    left, left_report = clean_eye(t_ns, left_mm, spans, worn, gate, post_blink_ms)
    right, right_report = clean_eye(t_ns, right_mm, spans, worn, gate, post_blink_ms)
    out, eyes = combine_eyes(left, right)
    return out, dict(left=left_report, right=right_report, combined=eyes)
