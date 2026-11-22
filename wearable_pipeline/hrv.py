# SPDX-License-Identifier: GPL-3.0-only
"""RR intervals to cleaned RMSSD, SDNN, pNN50 and LF/HF."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LF_BAND, HF_BAND = (0.04, 0.15), (0.15, 0.40)
MIN_FREQ_WINDOW_S = 60.0


def read_rr(path: str | Path) -> pd.DataFrame:
    """Polar Sensor Logger RR export: phone timestamp; RR interval in ms."""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    ts = next(c for c in df.columns if "timestamp" in c.lower())
    rr = next(c for c in df.columns if "rr" in c.lower())
    out = pd.DataFrame({"phone_ts": df[ts].astype(str).str.strip(),
                        "rr_ms": pd.to_numeric(df[rr], errors="coerce")})
    return out.dropna(subset=["rr_ms"]).reset_index(drop=True)


def clean_mask(rr_ms, min_ms: float = 300, max_ms: float = 2000, ectopic_frac: float = 0.20,
               reseed_after: int | None = 5) -> np.ndarray:
    """Beats to keep: range, then the 20% rule against the previous kept beat.

    That rule alone never recovers from a true step in heart rate, or from a bad
    first beat. After reseed_after consecutive rejected beats that agree with
    each other, they are accepted as the new level. None disables this.
    """
    rr = np.asarray(rr_ms, float)
    keep = np.isfinite(rr) & (rr >= min_ms) & (rr <= max_ms)
    previous = None
    if reseed_after and keep.any():  # so that one bad first beat cannot set the level
        previous = float(np.median(rr[keep][:reseed_after]))
    run: list[int] = []  # consecutive rejected beats that agree with each other
    for i in range(len(rr)):
        if not keep[i]:
            run = []
            continue
        if previous is None or abs(rr[i] - previous) <= ectopic_frac * previous:
            previous, run = rr[i], []
            continue
        keep[i] = False
        if run and abs(rr[i] - rr[run[-1]]) > ectopic_frac * rr[run[-1]]:
            run = []
        run.append(i)
        if reseed_after and len(run) >= reseed_after:
            keep[run] = True
            previous, run = rr[i], []
    return keep


def successive_diffs(rr_ms, keep=None, t_s=None, max_gap_s: float = 3.0,
                     ectopic_frac: float = 0.20) -> np.ndarray:
    """Differences of true successive pairs.

    Never across a removed beat, a dropout, or the edge of a re-seeded level.
    """
    rr = np.asarray(rr_ms, float)
    pair = np.ones(max(len(rr) - 1, 0), bool)
    if keep is not None:
        keep = np.asarray(keep, bool)
        pair &= keep[:-1] & keep[1:]
        pair &= np.abs(np.diff(rr)) <= ectopic_frac * rr[:-1]
    if t_s is not None:
        step = np.diff(np.asarray(t_s, float))
        pair &= (step >= 0) & (step <= max_gap_s)
    return np.diff(rr)[pair]


def rmssd(rr_ms, keep=None, t_s=None) -> float:
    d = successive_diffs(rr_ms, keep, t_s)
    return float(np.sqrt(np.mean(d * d))) if d.size else float("nan")


def sdnn(rr_ms, keep=None) -> float:
    rr = np.asarray(rr_ms, float)
    kept = rr if keep is None else rr[keep]
    return float(np.std(kept, ddof=1)) if kept.size > 1 else float("nan")


def pnn50(rr_ms, keep=None, t_s=None) -> float:
    d = np.abs(successive_diffs(rr_ms, keep, t_s))
    return float(100 * np.mean(d > 50)) if d.size else float("nan")


def _centres(t, window_s, step_s):
    if not t.size or t[-1] - t[0] < window_s:
        return np.array([])
    return np.arange(t[0] + window_s / 2, t[-1] - window_s / 2 + step_s / 2, step_s)


def rolling(t_s, rr_ms, window_s: float = 60, step_s: float = 5, keep=None,
            min_beats: int = 5) -> pd.DataFrame:
    """Time-domain HRV in sliding windows. Sparse windows are NaN."""
    t, rr = np.asarray(t_s, float), np.asarray(rr_ms, float)
    keep = clean_mask(rr) if keep is None else np.asarray(keep, bool)
    rows = []
    for c in _centres(t, window_s, step_s):
        a, b = np.searchsorted(t, [c - window_s / 2, c + window_s / 2])
        r, k, when = rr[a:b], keep[a:b], t[a:b]
        n = int(k.sum())
        if n >= min_beats:
            rows.append((c, rmssd(r, k, when), sdnn(r, k), 60000 / r[k].mean(),
                         pnn50(r, k, when), n))
        else:
            rows.append((c, *[np.nan] * 4, n))
    return pd.DataFrame(rows, columns=["t_s", "rmssd", "sdnn", "mean_hr", "pnn50", "n_beats"])


def lf_hf(t_s, rr_ms) -> tuple[float, float]:
    """LF and HF power (ms^2), Welch on a 4 Hz tachogram."""
    from scipy.signal import welch

    t, rr = np.asarray(t_s, float), np.asarray(rr_ms, float)
    if len(rr) < 20 or t[-1] - t[0] < MIN_FREQ_WINDOW_S:
        return float("nan"), float("nan")
    x = np.interp(np.arange(t[0], t[-1], 0.25), t, rr)
    f, p = welch(x - x.mean(), fs=4, nperseg=min(len(x), 256))

    def power(band):
        m = (f >= band[0]) & (f < band[1])
        return float(np.trapezoid(p[m], f[m]))
    return power(LF_BAND), power(HF_BAND)


def rolling_freq(t_s, rr_ms, window_s: float = 120, step_s: float = 5, keep=None) -> pd.DataFrame:
    t, rr = np.asarray(t_s, float), np.asarray(rr_ms, float)
    keep = clean_mask(rr) if keep is None else np.asarray(keep, bool)
    rows = []
    for c in _centres(t, window_s, step_s):
        m = (t >= c - window_s / 2) & (t <= c + window_s / 2) & keep
        lf, hf = lf_hf(t[m], rr[m])
        rows.append((c, lf, hf, lf / hf if hf > 0 else np.nan))
    return pd.DataFrame(rows, columns=["t_s", "lf", "hf", "lf_hf"])
