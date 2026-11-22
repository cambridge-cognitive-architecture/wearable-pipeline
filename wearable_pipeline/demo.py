# SPDX-License-Identifier: GPL-3.0-only
"""Three simulated devices with three clocks, put on one."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from . import align, hrv, pupil
from .timebase import NS, UTC, Clock

DURATION_S = 480


def simulate(seed: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    start = dt.datetime(2000, 1, 1, 12, tzinfo=UTC)
    utc_offset_s, logger_lag_s = 3600, 2.4
    local = start + dt.timedelta(seconds=utc_offset_s)
    info = dict(start_time=int(start.timestamp()) * NS, duration=DURATION_S * NS,
                template_data=dict(recording_name=local.strftime("%Y-%m-%d_%H:%M:%S")))

    t = np.arange(0, DURATION_S, 1 / 50)
    blinks = [(b, b + 0.2) for b in np.arange(4, DURATION_S - 4, 7.3)]
    base = 3.6 + 0.3 * np.sin(t / 30) + 0.35 * ((t > 200) & (t < 280))
    left, right = base + rng.normal(0, 0.03, t.size), base + 0.05 + rng.normal(0, 0.03, t.size)
    for a, b in blinks:
        left[(t >= a) & (t < b)] = right[(t >= a) & (t < b)] = 0.4
    worn = ~((t > 350) & (t < 370))

    beats, intervals = [], []
    now = 1.0
    while now < DURATION_S - 2:
        mean = 620 if 200 < now < 280 else 820
        rr = mean + 25 * np.sin(2 * np.pi * 0.25 * now) + rng.normal(0, 12)
        now += rr / 1000
        beats.append((local + dt.timedelta(seconds=now)).replace(tzinfo=None)
                     .isoformat(timespec="milliseconds"))
        intervals.append(rr * (1.6 if rng.random() < 0.015 else 1.0))

    def logged(s):
        stamp = start + dt.timedelta(seconds=s - logger_lag_s)
        return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    sync_true = np.array([10.0, 240.0, 470.0])
    return dict(info=info, t=t, left=left, right=right, blinks=blinks, worn=worn,
                beats=beats, rr=np.array(intervals),
                sync_ref=sync_true, sync_logged=[logged(s) for s in sync_true],
                task_logged=[logged(s) for s in (200.0, 280.0)])


def process(d: dict) -> dict:
    clock = Clock.from_info(d["info"])
    fit = align.fit_sync(clock.iso_to_rel(d["sync_logged"]), d["sync_ref"])
    beats = clock.local_to_rel(d["beats"])
    keep = hrv.clean_mask(d["rr"])
    grid = np.arange(*align.overlap_window(d["t"], beats), 0.1)
    cleaned, report = pupil.clean(d["t"], d["left"], d["right"], d["blinks"], d["worn"])
    return dict(fit=fit, beats=beats, keep=keep, grid=grid,
                pupil=align.bin_mean(d["t"], cleaned, grid, 0.1),
                pupil_report=report,
                hrv=hrv.rolling(beats, d["rr"], keep=keep),
                task=clock.iso_to_rel(d["task_logged"]) + fit.offset_s)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/demo.png"))
    a = ap.parse_args()
    d = simulate()
    r = process(d)
    fig, ax = plt.subplots(4, 1, figsize=(9, 6), sharex=True, layout="constrained")
    ax[0].plot(d["t"], d["left"], color="#b9bec7", lw=0.5)
    ax[0].set_ylabel("pupil, raw\n(mm)")
    ax[1].plot(r["grid"], r["pupil"], color="#2f6db5", lw=0.8)
    ax[1].set_ylabel("pupil, clean\n(mm)")
    ax[2].plot(r["beats"][~r["keep"]], d["rr"][~r["keep"]], "x", color="#999", label="rejected")
    ax[2].plot(r["beats"][r["keep"]], d["rr"][r["keep"]], ".", color="#c0392b", ms=1.5, label="kept")
    ax[2].set_ylabel("RR interval\n(ms)")
    ax[2].legend(loc="upper left", frameon=False, ncols=2)
    ax[3].plot(r["hrv"].t_s, r["hrv"].rmssd, color="#8e44ad")
    ax[3].set_ylabel("RMSSD, 60 s\n(ms)")
    for x in ax:
        x.axvspan(*r["task"], color="#f6c177", alpha=0.25, lw=0)
        x.spines[["top", "right"]].set_visible(False)
    ax[0].set_title(f"Simulated signals on one clock (logger moved by {r['fit'].offset_s:+.2f} s; "
                    "shaded = logged task)", loc="left", fontsize=10)
    ax[3].set_xlabel("seconds from eye-tracker start")
    ax[3].set_xlim(0, DURATION_S)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120, facecolor="white", metadata={"Software": None})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
