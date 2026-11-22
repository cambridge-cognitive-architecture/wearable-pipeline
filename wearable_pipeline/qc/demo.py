# SPDX-License-Identifier: GPL-3.0-only
"""A simulated session with deliberate faults: python -m wearable_pipeline.qc.demo"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .checks import run_qc
from .report import write_report
from .session import Event, Session, Stream, SyncCheck

DURATION_S = 5400.0
FAULTS = {
    "gaze_drops": "8% of gaze samples dropped, so the rate falls below 190 Hz",
    "not_worn": "glasses off for 7 minutes",
    "imu_gap": "a 3 s hole in the IMU",
    "video_rollover": "4 s lost where the scene video started a new file",
    "sync_outside": "the second sync flash is found 7 s after the tap",
    "heart_stopped_early": "the heart recorder stopped 90 s early",
    "break_unclosed": "a break near the end of work with no return",
}


def simulate(faults=tuple(FAULTS), seed: int = 4) -> Session:
    unknown = set(faults) - set(FAULTS)
    if unknown:
        raise ValueError(f"unknown faults {sorted(unknown)}")
    rng = np.random.default_rng(seed)

    def ticks(hz, start=0.2, end=DURATION_S - 0.2):
        return np.arange(round((end - start) * hz)) / hz + start

    t = ticks(200.0)
    if "gaze_drops" in faults:
        t = t[rng.random(t.size) > 0.08]
    imu = ticks(110.0)
    if "imu_gap" in faults:
        imu = imu[(imu < 2000) | (imu > 2003)]
    scene, parts = ticks(30.0), ()
    if "video_rollover" in faults:
        scene = scene[(scene < 2700) | (scene > 2704)]
        parts = (int(np.searchsorted(scene, 2702)),)
    stop = DURATION_S - 90 if "heart_stopped_early" in faults else DURATION_S + 20

    worn = np.ones(t.size, bool)
    if "not_worn" in faults:
        worn &= (t < 3000) | (t > 3420)
    blinks = [(float(b), float(b) + 0.2) for b in np.arange(4.0, DURATION_S - 4, 4.3)]
    pupil = 3.5 + 0.2 * np.sin(t / 40)
    closed = np.zeros(t.size, bool)
    for a, b in blinks:
        closed[np.searchsorted(t, a):np.searchsorted(t, b)] = True
    left = np.where(closed, 0.5, pupil + rng.normal(0, 0.03, t.size))
    right = np.where(closed, 0.5, pupil + rng.normal(0, 0.03, t.size))

    streams = dict(
        gaze=Stream(t), eye_state=Stream(t), imu=Stream(imu),
        scene_video=Stream(scene, parts=parts), eye_video=Stream(ticks(200.0)),
        ecg=Stream(ticks(130.0, -20.0, stop)), acc=Stream(ticks(50.0, -20.0, stop)),
        blinks=Stream(np.array([a for a, _ in blinks]), event=True))
    second = 7.0 if "sync_outside" in faults else 0.31
    syncs = [SyncCheck(20.0, 20.27), SyncCheck(5200.0, 5200.0 + second)]
    events = [Event(5, "session_start"), Event(30, "phase", "calibration"),
              Event(90, "phase", "settling"), Event(400, "phase", "work"),
              Event(1800, "leave"), Event(2100, "return"),
              Event(5150, "phase", "closing"), Event(5300, "session_end")]
    if "break_unclosed" in faults:
        events.append(Event(5000, "leave"))
    return Session(DURATION_S, streams, np.full(t.size, 800.0), np.full(t.size, 600.0), worn,
                   left, right, blinks, syncs, events, marker_s=0.8, closing_phase="closing")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/qc"))
    ap.add_argument("--clean", action="store_true", help="simulate without any fault")
    a = ap.parse_args()
    report = run_qc(simulate(() if a.clean else tuple(FAULTS)))
    for path in write_report(report, a.out):
        print(f"wrote {path}")
    print(report["verdict"], report["counts"])


if __name__ == "__main__":
    main()
