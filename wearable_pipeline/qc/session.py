# SPDX-License-Identifier: GPL-3.0-only
"""What one session's quality check needs.

Times are seconds from the start of the eye-tracker recording. Logger times (sync
taps, events) are left on the logger's own clock. Nothing here names a person,
a place or a file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stream:
    t: np.ndarray
    parts: tuple[int, ...] = ()  # the index at which each further recorder file starts
    event: bool = False          # events (blinks, fixations) may be quiet at the edges


@dataclass
class SyncCheck:
    logged_s: float                  # the tap, on the logger's clock
    detected_s: float | None = None  # the flash, on the reference clock
    review: str | None = None        # confirmed | set | rejected, by a person


@dataclass
class Event:
    t: float
    type: str          # phase | leave | return | session_start | anything else
    phase: str = ""    # for a phase event, the phase that starts
    void: bool = False


@dataclass
class Session:
    duration_s: float
    streams: dict[str, Stream]
    gaze_x: np.ndarray       # one per sample of streams["gaze"]
    gaze_y: np.ndarray
    worn: np.ndarray
    pupil_left: np.ndarray   # one per sample of streams["eye_state"]
    pupil_right: np.ndarray
    blinks: list[tuple[float, float]] = field(default_factory=list)
    syncs: list[SyncCheck] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    marker_s: float | None = None
    work_phase: str = "work"
    closing_phase: str | None = None  # the log may run on after this phase starts
