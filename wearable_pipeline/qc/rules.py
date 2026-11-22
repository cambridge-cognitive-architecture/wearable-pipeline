# SPDX-License-Identifier: GPL-3.0-only
"""How a number becomes PASS, WARN or FAIL."""
from __future__ import annotations

import math

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
BOOKKEEPING = ("session_start", "zone_start")  # log events that measure nothing


def finite(x) -> bool:
    return x is not None and math.isfinite(x)


def at_least(value, limits) -> str:
    if limits is None:
        return INFO
    if not finite(value):
        return FAIL
    return PASS if value >= limits["pass"] else WARN if value >= limits["warn"] else FAIL


def at_most(value, limits) -> str:
    if limits is None:
        return INFO
    if not finite(value):
        return FAIL
    return PASS if value <= limits["pass"] else WARN if value <= limits["warn"] else FAIL


def rate(hz, spec: dict) -> str:
    """An absolute floor where there is one, otherwise deviation from nominal."""
    if "pass_hz" in spec:
        return at_least(hz, {"pass": spec["pass_hz"], "warn": spec["warn_hz"]})
    if "pass_dev" not in spec:
        return INFO
    if not finite(hz):
        return FAIL
    nominal = spec["nominal_hz"]
    return at_most(abs(hz - nominal) / nominal, {"pass": spec["pass_dev"], "warn": spec["warn_dev"]})


def gaps(n: int, total_s: float, fail_s: float) -> str:
    return PASS if n == 0 else WARN if total_s < fail_s else FAIL


def span(head_s, tail_s, limits: dict, event: bool, event_warn_s: float) -> str:
    """Does a stream cover the recording?"""
    if not (finite(head_s) and finite(tail_s)):
        return FAIL
    worst = max(head_s, tail_s)
    if event:  # quiet edges are normal; events outside the recording are not
        if min(head_s, tail_s) < -limits["pass"]:
            return FAIL
        return PASS if worst <= event_warn_s else WARN
    return at_most(worst, limits)


def heart_coverage(lead_s: float, trail_s: float, tail_note_s: float) -> str:
    """A late start loses the first sync, so it fails. A short tail only warns."""
    if lead_s >= 0 and trail_s >= 0:
        return PASS
    if lead_s >= 0 and -trail_s <= tail_note_s:
        return WARN
    return FAIL


def log_window(starts_early: bool, ends_after_s: float, measured_before: int,
               bookkeeping_before: int, tail_note_s: float) -> str:
    """What matters is whether a measured event has a recording under it."""
    if not starts_early and ends_after_s <= 0:
        return PASS
    if measured_before:
        return FAIL
    if 0 < ends_after_s <= tail_note_s:
        return WARN
    if ends_after_s > tail_note_s:
        return FAIL
    return PASS if bookkeeping_before else FAIL


def overall(statuses) -> str:
    seen = set(statuses)
    return FAIL if FAIL in seen else WARN if WARN in seen else PASS
