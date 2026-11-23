# SPDX-License-Identifier: GPL-3.0-only
"""The checks on one session.

Only quality figures are reported: coverage, counts, rates of sampling and time.
No pupil size, gaze position, fixation or blink measure appears, so a report can be
read during data collection without showing an outcome.
"""
from __future__ import annotations

import numpy as np

from ..align import fit_sync
from . import rules
from .config import Config, load_config
from .rules import FAIL, INFO, PASS, WARN
from .session import Session

SCHEMA = "wearable-pipeline-qc/1"
SECTIONS = ("sync", "worn", "gaze", "pupil", "gaps", "breaks", "rates", "alignment")


def _number(x, digits: int = 3):
    return round(float(x), digits) if x is not None and np.isfinite(x) else None


def _show(x, unit: str = "", digits: int = 1) -> str:
    return "none" if x is None or not np.isfinite(x) else f"{x:.{digits}f}{unit}"


class _Report:
    def __init__(self, config: Config):
        self.c = config
        self.checks: list[dict] = []
        self.details: dict = {name: {} for name in SECTIONS}

    def add(self, key, label, status, value, shown, expected, threshold=None, tally=True, note=""):
        self.checks.append(dict(
            id=key, section=key.split(".")[0], label=label, status=status, value=value,
            shown=shown, expected=expected, threshold=threshold, note=note, tally=tally,
            basis=self.c.basis[threshold] if threshold else "structural"))

    def limits(self, threshold, sign, unit) -> str:
        limit = self.c[threshold]
        return f"{sign} {limit['pass']:g}{unit}" if limit else "reported"


def stream_stats(t, gap_s: float) -> dict:
    t = np.sort(np.asarray(t, float))
    if t.size < 2:
        return dict(n=int(t.size), hz=None, median_dt_s=None, n_gaps=0, gap_total_s=0.0, max_gap_s=None)
    dt = np.diff(t)
    long = dt[dt > gap_s]
    length = t[-1] - t[0]
    return dict(n=int(t.size), hz=(t.size - 1) / length if length > 0 else None,
                median_dt_s=float(np.median(dt)), n_gaps=int(long.size),
                gap_total_s=float(long.sum()), max_gap_s=float(dt.max()))


def _streams(r: _Report, s: Session) -> dict[str, np.ndarray]:
    """Rates, gaps, file boundaries and span. Returns the usable timestamps."""
    c, specs, usable = r.c, r.c["streams"], {}
    rates, timeline, hz_of = {}, {}, {}
    names = [*specs, *(n for n in s.streams if n not in specs)]
    for name in names:
        spec, stream = specs.get(name, {}), s.streams.get(name)
        if stream is None or len(stream.t) == 0:
            if spec:
                r.add(f"rates.{name}.present", f"{name}: present", FAIL, 0, "absent", "present", "streams")
            continue
        raw = np.asarray(stream.t, float)
        if not np.isfinite(raw).all():
            r.add(f"alignment.{name}.timestamps", f"{name}: timestamps valid", FAIL,
                  int((~np.isfinite(raw)).sum()), "non-finite timestamps", "all finite")
            continue
        usable[name] = raw
        backwards = np.flatnonzero(np.diff(raw) < 0) + 1
        r.add(f"alignment.{name}.order", f"{name}: in time order", FAIL if backwards.size else PASS,
              int(backwards.size), f"{backwards.size} steps back", "0")

        head, tail = float(raw.min()), float(s.duration_s - raw.max())
        timeline[name] = dict(start_s=_number(head), end_s=_number(raw.max()), gaps=[])
        if spec.get("group") != "heart":  # the heart recorder has its own coverage rule
            key = "span_event_warn_s" if stream.event else "span_s"
            limit = c["span_event_warn_s"] if stream.event else c["span_s"]["pass"]
            r.add(f"alignment.{name}.span", f"{name}: spans the recording",
                  rules.span(head, tail, c["span_s"], stream.event, c["span_event_warn_s"]),
                  dict(head_s=_number(head), tail_s=_number(tail)),
                  f"+{head:.1f} / -{tail:.1f} s", f"within {limit:g} s", key)

        if stream.parts:
            if any(not 0 < i < raw.size for i in stream.parts) or sorted(set(stream.parts)) != list(stream.parts):
                r.add(f"gaps.{name}.parts", f"{name}: file boundaries valid", FAIL, None,
                      "invalid indices", "ascending indices")
            else:
                # The loss is the largest boundary gap; an overlap is the smallest going negative.
                at = [float(raw[i] - raw[i - 1]) for i in stream.parts]
                lost = FAIL if min(at) < 0 else rules.at_most(max(at), c["part_boundary_s"])
                r.add(f"gaps.{name}.parts", f"{name}: lost between files", lost, _number(max(at)),
                      f"{max(at):.2f} s", r.limits("part_boundary_s", "<=", " s"), "part_boundary_s")

        if stream.event:
            continue
        st = stream_stats(raw, c["gap_ms"] / 1000)
        hz_of[name] = st["hz"]
        twin = spec.get("same_clock_as")
        expected = (f">= {spec['pass_hz']:g} Hz" if "pass_hz" in spec
                    else f"{spec['nominal_hz']:g} Hz ±{100 * spec['pass_dev']:g}%" if "pass_dev" in spec
                    else "reported")
        r.add(f"rates.{name}", f"{name}: rate" + (" (not counted twice)" if twin else ""), rules.rate(st["hz"], spec), _number(st["hz"]),
              _show(st["hz"], " Hz"), expected, "streams" if spec else None,
              tally=twin is None)  # one clock, one fact: counted once
        rates[name] = dict(effective_hz=_number(st["hz"]), nominal_hz=spec.get("nominal_hz"),
                           median_interval_ms=_number(1000 * st["median_dt_s"] if st["median_dt_s"] else None),
                           share_of_nominal=_number(st["hz"] / spec["nominal_hz"]
                                                    if st["hz"] and spec.get("nominal_hz") else None, 4))
        how = spec.get("gaps")
        if how == "short":
            r.add(f"gaps.{name}", f"{name}: gaps > {c['gap_ms']:g} ms",
                  rules.gaps(st["n_gaps"], st["gap_total_s"], c["gap_total_fail_s"]),
                  dict(n=st["n_gaps"], total_s=_number(st["gap_total_s"]), max_s=_number(st["max_gap_s"])),
                  f"{st['n_gaps']} ({st['gap_total_s']:.2f} s)", "0", "gap_total_fail_s")
        elif how == "long":
            long = stream_stats(raw, c["long_gap_s"])
            r.add(f"gaps.{name}", f"{name}: gaps > {c['long_gap_s']:g} s",
                  PASS if long["n_gaps"] <= c["long_gaps_pass_max"] else WARN,
                  dict(n=long["n_gaps"], total_s=_number(long["gap_total_s"])),
                  f"{long['n_gaps']} ({long['gap_total_s']:.0f} s)",
                  f"<= {c['long_gaps_pass_max']}", "long_gaps_pass_max")
        ordered = np.sort(raw)
        dt = np.diff(ordered)
        where = np.flatnonzero(dt > c["gap_ms"] / 1000)
        longest = where[np.argsort(-dt[where])][:200]
        timeline[name]["gaps"] = sorted([_number(ordered[i]), _number(ordered[i + 1])] for i in longest)

    for name, spec in specs.items():
        twin = spec.get("same_clock_as")
        if twin and name in usable and twin in usable:
            same = np.array_equal(usable[name], usable[twin])
            r.add(f"rates.{name}.clock", f"{name}: same clock as {twin}", PASS if same else FAIL,
                  bool(same), "identical" if same else "different", "identical")
    if hz_of.get("gaze") and hz_of.get("eye_video"):
        share = 100 * hz_of["gaze"] / hz_of["eye_video"]
        # Under 100% with a healthy camera means inference dropped samples, not a timing fault.
        r.add("rates.gaze_vs_camera", "gaze rate, of eye camera", INFO, _number(share),
              _show(share, "%"), "reported")
    r.details["rates"], r.details["alignment"]["timeline"] = rates, timeline
    return usable


def drift_grade(pairs, c) -> tuple:
    """Clock drift from (logged_s, detected_s) pairs, graded against the limit.

    The fit is end to end by onset: the earliest and latest detected flash,
    with any checks in between recorded but not fitted. The floor is the
    resolution of the measurement, one scene frame at each end over the
    baseline. The drift is graded against the limit and never against the
    floor: a drift inside the floor passes as within resolution. It is not
    resolvable only when the baseline is shorter than the minimum or the floor
    is itself at or above the limit.
    Returns drift, floor, hours, resolvable, status, shown, note.
    """
    pairs = sorted(pairs, key=lambda p: p[1])
    (l0, d0), (l1, d1) = pairs[0], pairs[-1]
    hours = (d1 - d0) / 3600
    limit, min_h = c["sync_drift_max_ms_per_hour"], c["sync_drift_min_baseline_h"]
    scene = c["streams"].get("scene_video", {}).get("nominal_hz", 30.0)
    floor = 2000 / scene / hours if hours > 0 else float("inf")
    drift = 1000 * ((d1 - l1) - (d0 - l0)) / hours if hours > 0 else None
    between = f"; {len(pairs) - 2} check(s) between recorded, not fitted" if len(pairs) > 2 else ""
    if hours < min_h or floor >= limit:
        why = (f"baseline {hours:.2f} h is under {min_h:g} h" if hours < min_h
               else f"floor {floor:.0f} ms/h is at or above the limit")
        return drift, floor, hours, False, WARN, "not resolvable", why + between
    shown = f"{drift:+.0f} ± {floor:.0f} ms/h"
    if abs(drift) <= floor:
        return drift, floor, hours, True, PASS, shown + ", within resolution", \
            f"over {hours:.2f} h" + between
    return drift, floor, hours, True, PASS if abs(drift) <= limit else WARN, shown, \
        f"over {hours:.2f} h" + between


def _sync(r: _Report, s: Session) -> None:
    c, (low, high) = r.c, r.c["sync_window_s"]
    n = len(s.syncs)
    # Another count is worth a look, not a fault: the fit uses whatever is usable.
    r.add("sync.logged", "sync checks logged", PASS if n == c["sync_expected"] else WARN, n,
          str(n), str(c["sync_expected"]), "sync_expected")
    rows, usable = [], []
    for i, check in enumerate(sorted(s.syncs, key=lambda x: x.logged_s), 1):
        if check.review not in (None, "confirmed", "set", "rejected"):
            raise ValueError(f"unknown review {check.review!r}")
        seen = check.detected_s is not None and np.isfinite(check.detected_s)
        offset = check.detected_s - check.logged_s if seen else None
        if check.review == "rejected":
            state = "rejected"
        elif not seen:
            state = "not found"
        elif check.review:
            state = check.review
        elif low - 1e-9 <= offset <= high + 1e-9:
            state = "accepted"
        else:
            state = "outside window"  # held for a person, never searched for more widely
        used = state in ("accepted", "confirmed", "set")
        row = dict(n=i, at_s=_number(check.logged_s), state=state, residual_ms=None,
                   offset_ms=_number(1000 * offset if seen else None, 1))
        rows.append(row)
        if used:
            usable.append((check, row))
        r.add(f"sync.{i}.offset", f"sync {i}: offset", INFO if used else WARN, row["offset_ms"],
              _show(row["offset_ms"], " ms", 0) if used else state,
              f"{low:g} to +{high:g} s", "sync_window_s")

    r.add("sync.usable", "usable sync checks", PASS if len(usable) >= c["sync_min_usable"] else FAIL,
          len(usable), f"{len(usable)} of {n}", f">= {c['sync_min_usable']}", "sync_min_usable",
          note="a failed sync check can be confirmed or set by hand")
    fit = None
    if usable:
        fit = fit_sync([x.logged_s for x, _ in usable], [x.detected_s for x, _ in usable],
                       max_residual_s=float("inf"))
        for (_, row), residual in zip(usable, fit.residual_s):
            row["residual_ms"] = _number(1000 * residual, 1)
    worst = float(1000 * np.abs(fit.residual_s).max()) if fit else None
    drift = uncertainty = hours = None
    resolvable = False
    status, shown, note = WARN, "needs two checks", "drift needs two usable checks"
    if len(usable) >= 2:
        drift, uncertainty, hours, resolvable, status, shown, note = drift_grade(
            [(x.logged_s, x.detected_s) for x, _ in usable], c)
    r.add("sync.residual", "largest sync residual",
          rules.at_most(worst, c["sync_residual_ms"]) if len(usable) >= 2 else INFO,
          _number(worst, 1), _show(worst, " ms", 0) if len(usable) >= 2 else "needs two checks",
          r.limits("sync_residual_ms", "<=", " ms"), "sync_residual_ms")
    r.add("sync.drift", "clock drift", status, _number(drift, 1), shown,
          f"<= {c['sync_drift_max_ms_per_hour']:g} ms/h", "sync_drift_max_ms_per_hour", note=note)
    r.details["sync"] = dict(checks=rows, offset_ms=_number(1000 * fit.offset_s if fit else None, 1),
                             drift_ms_per_hour=_number(drift, 1),
                             drift_uncertainty_ms_per_hour=_number(uncertainty, 1),
                             drift_baseline_h=_number(hours, 3), drift_resolvable=bool(resolvable))


def _worn_and_gaze(r: _Report, s: Session, usable: dict) -> None:
    c, t = r.c, usable.get("gaze")
    arrays = [np.asarray(a) for a in (s.gaze_x, s.gaze_y, s.worn)]
    if t is None or any(a.shape != t.shape for a in arrays):
        r.add("gaze.pairing", "gaze and worn pair by sample", FAIL, None, "do not pair", "equal lengths")
        return
    x, y, worn = arrays
    worn = np.nan_to_num(worn.astype(float)) > 0
    pct = 100 * float(worn.mean())
    r.add("worn.pct", "worn", rules.at_least(pct, c["worn_pct"]), _number(pct), _show(pct, "%"),
          r.limits("worn_pct", ">=", "%"), "worn_pct")

    dt = float(np.median(np.diff(np.sort(t)))) if t.size > 1 else 0.0
    if dt <= 0:
        r.add("gaze.valid", "valid gaze", FAIL, None, "too few samples", "measurable")
        return
    # Out of the samples expected, so dropped samples and gaps count as invalid.
    inside = (t >= 0) & (t < s.duration_s)
    valid = t[inside & worn & np.isfinite(x) & np.isfinite(y)]
    pct = 100 * min(1.0, valid.size / (s.duration_s / dt))
    floor = c["gaze_valid_min_pct"]
    r.add("gaze.valid", "valid gaze", PASS if pct >= floor else WARN, _number(pct), _show(pct, "%"),
          f">= {floor:g}%", "gaze_valid_min_pct", note="a sensitivity restriction, not an exclusion")

    whole = int(s.duration_s)
    counts = np.bincount(valid.astype(np.int64), minlength=whole)[:whole]
    second_pct = np.minimum(100.0, 100 * counts * dt)
    ok = 100 * float((second_pct >= c["gaze_second_valid_min_pct"] - 1e-6).mean()) if whole else None
    r.add("gaze.seconds", f"seconds >= {c['gaze_second_valid_min_pct']:g}% valid",
          rules.at_least(ok, c["gaze_seconds_valid_pct"]), _number(ok), _show(ok, "%"),
          r.limits("gaze_seconds_valid_pct", ">=", "%"), "gaze_seconds_valid_pct")
    r.details["gaze"] = dict(valid_pct_per_minute=[_number(second_pct[i:i + 60].mean(), 1)
                                                  for i in range(0, whole, 60)])


def masked(t, blinks, worn, post_blink_ms: float) -> np.ndarray:
    """Blinks plus the recovery after them, and time the glasses were off."""
    mask = np.zeros(t.size, bool) if worn is None else ~worn
    order = np.argsort(t)
    ordered = t[order]
    for start, end in blinks:
        a, b = np.searchsorted(ordered, [start, end + post_blink_ms / 1000])
        mask[order[a:b]] = True
    return mask


def _pupil(r: _Report, s: Session, usable: dict) -> None:
    c, t = r.c, usable.get("eye_state")
    eyes = dict(left=np.asarray(s.pupil_left, float), right=np.asarray(s.pupil_right, float))
    if t is None or any(a.shape != t.shape for a in eyes.values()):
        r.add("pupil.pairing", "pupil pairs by sample", FAIL, None, "do not pair", "equal lengths")
        return
    gaze, worn = usable.get("gaze"), None
    if gaze is not None and np.array_equal(gaze, t) and np.shape(s.worn) == t.shape:
        worn = np.nan_to_num(np.asarray(s.worn).astype(float)) > 0
    out = masked(t, s.blinks, worn, c["post_blink_ms"])
    low, high = c["pupil_range_mm"]
    for eye, d in eyes.items():
        ok = np.isfinite(d) & (d >= low) & (d <= high)
        on = worn if worn is not None else np.ones(ok.shape, bool)
        # Time with the glasses off says nothing about the pupil, so it is left out here.
        before = 100 * float(ok[on].mean()) if on.any() else None
        after = 100 * float((ok & ~out).mean())
        r.add(f"pupil.{eye}.range", f"{eye} in {low:g}-{high:g} mm",
              rules.at_least(before, c["pupil_in_range_pct"]), _number(before), _show(before, "%"),
              r.limits("pupil_in_range_pct", ">=", "%"), "pupil_in_range_pct")
        r.add(f"pupil.{eye}.masked", f"{eye} after blink mask",
              rules.at_least(after, c["pupil_after_mask_pct"]), _number(after), _show(after, "%"),
              r.limits("pupil_after_mask_pct", ">=", "%"), "pupil_after_mask_pct")


def _breaks(r: _Report, s: Session) -> None:
    c = r.c
    events = sorted((e for e in s.events if not e.void), key=lambda e: e.t)
    last = events[-1].t if events else 0.0
    phases = [e for e in events if e.type == "phase"]
    ends = [e.t for e in phases[1:]] + [last]
    work = [(a.t, b) for a, b in zip(phases, ends) if a.phase == s.work_phase and b > a.t]
    breaks, since, stray = [], None, 0
    for e in events:
        if e.type == "leave" and since is None:
            since = e.t
        elif e.type == "return":
            if since is None:
                stray += 1
            elif e.t > since:
                breaks.append((since, e.t))
            since = None
    still_out = since is not None
    if still_out and last > since:
        breaks.append((since, last))  # they did not come back inside the log
    work_s = sum(b - a for a, b in work)
    lost = sum(max(0.0, min(b, y) - max(a, x)) for a, b in breaks for x, y in work)
    minutes = max(0.0, work_s - lost) / 60

    r.add("breaks.logged", "breaks logged", INFO,
          dict(n=len(breaks), total_s=_number(sum(b - a for a, b in breaks))),
          f"{len(breaks)} ({sum(b - a for a, b in breaks) / 60:.1f} min)", "reported")
    unpaired = stray + int(still_out)
    r.add("breaks.paired", "breaks have start and end", WARN if unpaired else PASS, unpaired,
          f"{unpaired} unpaired", "0")
    need = c["analysable_min_minutes"]
    r.add("breaks.analysable", "analysable work", PASS if minutes >= need else FAIL, _number(minutes),
          _show(minutes, " min"), f">= {need:g} min", "analysable_min_minutes",
          note="work phase, less breaks inside it")
    r.details["breaks"] = dict(work=[[_number(a), _number(b)] for a, b in work],
                               breaks=[[_number(a), _number(b)] for a, b in breaks],
                               work_s=_number(work_s), break_in_work_s=_number(lost))


def _alignment(r: _Report, s: Session, usable: dict) -> None:
    c = r.c
    heart = [usable[n] for n, spec in c["streams"].items() if spec.get("group") == "heart" and n in usable]
    if heart:
        lead = -min(float(t.min()) for t in heart)
        trail = max(float(t.max()) for t in heart) - s.duration_s
        r.add("alignment.heart", "heart recorder covers it",
              rules.heart_coverage(lead, trail, c["heart_tail_note_s"]),
              dict(lead_s=_number(lead), trail_s=_number(trail)),
              f"{lead:+.1f} / {trail:+.1f} s", "both >= 0", "heart_tail_note_s")

    events = sorted((e for e in s.events if not e.void), key=lambda e: e.t)
    if events:
        closing = [e.t for e in events if e.type == "phase" and s.closing_phase and e.phase == s.closing_phase]
        end = closing[0] if closing else events[-1].t
        before = [e for e in events if e.t < 0]
        measured = sum(e.type not in rules.BOOKKEEPING for e in before)
        after = float(end - s.duration_s)
        r.add("alignment.log", "log inside the recording",
              rules.log_window(events[0].t < 0, after, measured, len(before), c["log_tail_note_s"]),
              dict(start_s=_number(events[0].t), ends_after_s=_number(after), measured_before=measured),
              f"+{events[0].t:.0f} / -{-after:.0f} s" if after <= 0 else f"ends {after:.0f} s after",
              "inside", "log_tail_note_s")
    else:
        r.add("alignment.log", "field log present", FAIL, 0, "no events", "present")

    if s.marker_s is not None:
        r.add("alignment.marker", "marker vs recording start",
              PASS if abs(s.marker_s) <= c["marker_s"] else WARN, _number(s.marker_s),
              _show(s.marker_s, " s", 2), f"within {c['marker_s']:g} s", "marker_s")


def run_qc(session: Session, config: Config | dict | None = None) -> dict:
    """The report for one session, ready for JSON."""
    c = config if isinstance(config, Config) else load_config(config)
    if not (np.isfinite(session.duration_s) and session.duration_s > 0):
        raise ValueError("the recording needs a positive duration")
    r = _Report(c)
    usable = _streams(r, session)
    _sync(r, session)
    _worn_and_gaze(r, session, usable)
    _pupil(r, session, usable)
    _breaks(r, session)
    _alignment(r, session, usable)
    r.checks.sort(key=lambda check: SECTIONS.index(check["section"]))
    counted = [check["status"] for check in r.checks if check["tally"]]
    return dict(schema=SCHEMA, verdict=rules.overall(counted),
                counts={status: counted.count(status) for status in (FAIL, WARN, PASS, INFO)},
                duration_s=_number(session.duration_s), checks=r.checks, details=r.details,
                thresholds={k: dict(value=v, basis=c.basis[k], rule=c.rules[k]) for k, v in c.values.items()})
