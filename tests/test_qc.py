# SPDX-License-Identifier: GPL-3.0-only
import copy
import json
import struct

import numpy as np
import pytest

from wearable_pipeline import pupil, qc
from wearable_pipeline.qc import demo, rules
from wearable_pipeline.qc.checks import SECTIONS, masked, stream_stats

MISSING = object()


@pytest.fixture(scope="module")
def base():
    return demo.simulate(())


@pytest.fixture
def session(base):
    return copy.deepcopy(base)


@pytest.fixture(scope="module")
def clean(base):
    return qc.run_qc(base)


@pytest.fixture(scope="module")
def faulty():
    return qc.run_qc(demo.simulate())


def status(report, key):
    return next((c["status"] for c in report["checks"] if c["id"] == key), MISSING)


def value(report, key):
    return next(c["value"] for c in report["checks"] if c["id"] == key)


# the demo

def test_clean_session_passes(clean):
    assert clean["verdict"] == "PASS"
    assert not [c["id"] for c in clean["checks"] if c["status"] in ("FAIL", "WARN")]


def test_every_requested_section_is_reported(clean):
    assert {c["section"] for c in clean["checks"]} == set(SECTIONS)


@pytest.mark.parametrize("fault,key,expected", [
    ("gaze_drops", "rates.gaze", "WARN"),
    ("not_worn", "worn.pct", "WARN"),
    ("imu_gap", "gaps.imu", "FAIL"),
    ("video_rollover", "gaps.scene_video.parts", "WARN"),
    ("sync_outside", "sync.2.offset", "WARN"),
    ("heart_stopped_early", "alignment.heart", "WARN"),
    ("break_unclosed", "breaks.paired", "WARN"),
])
def test_each_deliberate_fault_is_found(fault, key, expected):
    report = qc.run_qc(demo.simulate((fault,)))
    assert status(report, key) == expected
    assert report["verdict"] != "PASS"


def test_all_faults_together(faulty):
    assert faulty["verdict"] == "FAIL"
    failed = {c["id"] for c in faulty["checks"] if c["status"] == "FAIL"}
    assert failed == {"gaps.imu", "gaps.scene_video"}


def test_unknown_fault_is_refused():
    with pytest.raises(ValueError):
        demo.simulate(("nonsense",))


# thresholds

def test_registered_defaults():
    c = qc.load_config()
    registered = {k: c[k] for k, v in c.basis.items() if v == "registered"}
    assert registered == dict(gaze_valid_min_pct=70.0, gaze_second_valid_min_pct=80.0,
                              analysable_min_minutes=60.0, sync_min_usable=1,
                              pupil_range_mm=[1.5, 9.0], post_blink_ms=500.0)


def test_registered_pupil_values_match_the_pupil_module():
    c = qc.load_config()
    assert tuple(c["pupil_range_mm"]) == pupil.GATE_MM and c["post_blink_ms"] == pupil.POST_BLINK_MS


def test_operational_limits_are_not_called_registered():
    c = qc.load_config()
    for key in ("worn_pct", "pupil_in_range_pct", "streams", "sync_window_s", "span_s"):
        assert c.basis[key] == "operational"


def test_unset_thresholds_report_without_grading(clean):
    for key in ("sync.residual", "pupil.left.masked", "pupil.right.masked", "gaze.seconds"):
        assert status(clean, key) == "INFO"


def test_an_unset_threshold_can_be_set(session):
    report = qc.run_qc(session, {"pupil_after_mask_pct": {"pass": 99, "warn": 50}})
    assert status(report, "pupil.left.masked") == "WARN"
    assert report["thresholds"]["pupil_after_mask_pct"]["basis"] == "custom"


def test_config_file(tmp_path):
    path = tmp_path / "qc.json"
    path.write_text('{"analysable_min_minutes": 90}')
    c = qc.load_config(path)
    assert c["analysable_min_minutes"] == 90 and c.basis["analysable_min_minutes"] == "custom"


def test_repeating_a_default_keeps_its_basis():
    assert qc.load_config({"analysable_min_minutes": 60.0}).basis["analysable_min_minutes"] == "registered"


@pytest.mark.parametrize("bad,error", [({"no_such": 1}, KeyError), ({"worn_pct": {"pass": 9}}, ValueError)])
def test_bad_config_stops(bad, error):
    with pytest.raises(error):
        qc.load_config(bad)


def test_a_device_can_be_switched_off(session):
    del session.streams["acc"]
    assert status(qc.run_qc(session), "rates.acc.present") == "FAIL"
    assert status(qc.run_qc(session, {"streams": {"acc": None}}), "rates.acc.present") is MISSING


# rules, at their edges

@pytest.mark.parametrize("hz,expected", [(200, "PASS"), (190, "PASS"), (189.9, "WARN"),
                                         (150, "WARN"), (149.9, "FAIL"), (None, "FAIL")])
def test_gaze_rate_floor(hz, expected):
    assert rules.rate(hz, qc.load_config()["streams"]["gaze"]) == expected


@pytest.mark.parametrize("name,hz,expected", [
    ("imu", 103, "PASS"), ("imu", 96, "WARN"), ("imu", 80, "FAIL"),
    ("scene_video", 29.5, "PASS"), ("scene_video", 29, "WARN"), ("scene_video", 28, "FAIL"),
    ("ecg", 124, "PASS"), ("ecg", 115, "WARN"), ("ecg", 100, "FAIL"),
])
def test_relative_rate_bands(name, hz, expected):
    assert rules.rate(hz, qc.load_config()["streams"][name]) == expected


@pytest.mark.parametrize("pct,expected", [(95, "PASS"), (94.9, "WARN"), (80, "WARN"), (79.9, "FAIL")])
def test_worn_band(pct, expected):
    assert rules.at_least(pct, qc.load_config()["worn_pct"]) == expected


@pytest.mark.parametrize("head,tail,event,expected", [
    (5, 5, False, "PASS"), (5.1, 0, False, "WARN"), (0, 15.1, False, "FAIL"),
    (80, 20, True, "PASS"), (91, 0, True, "WARN"), (-6, 0, True, "FAIL"),
])
def test_span(head, tail, event, expected):
    assert rules.span(head, tail, {"pass": 5, "warn": 15}, event, 90) == expected


@pytest.mark.parametrize("lead,trail,expected", [(1, 1, "PASS"), (1, -120, "WARN"),
                                                 (1, -121, "FAIL"), (-1, 5, "FAIL")])
def test_heart_coverage(lead, trail, expected):
    assert rules.heart_coverage(lead, trail, 120) == expected


@pytest.mark.parametrize("early,after,measured,bookkeeping,expected", [
    (False, -10, 0, 0, "PASS"), (False, 120, 0, 0, "WARN"), (False, 121, 0, 0, "FAIL"),
    (True, -10, 0, 2, "PASS"), (True, -10, 1, 1, "FAIL"),
])
def test_log_window(early, after, measured, bookkeeping, expected):
    assert rules.log_window(early, after, measured, bookkeeping, 120) == expected


def test_gap_stats():
    st = stream_stats(np.array([0, 0.01, 0.02, 0.2, 0.21]), 0.05)
    assert st["n_gaps"] == 1 and st["gap_total_s"] == pytest.approx(0.18)


# sync checks

def test_offsets_and_residuals(session):
    session.syncs = [qc.SyncCheck(20, 20.2), qc.SyncCheck(5200, 5200.4)]
    report = qc.run_qc(session)
    checks = report["details"]["sync"]["checks"]
    assert [c["offset_ms"] for c in checks] == [200, 400]
    assert [c["residual_ms"] for c in checks] == [-100, 100]


@pytest.mark.parametrize("offset,state", [(-0.2, "accepted"), (-0.21, "outside window"),
                                          (5.0, "accepted"), (5.01, "outside window")])
def test_sync_window(session, offset, state):
    session.syncs[0] = qc.SyncCheck(20.0, 20.0 + offset)
    assert qc.run_qc(session)["details"]["sync"]["checks"][0]["state"] == state


def test_no_usable_sync_fails(session):
    session.syncs = [qc.SyncCheck(20), qc.SyncCheck(5200, 5260)]
    assert status(qc.run_qc(session), "sync.usable") == "FAIL"


def test_one_usable_sync_gives_an_offset_but_no_drift(session):
    session.syncs[1] = qc.SyncCheck(5200)
    report = qc.run_qc(session)
    assert status(report, "sync.usable") == "PASS" and status(report, "sync.drift") == "WARN"


def test_a_person_can_confirm_or_reject_a_sync(session):
    session.syncs = [qc.SyncCheck(20, 20.3, "rejected"), qc.SyncCheck(5200, 5207, "confirmed")]
    assert [c["state"] for c in qc.run_qc(session)["details"]["sync"]["checks"]] == ["rejected", "confirmed"]


def test_unknown_review_stops(session):
    session.syncs[0].review = "maybe"
    with pytest.raises(ValueError):
        qc.run_qc(session)


def test_drift_over_the_limit_warns(session):
    session.syncs = [qc.SyncCheck(20, 20.1), qc.SyncCheck(5200, 5201.1)]
    assert status(qc.run_qc(session), "sync.drift") == "WARN"


def test_short_baseline_cannot_pass_drift(session):
    session.syncs = [qc.SyncCheck(20, 20.2), qc.SyncCheck(200, 200.2)]
    assert status(qc.run_qc(session), "sync.drift") == "WARN"


def drift_check(report):
    return next(c for c in report["checks"] if c["id"] == "sync.drift")


@pytest.mark.parametrize("n", [1, 3])
def test_an_unexpected_number_of_syncs_warns_not_fails(session, n):
    session.syncs = [qc.SyncCheck(20 + 2000 * i, 20.3 + 2000 * i) for i in range(n)]
    assert status(qc.run_qc(session), "sync.logged") == "WARN"


def test_drift_inside_the_floor_passes_as_within_resolution(session):
    # 20 ms over about 1.4 h is under the floor of one frame at each end
    session.syncs = [qc.SyncCheck(20, 20.30), qc.SyncCheck(5200, 5200.32)]
    report = qc.run_qc(session)
    d = drift_check(report)
    assert d["status"] == "PASS" and "within resolution" in d["shown"]
    assert report["details"]["sync"]["drift_resolvable"] is True


def test_drift_above_the_floor_is_graded_against_the_limit(session):
    session.syncs = [qc.SyncCheck(20, 20.3), qc.SyncCheck(5200, 5200.7)]   # about 280 ms/h
    d = drift_check(qc.run_qc(session))
    assert d["status"] == "PASS" and "within resolution" not in d["shown"]


def test_a_baseline_under_an_hour_is_not_resolvable(session):
    # 0.9 h: the floor (about 74 ms/h) is well under the limit, the baseline is still too short
    session.syncs = [qc.SyncCheck(20, 20.3), qc.SyncCheck(3260, 3260.3)]
    report = qc.run_qc(session)
    d = drift_check(report)
    assert d["status"] == "WARN" and d["shown"] == "not resolvable"
    assert report["details"]["sync"]["drift_resolvable"] is False


def test_a_floor_at_the_limit_is_not_resolvable(session):
    # over an hour, but a limit tighter than the measurement can resolve
    session.syncs = [qc.SyncCheck(20, 20.3), qc.SyncCheck(5200, 5200.3)]
    d = drift_check(qc.run_qc(session, {"sync_drift_max_ms_per_hour": 40.0}))
    assert d["status"] == "WARN" and d["shown"] == "not resolvable"


def test_three_syncs_fit_end_to_end_by_onset(session):
    # the middle check is far off the line; logged out of order on purpose
    session.syncs = [qc.SyncCheck(5200, 5200.7), qc.SyncCheck(2600, 2604.0), qc.SyncCheck(20, 20.3)]
    report = qc.run_qc(session)
    hours = (5200.7 - 20.3) / 3600
    assert report["details"]["sync"]["drift_ms_per_hour"] == pytest.approx(400 / hours, abs=0.1)
    assert "between" in drift_check(report)["note"]


def test_not_worn_time_is_left_out_of_pupil_range(session, clean):
    t = session.streams["eye_state"].t
    off = (t > 1000) & (t < 2500)                 # glasses off for 25 minutes
    session.worn = session.worn & ~off
    session.pupil_left[off] = np.nan
    report = qc.run_qc(session)
    assert status(report, "pupil.left.range") == "PASS"
    assert value(report, "pupil.left.range") == pytest.approx(value(clean, "pupil.left.range"), abs=0.5)
    assert status(report, "worn.pct") != "PASS"


# gaze, worn, pupil

def test_gaze_under_70_warns_and_never_fails(session):
    session.gaze_x[: int(0.4 * session.gaze_x.size)] = np.nan
    assert status(qc.run_qc(session), "gaze.valid") == "WARN"


def test_dropped_samples_count_as_invalid_gaze(session):
    keep = np.ones(session.gaze_x.size, bool)
    keep[100000:500000] = False
    t = session.streams["gaze"].t[keep]
    session.streams["gaze"], session.streams["eye_state"] = qc.Stream(t), qc.Stream(t)
    for name in ("gaze_x", "gaze_y", "worn", "pupil_left", "pupil_right"):
        setattr(session, name, getattr(session, name)[keep])
    assert value(qc.run_qc(session), "gaze.valid") < 65


def test_pupil_mask_is_blink_plus_500_ms():
    t = np.arange(0, 3, 0.005)
    out = masked(t, [(1.0, 1.2)], None, 500.0)
    assert out[(t >= 1.0) & (t < 1.695)].all() and not out[(t < 0.995) | (t > 1.705)].any()


def test_masking_lowers_pupil_coverage(clean):
    assert value(clean, "pupil.left.masked") < value(clean, "pupil.left.range")


def test_worn_of_the_wrong_length_fails(session):
    session.worn = session.worn[:-1]
    assert status(qc.run_qc(session), "gaze.pairing") == "FAIL"


# gaps, rates, alignment

def test_overlapping_files_fail(session):
    stream = session.streams["imu"]
    t = stream.t.copy()
    t[1000:] -= 2.0
    session.streams["imu"] = qc.Stream(t, parts=(1000,))
    assert status(qc.run_qc(session), "gaps.imu.parts") == "FAIL"


def test_the_largest_boundary_loss_is_graded(session):
    t = session.streams["imu"].t.copy()
    t[1000:] += 0.1
    t[2000:] += 31.0
    session.streams["imu"] = qc.Stream(t, parts=(1000, 2000))
    assert status(qc.run_qc(session), "gaps.imu.parts") == "FAIL"


def test_different_clocks_fail(session):
    session.streams["eye_state"] = qc.Stream(session.streams["eye_state"].t + 0.001)
    assert status(qc.run_qc(session), "rates.eye_state.clock") == "FAIL"


def test_one_shared_clock_is_counted_once():
    report = qc.run_qc(demo.simulate(("gaze_drops",)))
    assert report["counts"]["WARN"] == 1


def test_stream_ending_early_fails(session):
    t = session.streams["imu"].t
    session.streams["imu"] = qc.Stream(t[t < 3000])
    assert status(qc.run_qc(session), "alignment.imu.span") == "FAIL"


@pytest.mark.parametrize("name", ["gaze", "eye_state", "imu", "ecg"])
def test_missing_or_empty_stream_fails_without_crashing(session, name):
    session.streams[name] = qc.Stream(np.array([]))
    assert qc.run_qc(session)["verdict"] == "FAIL"


def test_bad_timestamps_fail(session):
    session.streams["imu"].t[5] = np.nan
    assert status(qc.run_qc(session), "alignment.imu.timestamps") == "FAIL"


def test_marker_never_fails(session):
    session.marker_s = 500
    assert status(qc.run_qc(session), "alignment.marker") == "WARN"


def test_log_after_the_closing_phase_does_not_count(session):
    session.events.append(qc.Event(session.duration_s + 9000, "note"))
    assert status(qc.run_qc(session), "alignment.log") == "PASS"


# breaks

def test_break_is_taken_out_of_work(clean):
    assert value(clean, "breaks.analysable") == pytest.approx((5150 - 400 - 300) / 60, abs=0.01)


def test_break_outside_work_is_not_taken_out(session):
    session.events += [qc.Event(100, "leave"), qc.Event(200, "return")]
    assert value(qc.run_qc(session), "breaks.analysable") == pytest.approx(74.167, abs=0.01)


def test_under_60_minutes_of_work_fails(session):
    session.events += [qc.Event(2500, "leave"), qc.Event(3700, "return")]
    assert status(qc.run_qc(session), "breaks.analysable") == "FAIL"


def test_void_events_are_ignored(session):
    session.events.append(qc.Event(2500, "leave", void=True))
    assert status(qc.run_qc(session), "breaks.paired") == "PASS"


# outputs

def test_json_and_one_page_png(faulty, tmp_path):
    json_path, png_path = qc.write_report(faulty, tmp_path)
    assert json.loads(json_path.read_text())["verdict"] == "FAIL"
    png = png_path.read_bytes()
    width, height = struct.unpack(">II", png[16:24])
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and height / width == pytest.approx(11.69 / 8.27, rel=0.01)


def test_json_is_strict_for_a_broken_session(session, tmp_path):
    session.streams = {}
    session.syncs, session.events = [], []
    report = qc.run_qc(session)
    qc.write_report(report, tmp_path)
    assert report["verdict"] == "FAIL"


def test_report_holds_no_outcome_path_or_date(faulty):
    text = json.dumps(faulty).lower()
    for word in ("fixation_ms", "pupil_mm\"", "median_pupil", "blinks_per_min", "entropy", "saccade", "/users/"):
        assert word not in text
    assert "20:" not in text and "t00:" not in text


def test_shown_values_never_carry_two_signs(faulty, clean):
    for report in (faulty, clean):
        for c in report["checks"]:
            shown = str(c["shown"])
            assert "--" not in shown and "+-" not in shown and "-+" not in shown, (c["id"], shown)
            assert "-0.0 " not in shown + " " and "-0 " not in shown + " ", (c["id"], shown)


def test_signed_formatting():
    from wearable_pipeline.qc.checks import _signed
    assert [_signed(x) for x in (0.03, -0.03, -1.26, 2)] == ["+0.0", "+0.0", "-1.3", "+2.0"]
