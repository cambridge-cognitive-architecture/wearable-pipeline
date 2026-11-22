# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest

from wearable_pipeline import align, hrv, pupil
from wearable_pipeline.demo import process, simulate
from wearable_pipeline.timebase import INAT, Clock, iso_utc_to_ns, local_naive_to_ns


def test_local_and_utc_agree():
    assert local_naive_to_ns("2000-01-01T13:00:00.250", 3600) == iso_utc_to_ns(
        "2000-01-01T12:00:00.250Z")
    many = iso_utc_to_ns(["2000-01-01T12:00:00Z", "2000-01-01T12:00:00.001Z", "bad"])
    assert many[1] - many[0] == 1_000_000 and many[2] == INAT


def test_clock_precision_and_missing():
    start = 946728000_000_000_000
    out = Clock(start, start, 0).rel_s([start + 1000, INAT])
    assert out[0] == 1e-6 and np.isnan(out[1])


def test_sync():
    assert align.fit_sync([1, 11, 21], [3, 13, 23]).offset_s == 2
    with pytest.raises(ValueError):
        align.fit_sync([1, 11], [3, 19])


def test_gaps_stay_gaps():
    out = align.bin_mean([0, 0.1, 3], [1, 3, 5], [0, 1, 2, 3], 1)
    np.testing.assert_equal(out, [2, np.nan, np.nan, 5])


T = np.arange(0, 10, 1 / 200)


def steady(level=4.0):
    return level + 0.01 * np.sin(2 * np.pi * 0.3 * T) + np.random.default_rng(1).normal(0, 0.002, T.size)


def ns(x):
    return np.round(np.asarray(x) * 1e9).astype(np.int64)


def test_gate_is_per_eye_and_has_a_sensitivity():
    out, report = pupil.clean(T, steady(1.2), steady(4.0))
    assert report["left"]["removed_gate"] == T.size and report["right"]["removed_gate"] == 0
    assert np.nanmedian(out) == pytest.approx(4.0, abs=0.02)  # not dragged to 2.6
    _, loose = pupil.clean(T, steady(1.2), steady(4.0), gate=pupil.GATE_SENSITIVITY_MM)
    assert loose["left"]["removed_gate"] == 0


def test_blink_plus_500_ms_is_masked():
    out, _ = pupil.clean(T, steady(), steady(), [(4.0, 4.2)])
    assert np.isnan(out[(T >= 4.0) & (T < 4.7)]).all()
    assert np.isfinite(out[T > 4.8]).all() and np.isfinite(out[T < 3.9]).all()


def test_short_gap_is_interpolated_long_gap_is_not():
    d = steady()
    d[(T >= 2) & (T < 2.04)] = np.nan
    d[(T >= 6) & (T < 6.4)] = np.nan
    out, _ = pupil.clean_eye(ns(T), d)
    assert np.isfinite(out[(T >= 2) & (T < 2.04)]).all()
    assert np.isnan(out[(T >= 5.96) & (T < 6.44)]).all()  # padded by 50 ms


def test_speed_filter_takes_a_spike():
    d = steady()
    d[1000] += 1.5
    out, report = pupil.clean_eye(ns(T), d)
    assert report["removed_speed"] >= 1 and abs(out[1000] - 4.0) < 0.05


def test_worn_is_applied():
    out, report = pupil.clean(T, steady(), steady(), worn=~((T >= 5) & (T < 6)))
    assert np.isnan(out[(T >= 5) & (T < 6)]).all() and report["left"]["removed_not_worn"] == 200


def test_single_eye_is_put_on_the_mean_scale():
    out, report = pupil.combine_eyes([4.2, 4.2, np.nan], [4.0, np.nan, 4.0])
    np.testing.assert_allclose(out, [4.1, 4.1, 4.1])
    assert report["between_eye_difference_mm"] == pytest.approx(0.2)


def test_step_in_heart_rate_is_not_lost():
    rr = np.r_[np.full(20, 900.0), np.full(200, 690.0)]
    assert hrv.clean_mask(rr).all()
    assert hrv.clean_mask(rr, reseed_after=None).sum() == 20
    first_bad = hrv.clean_mask(np.r_[1500.0, np.full(200, 800.0)])
    assert not first_bad[0] and first_bad[1:].all()


def test_single_ectopic_is_removed():
    rr = np.full(30, 800.0)
    rr[10] = 1300
    assert list(np.nonzero(~hrv.clean_mask(rr))[0]) == [10]


def test_no_difference_across_a_dropout():
    t = np.r_[np.arange(10.0), np.arange(100.0, 110.0)]
    assert hrv.rmssd(np.r_[np.full(10, 800.0), np.full(10, 900.0)], t_s=t) == 0


def test_no_difference_across_a_new_level():
    rr = np.r_[np.full(30, 900.0), np.full(30, 690.0)]
    assert hrv.rmssd(rr, hrv.clean_mask(rr)) == 0


def test_known_hf_oscillation():
    t = np.arange(0, 180, 0.8)
    lf, hf = hrv.lf_hf(t, 800 + 30 * np.sin(2 * np.pi * 0.25 * t))
    assert hf > 10 * lf


def test_end_to_end():
    r = process(simulate())
    assert abs(r["fit"].offset_s - 2.4) < 0.002
    np.testing.assert_allclose(r["task"], [200, 280], atol=0.002)
    assert r["keep"].mean() > 0.95
