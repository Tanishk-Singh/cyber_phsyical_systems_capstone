"""Tests for the Step 7 detection rule and alarm scoring (small made-up arrays,
including the worked examples from the docstrings)."""
import numpy as np
import pytest

from ics_tsfm.detect.rule import alarms, events, run_lengths, zero_alarm_threshold
from ics_tsfm.metrics.alarms import score_alarms


# --- the rule --------------------------------------------------------------------


def test_worked_example_from_the_docstring():
    scores = np.array([1, 5, 6, 7, 2, 8, 9, 4], float)
    assert alarms(scores, tau=4, w=3).tolist() == [0, 0, 0, 1, 0, 0, 0, 0]


def test_run_lengths_and_events_examples():
    flag = np.array([0, 1, 1, 0, 1], bool)
    assert run_lengths(flag).tolist() == [0, 1, 2, 0, 1]
    assert events(flag).tolist() == [[1, 3], [4, 5]]


def test_w_one_alarms_on_every_row_above_tau():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    assert alarms(scores, tau=0.5, w=1).tolist() == [0, 1, 0, 1]


def test_nan_rows_break_a_run():
    scores = np.array([9, 9, np.nan, 9, 9, 9], float)
    assert alarms(scores, tau=1, w=3).tolist() == [0, 0, 0, 0, 0, 1]


# --- zero false alarms on validation -----------------------------------------------


def test_zero_alarm_threshold_is_the_highest_of_the_stretch_minimums():
    val = np.array([1, 5, 6, 2, 7, 8, 3], float)
    # stretches of 2: mins 1, 5, 2, 2, 7, 3 -> highest 7 (the pair 7, 8)
    assert zero_alarm_threshold(val, w=2) == 7
    assert zero_alarm_threshold(val, w=1) == 8              # w = 1: the highest score


def test_zero_alarm_threshold_really_gives_zero_validation_alarms():
    rng = np.random.default_rng(0)
    val = rng.gamma(2.0, size=5000)
    val[rng.choice(5000, 300, replace=False)] = np.nan     # unscored rows, like seams
    for w in (1, 5, 10, 30):
        tau = zero_alarm_threshold(val, w)
        assert not alarms(val, tau, w).any()
        # and it is the lowest such tau: just below it, something fires
        assert alarms(val, np.nextafter(tau, -np.inf), w).any()


def test_longer_persistence_never_raises_the_threshold():
    rng = np.random.default_rng(1)
    val = rng.gamma(2.0, size=5000)
    taus = [zero_alarm_threshold(val, w) for w in (1, 5, 10, 30)]
    assert taus == sorted(taus, reverse=True)


def test_stretches_containing_nan_are_skipped():
    val = np.array([1, 100, np.nan, 2, 3], float)
    # the only clean stretches of 2 are (1, 100) -> min 1 and (2, 3) -> min 2
    assert zero_alarm_threshold(val, w=2) == 2


def test_no_clean_stretch_is_an_error():
    with pytest.raises(ValueError, match="no stretch"):
        zero_alarm_threshold(np.array([1, np.nan, 2, np.nan], float), w=2)


# --- scoring the alarms --------------------------------------------------------------


def test_score_alarms_counts_rows_attacks_delays_and_false_events():
    #           0  1  2  3  4  5  6  7  8  9 10 11
    y     = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1], bool)    # attacks 2-5 and 10-11
    alarm = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0], bool)
    scored = np.ones(12, bool)
    r = score_alarms(alarm, y, scored, [(2, 6, 1), (10, 12, 2)])
    assert r["attacks_detected"] == 1 and r["attacks_total"] == 2
    assert r["per_attack"][0]["delay_s"] == 1           # first alarm at row 3, attack began at 2
    assert r["per_attack"][1]["detected"] is False
    # runs: [0,1) false, [3,7) overlaps attack 1 (true, its tail row 6 still counts
    # as a false-positive ROW), [8,9) false
    assert r["false_alarm_events"] == 2
    assert r["point_precision"] == pytest.approx(3 / 6)
    assert r["point_recall"] == pytest.approx(3 / 6)


def test_unscored_rows_are_ignored():
    y = np.array([0, 1, 1, 0], bool)
    alarm = np.array([1, 1, 1, 1], bool)
    scored = np.array([False, True, True, True])
    r = score_alarms(alarm, y, scored, [(1, 3, 1)])
    assert r["false_alarm_events"] == 0                  # row 0 is not scored; row 3 extends the true run
    assert r["point_precision"] == pytest.approx(2 / 3)
