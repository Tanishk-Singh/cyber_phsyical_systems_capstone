"""Tests for the windowing utility. The first group uses small made-up arrays
(the worked examples from windows.py's docstrings); the last test checks the
real 82 s gap in the SWaT Attack file."""
import numpy as np

from ics_tsfm.data.swat import TIMESTAMP_COL
from ics_tsfm.data.windows import contiguous_anchors, scored_rows, window_batches


def seconds(n, gap_after=None, gap_s=82):
    """n timestamps one second apart, optionally with a jump after one row."""
    steps = np.ones(n, int)
    steps[0] = 0
    if gap_after is not None:
        steps[gap_after + 1] = gap_s
    return np.datetime64("2015-12-28T10:00:00") + np.cumsum(steps).astype("timedelta64[s]")


# --- which rows can be scored ---------------------------------------------------


def test_worked_example_from_the_docstring():
    clean = contiguous_anchors(seconds(9, gap_after=4), history=2)
    assert clean.tolist() == [False, False, True, True, True, False, False, True, True]


def test_no_gap_means_only_the_first_rows_are_skipped():
    clean = contiguous_anchors(seconds(100), history=10)
    assert np.flatnonzero(~clean).tolist() == list(range(10))


def test_history_zero_scores_every_row_even_across_a_gap():
    assert contiguous_anchors(seconds(20, gap_after=9), history=0).all()


def test_scored_rows_follow_the_longest_history():
    ts = seconds(200, gap_after=99)
    rows = scored_rows(ts, histories=[0, 50, 10])
    skipped = sorted(set(range(200)) - set(rows.tolist()))
    assert skipped == list(range(50)) + list(range(100, 150))


def test_no_scored_sample_crosses_the_gap():
    ts = seconds(1000, gap_after=499)
    history = 30
    for i in scored_rows(ts, [history]):
        span = ts[i - history:i + 1]
        assert (np.diff(span) == np.timedelta64(1, "s")).all()


# --- cutting the windows ---------------------------------------------------------


def test_window_batches_example_from_the_docstring():
    x = np.array([[10], [11], [12], [13], [14]], float)
    (windows, targets, anchors), = window_batches(x, np.array([3, 4]), history=3)
    assert windows[:, :, 0].tolist() == [[10, 11, 12], [11, 12, 13]]
    assert targets[:, 0].tolist() == [13, 14]
    assert anchors.tolist() == [3, 4]


def test_batches_cover_every_anchor_in_order_and_match_plain_slicing():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 4))
    anchors = np.array([60, 61, 99, 250, 251, 499])
    got = list(window_batches(x, anchors, history=50, batch_size=4))
    assert [len(b[2]) for b in got] == [4, 2]
    windows = np.concatenate([b[0] for b in got])
    targets = np.concatenate([b[1] for b in got])
    for k, i in enumerate(anchors):
        assert np.array_equal(windows[k], x[i - 50:i])
        assert np.array_equal(targets[k], x[i])


def test_history_zero_gives_empty_windows_and_the_row_as_target():
    x = np.arange(12, dtype=float).reshape(6, 2)
    (windows, targets, _), = window_batches(x, np.arange(6), history=0)
    assert windows.shape == (6, 0, 2)
    assert np.array_equal(targets, x)


# --- the real gap ----------------------------------------------------------------


def test_swat_attack_file_gap_skips_50_rows_after_row_298836(swat_attack_df):
    ts = swat_attack_df[TIMESTAMP_COL].to_numpy()
    rows = scored_rows(ts, histories=[0, 50])
    skipped = sorted(set(range(len(ts))) - set(rows.tolist()))
    assert skipped == list(range(50)) + list(range(298_837, 298_887))
