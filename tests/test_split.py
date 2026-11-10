"""Tests for the block split and anchor rule on small made-up arrays (no
dataset needed, runs in a second). The worked example is the one in
split_anchors' docstring, so it can be checked by hand."""
import numpy as np

from ics_tsfm.data.split import assign_blocks, split_anchors


def rows_touched(anchors, history):
    """Every row used by these samples: inputs [i - history, i) plus target i."""
    if len(anchors) == 0:
        return set()
    return set(np.concatenate([np.arange(i - history, i + 1) for i in anchors]).tolist())


# --- assign_blocks -----------------------------------------------------------


def test_whole_blocks_share_one_side():
    is_val = assign_blocks(n_rows=1000, block_rows=30, val_share=0.2, seed=0)
    for start in range(0, 1000, 30):
        assert len(set(is_val[start:start + 30])) == 1


def test_exact_number_of_validation_blocks():
    # 1000 rows / 30 = 34 blocks (the last has 10 rows); round(34 * 0.2) = 7
    is_val = assign_blocks(n_rows=1000, block_rows=30, val_share=0.2, seed=0)
    block_sides = is_val[::30]
    assert len(block_sides) == 34
    assert block_sides.sum() == 7


def test_same_seed_same_split_different_seed_different_split():
    a = assign_blocks(1000, 30, 0.2, seed=1)
    b = assign_blocks(1000, 30, 0.2, seed=1)
    c = assign_blocks(1000, 30, 0.2, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- split_anchors -------------------------------------------------------------


def test_worked_example_from_the_docstring():
    is_val = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], bool)
    train, val = split_anchors(is_val, history=2)
    assert train.tolist() == [2, 8]
    assert val.tolist() == [5]


def test_no_row_is_shared_between_train_and_validation_samples():
    # The core leak check, including target rows, on a realistic-sized case.
    is_val = assign_blocks(n_rows=50_000, block_rows=500, val_share=0.2, seed=0)
    train, val = split_anchors(is_val, history=50)
    assert rows_touched(train, 50).isdisjoint(rows_touched(val, 50))


def test_every_sample_stays_on_its_own_side():
    is_val = assign_blocks(n_rows=20_000, block_rows=500, val_share=0.2, seed=3)
    train, val = split_anchors(is_val, history=50)
    assert not any(is_val[sorted(rows_touched(train, 50))])
    assert all(is_val[sorted(rows_touched(val, 50))])


def test_history_anchors_dropped_per_label_change_seam_and_none_elsewhere():
    is_val = assign_blocks(n_rows=50_000, block_rows=500, val_share=0.2, seed=0)
    history = 50
    train, val = split_anchors(is_val, history)
    n_label_changes = int((is_val[1:] != is_val[:-1]).sum())
    n_possible = len(is_val) - history           # anchors with a full history
    assert n_possible - len(train) - len(val) == history * n_label_changes


def test_dropped_anchors_sit_at_the_start_of_the_later_block():
    is_val = np.repeat([False, True, False], 100)
    train, val = split_anchors(is_val, history=10)
    kept = set(train) | set(val)
    dropped = sorted(set(range(10, 300)) - kept)
    assert dropped == list(range(100, 110)) + list(range(200, 210))


def test_seam_between_two_blocks_on_the_same_side_drops_nothing():
    is_val = np.zeros(300, bool)                 # three training blocks in a row
    train, val = split_anchors(is_val, history=10)
    assert train.tolist() == list(range(10, 300))
    assert len(val) == 0


def test_history_zero_keeps_every_row():
    is_val = assign_blocks(n_rows=1000, block_rows=30, val_share=0.2, seed=0)
    train, val = split_anchors(is_val, history=0)
    assert len(train) + len(val) == 1000
    assert set(val.tolist()) == set(np.flatnonzero(is_val).tolist())
