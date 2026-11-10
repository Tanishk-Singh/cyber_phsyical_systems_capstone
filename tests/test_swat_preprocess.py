"""Step 3 on the real SWaT files: the decisions made in discussion, checked
against the data. Uses the session fixtures in conftest.py (both files load
once, about 40 s each)."""
import numpy as np
import pytest

from ics_tsfm.data.swat_preprocess import (
    BLOCK_ROWS,
    STARTUP_CUTOFF_ROWS,
    model_columns,
    prepare_swat,
)
from ics_tsfm.data.swat import TIMESTAMP_COL, corrected_attack_mask


@pytest.fixture(scope="module")
def split(swat_normal_df, swat_attack_df):
    return prepare_swat(swat_normal_df, swat_attack_df, seed=0)


def test_column_sets(swat_normal_df):
    ours = model_columns(swat_normal_df)
    assert len(ours) == 36
    assert len(model_columns(swat_normal_df, "fung")) == 34
    assert len(model_columns(swat_normal_df, "all")) == 51
    # kept back from Fung's removals, and kept although it drifts
    for col in ("AIT202", "P205", "AIT504"):
        assert col in ours
    for col in ("AIT201", "AIT402", "AIT502", "P201", "P202", "P603"):
        assert col not in ours


def test_startup_cutoff_is_3_75_hours(split, swat_normal_df):
    assert STARTUP_CUTOFF_ROWS == 13_500
    assert len(split.normal) == 495_000 - 13_500
    assert split.normal_timestamps[0] == swat_normal_df[TIMESTAMP_COL].iloc[13_500]


def test_107_blocks_of_75_minutes_21_for_validation(split):
    # 481,500 rows / 4,500 = 107 whole blocks; round(107 * 0.2) = 21
    assert len(split.normal) % BLOCK_ROWS == 0
    block_sides = split.is_val[::BLOCK_ROWS]
    assert len(block_sides) == 107
    assert block_sides.sum() == 21


def test_no_row_shared_between_train_and_validation_samples(split):
    history = 50
    train, val = split.anchors(history)
    covered = np.zeros((2, len(split.normal)), bool)
    for side, anchors in enumerate((train, val)):
        # mark rows [i - history, i] for every anchor via a difference array
        diff = np.zeros(len(split.normal) + 1, int)
        np.add.at(diff, anchors - history, 1)
        np.add.at(diff, anchors + 1, -1)
        covered[side] = np.cumsum(diff)[:-1] > 0
    assert not (covered[0] & covered[1]).any()


def test_scaler_fit_on_training_rows_only(split, swat_normal_df):
    train_rows = split.normal[~split.is_val]
    changing = split.scaler.var_ > 0
    assert np.allclose(train_rows.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(train_rows[:, changing].std(axis=0), 1, atol=1e-9)
    # and it really is different from fitting on the whole steady file
    raw = swat_normal_df[split.columns].to_numpy(float)[STARTUP_CUTOFF_ROWS:]
    assert not np.allclose(split.scaler.mean_, raw.mean(axis=0))


def test_actuators_constant_in_training_are_kept_and_attack_shows_as_minus_one(split):
    # P402, P501, UV401 are on (2) for the whole steady Normal file; attacks
    # 22, 28, 40 and 10 switch them off (1).
    for col in ("P402", "P501", "UV401"):
        j = split.columns.index(col)
        assert split.scaler.scale_[j] == 1.0
        assert np.all(split.normal[:, j] == 0)
        off = split.test[:, j] == -1
        assert off.sum() > 30_000
        assert split.test_is_attack[off].mean() > 0.99


def test_test_set_is_the_whole_attack_file_with_corrected_labels(split, swat_attack_df):
    assert split.test.shape == (449_919, 36)
    assert np.array_equal(split.test_is_attack, corrected_attack_mask(swat_attack_df).to_numpy())


def test_unknown_column_set_is_rejected(swat_normal_df):
    with pytest.raises(ValueError, match="column_set"):
        model_columns(swat_normal_df, "mine")
