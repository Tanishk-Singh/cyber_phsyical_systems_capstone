"""Regression tests encoding the exact facts verified by hand in the
exploratory EDA (codebase/data_analysis.ipynb). If these ever fail, either the
dataset on disk changed/got corrupted, or the loader broke -- not a false alarm.
"""
import pytest

from ics_tsfm.data.swat import (
    LABEL_COL,
    attack_segments,
    corrected_attack_mask,
    is_attack_mask,
    sensor_columns,
)
from ics_tsfm.data.swat_corrections import (
    BENIGN_ATTACK_RANGES,
    CORRECTED_ATTACK_RANGES,
    PHANTOM_RANGE,
)


def test_attack_file_shape(swat_attack_df):
    assert swat_attack_df.shape == (449919, 53)


def test_normal_file_shape(swat_normal_df):
    assert swat_normal_df.shape == (495000, 53)


def test_sensor_column_count(swat_attack_df):
    assert len(sensor_columns(swat_attack_df)) == 51


def test_columns_match_between_files(swat_normal_df, swat_attack_df):
    # Both files have the same 53 columns once whitespace is stripped (the
    # loader already strips it) -- including the label column in the Normal
    # file, which is just constant "Normal".
    assert set(swat_normal_df.columns) == set(swat_attack_df.columns)


def test_normal_file_label_is_constant_normal(swat_normal_df):
    counts = swat_normal_df[LABEL_COL].value_counts()
    assert counts.to_dict() == {"Normal": 495000}


def test_label_typo_is_counted_as_attack(swat_attack_df):
    counts = swat_attack_df[LABEL_COL].value_counts()
    assert counts["Normal"] == 395298
    assert counts["Attack"] == 54584
    assert counts["A ttack"] == 37  # the stray-space typo

    mask = is_attack_mask(swat_attack_df)
    assert mask.sum() == 54584 + 37  # substring match catches the typo too


def test_attack_segments(swat_attack_df):
    mask = is_attack_mask(swat_attack_df)
    segments = attack_segments(mask)
    attack_only = segments[segments["is_attack"]]

    assert len(attack_only) == 35
    assert attack_only["length"].min() == 101
    assert attack_only["length"].max() == 35900  # the ~10h dominant segment

    # segment boundaries must actually line up with the mask
    longest = attack_only.loc[attack_only["length"].idxmax()]
    assert mask.iloc[longest["start"]:longest["end"] + 1].all()


# --- corrected labels (Fung et al.) ------------------------------------------


def test_corrected_ranges_are_sorted_and_disjoint():
    ends = [e for _, e, _ in CORRECTED_ATTACK_RANGES]
    starts = [s for s, _, _ in CORRECTED_ATTACK_RANGES]
    assert all(s < e for s, e, _ in CORRECTED_ATTACK_RANGES)
    # strictly increasing with a real gap, so no two windows merge into one
    # segment and the 32 windows stay 32 segments
    assert all(ends[i] < starts[i + 1] for i in range(len(starts) - 1))


def test_corrected_mask_yields_exactly_the_published_windows(swat_attack_df):
    mask = corrected_attack_mask(swat_attack_df)
    segments = attack_segments(mask)
    attack_only = segments[segments["is_attack"]].reset_index(drop=True)

    assert len(attack_only) == 32
    assert list(attack_only["start"]) == [s for s, _, _ in CORRECTED_ATTACK_RANGES]
    # attack_segments reports an inclusive end; the ranges are half-open
    assert list(attack_only["end"]) == [e - 1 for _, e, _ in CORRECTED_ATTACK_RANGES]


def test_phantom_segment_is_attack_in_raw_labels_but_normal_once_corrected(swat_attack_df):
    lo, hi = PHANTOM_RANGE
    assert is_attack_mask(swat_attack_df).iloc[lo:hi].all()
    assert not corrected_attack_mask(swat_attack_df).iloc[lo:hi].any()


def test_benign_attacks_are_normal_once_corrected(swat_attack_df):
    raw = is_attack_mask(swat_attack_df)
    corrected = corrected_attack_mask(swat_attack_df)
    for lo, hi, _ in BENIGN_ATTACK_RANGES:
        assert raw.iloc[lo:hi].all()
        assert not corrected.iloc[lo:hi].any()


def test_normal_stretch_between_attacks_27_and_28_is_recovered(swat_attack_df):
    # the xlsx runs 27 and 28 together across 1157s of genuinely normal data
    assert is_attack_mask(swat_attack_df).iloc[228362:229519].all()
    assert not corrected_attack_mask(swat_attack_df).iloc[228362:229519].any()


def test_correction_reduces_attack_labeled_timesteps(swat_attack_df):
    raw = is_attack_mask(swat_attack_df).sum()
    corrected = corrected_attack_mask(swat_attack_df).sum()
    assert corrected < raw
    # the dominant segment shrinks from 35900s to 34208s, so it no longer
    # swallows two thirds of every point-wise metric
    segments = attack_segments(corrected_attack_mask(swat_attack_df))
    assert segments[segments["is_attack"]]["length"].max() == 34208


def test_corrected_mask_refuses_the_normal_file(swat_normal_df):
    with pytest.raises(ValueError, match="row-indexed"):
        corrected_attack_mask(swat_normal_df)
