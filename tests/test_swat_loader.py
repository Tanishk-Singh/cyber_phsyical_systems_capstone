"""Regression tests encoding the exact facts verified by hand in the
exploratory EDA (codebase/data_analysis.ipynb). If these ever fail, either the
dataset on disk changed/got corrupted, or the loader broke -- not a false alarm.
"""
from ics_tsfm.data.swat import LABEL_COL, attack_segments, is_attack_mask, sensor_columns


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
