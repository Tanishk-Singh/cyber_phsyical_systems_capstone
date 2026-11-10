"""SWaT preprocessing (Phase A, Step 3): startup cutoff, columns, split, scaling.

Order matters, and each step only ever looks at training rows:
  1. Cut the startup from the Normal file (first 3.75 h).
  2. Keep the model columns (36 by default).
  3. Cut the rest into 75-minute blocks; 20% of blocks go to validation.
  4. Fit a StandardScaler on the training blocks only.
  5. Apply that same scaler to validation and to the Attack file.

Anchors (which samples a windowed model may use) depend on the model's history
length, so they are not fixed here: call `SwatSplit.anchors(history)`.

Where the numbers come from (analysis scripts in codebase/):
  - 75-minute blocks: just above the longest normal plant cycle, 72.9 min
    (swat_cycle_analysis.py).
  - 3.75 h cutoff: by then the tanks have filled and 30 of 36 kept columns have
    settled (swat_startup_analysis.py). The later stage-2 dosing change
    (until about hour 20) is kept on purpose: it is normal plant behaviour.
  - Columns: Fung et al.'s SWAT-CLEAN drops 17; we keep two of them back.
    AIT202 drifts little once the early dosing pulses are in training, and
    attack 6 forces it to 6.0; P205 drifts little and is attack 24's target.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ics_tsfm.data.split import assign_blocks, split_anchors
from ics_tsfm.data.swat import (
    TIMESTAMP_COL,
    corrected_attack_mask,
    sensor_columns,
)

BLOCK_ROWS = 75 * 60                  # 75 minutes at 1 Hz
STARTUP_CUTOFF_ROWS = 3 * BLOCK_ROWS  # 3.75 h = 13,500 rows
VAL_SHARE = 0.2

# Fung et al.'s SWAT-CLEAN (data_loader.py): 6 actuators constant in both
# files, plus 11 columns flagged by their K-S* drift test.
FUNG_CONSTANT_COLUMNS = ("P202", "P401", "P404", "P502", "P601", "P603")
FUNG_KS_COLUMNS = (
    "AIT201", "AIT202", "AIT203", "AIT401", "AIT402", "AIT501", "AIT502",
    "AIT503", "PIT502", "P201", "P205",
)
KEPT_BACK = ("AIT202", "P205")

COLUMN_SETS = {
    "ours": tuple(c for c in FUNG_CONSTANT_COLUMNS + FUNG_KS_COLUMNS if c not in KEPT_BACK),
    "fung": FUNG_CONSTANT_COLUMNS + FUNG_KS_COLUMNS,
    "all": (),
}
"""Columns DROPPED by each set: 'ours' keeps 36, 'fung' 34, 'all' 51."""


def model_columns(df: pd.DataFrame, column_set: str = "ours") -> list[str]:
    """The columns a model sees, in the file's original order."""
    if column_set not in COLUMN_SETS:
        raise ValueError(f"column_set must be one of {list(COLUMN_SETS)}, got {column_set!r}")
    dropped = set(COLUMN_SETS[column_set])
    return [c for c in sensor_columns(df) if c not in dropped]


@dataclass
class SwatSplit:
    """Everything Step 4 onward needs, already scaled.

    `normal` holds the Normal file after the startup cutoff; `is_val` marks its
    validation rows. Rows of `normal` are numbered from 0 at the cutoff, so
    normal row 0 is row 13,500 of the original file.
    """
    columns: list[str]
    normal: np.ndarray            # (481,500, n_columns), scaled
    normal_timestamps: np.ndarray
    is_val: np.ndarray            # (481,500,) True = validation row
    test: np.ndarray              # (449,919, n_columns), scaled with the same scaler
    test_timestamps: np.ndarray   # one 82 s gap at row 298,837: windows must not span it
    test_is_attack: np.ndarray    # corrected labels (Fung et al.'s 32 ranges)
    scaler: StandardScaler

    def anchors(self, history: int) -> tuple[np.ndarray, np.ndarray]:
        """(train_anchors, val_anchors) into `normal` for a model that reads
        `history` rows before each target row. history = 0 for point-wise
        models (every row is its own sample)."""
        return split_anchors(self.is_val, history)


def prepare_swat(normal_df: pd.DataFrame, attack_df: pd.DataFrame,
                 seed: int = 0, column_set: str = "ours") -> SwatSplit:
    """Run Step 3 end to end on the two loaded SWaT files.

    `seed` picks which blocks go to validation; keep it fixed to reproduce a
    run, vary it to check that results do not hinge on one lucky draw.
    """
    columns = model_columns(normal_df, column_set)
    if model_columns(attack_df, column_set) != columns:
        raise ValueError("Normal and Attack files have different sensor columns.")

    steady = normal_df.iloc[STARTUP_CUTOFF_ROWS:]
    x_normal = steady[columns].to_numpy(float)
    is_val = assign_blocks(len(x_normal), BLOCK_ROWS, VAL_SHARE, seed)

    # Fit on training rows only. A column that never changes in training
    # (P402, P501, UV401 after the cutoff) has zero spread; StandardScaler then
    # divides by 1, so it is 0 throughout training and an attack that switches
    # it off shows up as -1. It is kept, never dropped automatically.
    scaler = StandardScaler().fit(x_normal[~is_val])

    return SwatSplit(
        columns=columns,
        normal=scaler.transform(x_normal),
        normal_timestamps=steady[TIMESTAMP_COL].to_numpy(),
        is_val=is_val,
        test=scaler.transform(attack_df[columns].to_numpy(float)),
        test_timestamps=attack_df[TIMESTAMP_COL].to_numpy(),
        test_is_attack=corrected_attack_mask(attack_df).to_numpy(),
        scaler=scaler,
    )
