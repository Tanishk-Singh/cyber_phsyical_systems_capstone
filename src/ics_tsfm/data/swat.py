"""SWaT loader: everything the exploratory EDA (codebase/data_analysis.ipynb)
found about this dataset, turned into a reusable, self-checking module.

Scope note: this module only loads and cleans the two files -- it does NOT
decide how to treat constant/discrete-actuator columns or trim the startup
transient. Those are modeling decisions that belong to Step 3
(preprocessing), not the loader.
"""
import warnings
from pathlib import Path

import pandas as pd

from ics_tsfm.data._xlsx_fast import fast_xlsx_rows, rows_to_df
from ics_tsfm.data.swat_corrections import CORRECTED_ATTACK_RANGES, N_ATTACK_ROWS

TIMESTAMP_COL = "Timestamp"
LABEL_COL = "Normal/Attack"
N_SENSOR_COLS = 51

# The two files ship with DIFFERENT header shapes (found in notebook Step 8):
# Attack file has a single header row; Normal file has a merged two-row header
# (process-stage group labels in row 0, real sensor names in row 1).
_ATTACK_HEADER_ROW = 0
_NORMAL_HEADER_ROW = 1

_ATTACK_FILENAME = "SWaT_Dataset_Attack_v0.xlsx"
_NORMAL_FILENAME = "SWaT_Dataset_Normal_v1.xlsx"


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """The 51 sensor/actuator columns -- everything except Timestamp and the label."""
    return [c for c in df.columns if c not in (TIMESTAMP_COL, LABEL_COL)]


def is_attack_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask, True where labeled attack. Substring match, NOT exact
    equality -- the attack file has 37 rows labeled 'A ttack' (stray space)
    that `label == 'Attack'` would silently miss."""
    return df[LABEL_COL].astype(str).str.contains("ttack", case=False, na=False)


def corrected_attack_mask(df: pd.DataFrame) -> pd.Series:
    """Ground-truth attack mask built from Fung et al.'s corrected row ranges.

    Replaces the file's own label column outright rather than patching it, so
    anything the attack list does not account for becomes Normal -- including
    the phantom AM/PM segment and the three attacks that left no trace in the
    recorded columns. See swat_corrections.py for why the raw labels are wrong.

    Use this, not `is_attack_mask`, for evaluation; `is_attack_mask` stays
    available so the writeup can quantify what the correction changes."""
    if len(df) != N_ATTACK_ROWS:
        raise ValueError(
            f"Corrected labels are row-indexed against the {N_ATTACK_ROWS}-row SWaT "
            f"attack file, but this DataFrame has {len(df)} rows -- they cannot be "
            f"applied to it (the Normal file, or an already-sliced frame)."
        )
    mask = pd.Series(False, index=df.index)
    for start, end, _ in CORRECTED_ATTACK_RANGES:
        mask.iloc[start:end] = True
    return mask


def attack_segments(is_attack: pd.Series) -> pd.DataFrame:
    """One row per contiguous run of the mask (both attack AND normal runs),
    with its boundaries. Filter `df["is_attack"]` to keep only attack runs."""
    seg_id = (is_attack != is_attack.shift()).cumsum()
    position = pd.Series(range(len(is_attack)), index=is_attack.index)
    return pd.DataFrame({
        "is_attack": is_attack.groupby(seg_id).first(),
        "length": is_attack.groupby(seg_id).size(),
        "start": position.groupby(seg_id).first(),
        "end": position.groupby(seg_id).last(),
    }).reset_index(drop=True)


def _load_and_clean(path: Path, header_row_idx: int, source: str) -> pd.DataFrame:
    raw = fast_xlsx_rows(str(path))
    df = rows_to_df(raw, header_row_idx=header_row_idx)
    df.columns = [c.strip() for c in df.columns]

    if TIMESTAMP_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(
            f"SWaT {source} file doesn't have the expected '{TIMESTAMP_COL}'/'{LABEL_COL}' "
            f"columns after loading (got {list(df.columns)[:6]}...). The Normal and Attack "
            f"files have DIFFERENT header row positions -- this usually means "
            f"header_row_idx={header_row_idx} is wrong for this file."
        )

    sensor_cols = sensor_columns(df)
    if len(sensor_cols) != N_SENSOR_COLS:
        raise ValueError(f"Expected {N_SENSOR_COLS} sensor columns in SWaT {source} file, got {len(sensor_cols)}")

    numeric = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    bad = numeric.isna().sum()
    bad = bad[bad > 0]
    if len(bad):
        raise ValueError(f"Non-numeric sensor values in SWaT {source} file (after coercion): {bad.to_dict()}")
    df[sensor_cols] = numeric

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL].str.strip(), format="%d/%m/%Y %I:%M:%S %p")

    gaps = df[TIMESTAMP_COL].diff().dropna()
    bad_gaps = gaps[gaps != pd.Timedelta(seconds=1)]
    if len(bad_gaps):
        warnings.warn(
            f"SWaT {source} file: {len(bad_gaps)} timestamp gap(s) != 1s "
            f"(largest: {bad_gaps.max()}) -- windowing must not span these."
        )

    n_dup = df.duplicated().sum()
    if n_dup:
        warnings.warn(f"SWaT {source} file: {n_dup} fully duplicate rows found.")

    return df.reset_index(drop=True)


def load_swat_attack(swat_dir: str | Path) -> pd.DataFrame:
    """Load SWaT_Dataset_Attack_v0.xlsx: Timestamp + 51 sensor cols + label col."""
    return _load_and_clean(Path(swat_dir) / _ATTACK_FILENAME, _ATTACK_HEADER_ROW, source="attack")


def load_swat_normal(swat_dir: str | Path) -> pd.DataFrame:
    """Load SWaT_Dataset_Normal_v1.xlsx: Timestamp + 51 sensor cols + label col
    (the label is constant "Normal" for every row -- present only for schema
    consistency with the attack file)."""
    return _load_and_clean(Path(swat_dir) / _NORMAL_FILENAME, _NORMAL_HEADER_ROW, source="normal")
