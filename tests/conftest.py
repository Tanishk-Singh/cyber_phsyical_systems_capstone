import os
from pathlib import Path

import pytest

from ics_tsfm.data.swat import load_swat_attack, load_swat_normal
from ics_tsfm.data.swat_preprocess import prepare_swat

# Default: the data folder inside this clone, so the tests find it on any
# machine without SWAT_DIR set (same rule as codebase/swat_cycle_analysis.py).
_DEFAULT_SWAT_DIR = str(Path(__file__).resolve().parents[1] / "$$SWAT Dataset A2-Dec 2015")


@pytest.fixture(scope="session")
def swat_dir():
    path = os.environ.get("SWAT_DIR", _DEFAULT_SWAT_DIR)
    if not os.path.isdir(path):
        pytest.skip(f"SWaT data not found at {path!r} (set the SWAT_DIR env var to override)")
    return path


@pytest.fixture(scope="session")
def swat_attack_df(swat_dir):
    return load_swat_attack(swat_dir)


@pytest.fixture(scope="session")
def swat_normal_df(swat_dir):
    return load_swat_normal(swat_dir)


@pytest.fixture(scope="session")
def swat_split(swat_normal_df, swat_attack_df):
    """Step 3 output with the default settings (seed 0, 36 columns)."""
    return prepare_swat(swat_normal_df, swat_attack_df, seed=0)
