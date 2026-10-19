import os

import pytest

from ics_tsfm.data.swat import load_swat_attack, load_swat_normal

_DEFAULT_SWAT_DIR = r"E:\UniversityOfArizona\Coursework\Capstone\$$SWAT Dataset A2-Dec 2015"


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
