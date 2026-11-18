"""Tests for the Step 8 pieces: attack types, the TSB-AD metric wrappers, and
PCA scores for several k in one pass."""
import numpy as np
import pytest

from ics_tsfm.data.swat_attacks import attack_points, attack_type, load_attack_types
from ics_tsfm.data.swat_corrections import CORRECTED_ATTACK_RANGES
from ics_tsfm.metrics.threshold_free import range_f1, roc_auc, vus
from ics_tsfm.models.pca import PCADetector


# --- attack types ------------------------------------------------------------------


def test_attack_points_and_types():
    assert attack_points("P-101; P-102") == ["P-101", "P-102"]
    assert attack_points("UV-401, AIT-502, P-501") == ["UV-401", "AIT-502", "P-501"]
    assert attack_type(["LIT-101"]) == "SSSP"
    assert attack_type(["P-203", "P-205"]) == "SSMP"
    assert attack_type(["P-101", "LIT-301"]) == "MSMP"


def test_all_32_corrected_attacks_get_a_type(swat_dir):
    types = load_attack_types(swat_dir)
    numbers = [n for _, _, n in CORRECTED_ATTACK_RANGES]
    assert set(numbers) <= set(types.index)
    counts = types.loc[numbers, "type"].value_counts().to_dict()
    assert counts == {"SSSP": 20, "MSMP": 7, "SSMP": 5}
    assert types.loc[26, "type"] == "MSMP" and types.loc[24, "type"] == "SSMP"


# --- metric wrappers --------------------------------------------------------------------


def toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, int)
    y[1000:1300] = 1
    y[3000:3100] = 1
    s = rng.random(n)
    return y, s


def test_perfect_scores_get_perfect_auc_and_vus():
    y, _ = toy()
    s = y + 0.001 * np.arange(len(y)) / len(y)          # attacks strictly above normal rows
    assert roc_auc(s, y) == pytest.approx(1.0)
    assert vus(s, y)[0] > 0.95


def test_random_scores_are_near_chance():
    y, s = toy()
    assert roc_auc(s, y) == pytest.approx(0.5, abs=0.05)
    assert vus(s, y)[0] < 0.3


def test_range_f1_uses_the_given_alarms():
    y, _ = toy()
    assert range_f1(y, y) == pytest.approx(1.0)
    assert range_f1(y, np.zeros_like(y)) == 0.0
    half = np.zeros_like(y)
    half[1000:1300] = 1                                    # first attack caught, second missed
    assert 0.3 < range_f1(y, half) < 1.0


# --- PCA scores for several k in one pass ---------------------------------------------------


def test_score_for_each_k_matches_separate_fits():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, 3)) @ rng.normal(size=(3, 6)) + 0.05 * rng.normal(size=(800, 6))
    anchors = np.arange(4, 800)
    big = PCADetector(window=5, n_components=12).fit(x, anchors)
    many = big.score_for_each_k(x, anchors, ks=[2, 5, 12])
    for k in (2, 5, 12):
        single = PCADetector(window=5, n_components=k).fit(x, anchors).score(x, anchors)
        assert np.allclose(many[k], single, rtol=1e-6, atol=1e-9)


def test_score_for_each_k_rejects_k_above_the_fit():
    x = np.random.default_rng(1).normal(size=(200, 4))
    det = PCADetector(window=1, n_components=2).fit(x, np.arange(200))
    with pytest.raises(ValueError, match="larger than"):
        det.score_for_each_k(x, np.arange(200), ks=[3])
