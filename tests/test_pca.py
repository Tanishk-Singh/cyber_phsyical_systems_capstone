"""Tests for the PCA detectors. Small made-up data first (checked against
sklearn and against cases you can reason about by hand), then the real split."""
import numpy as np
import pytest
from sklearn.decomposition import PCA

from ics_tsfm.models.pca import PCADetector


def correlated(n=2000, seed=0):
    """Three columns driven by two hidden signals, plus a little noise:
    two directions carry almost all the variation, the third is noise."""
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=(n, 2))
    mix = np.array([[1.0, 0.5, -0.3], [0.2, -1.0, 0.8]])
    return hidden @ mix + 0.01 * rng.normal(size=(n, 3))


# --- plain PCA against sklearn --------------------------------------------------


def test_matches_sklearn_explained_variance_and_reconstruction_error():
    x = correlated()
    rows = np.arange(len(x))
    det = PCADetector(window=1, n_components=2).fit(x, rows)
    ref = PCA(n_components=3).fit(x)
    assert np.allclose(det.explained_variance_ratio_, ref.explained_variance_ratio_, atol=1e-6)

    ref2 = PCA(n_components=2).fit(x)
    ref_err = ((x - ref2.inverse_transform(ref2.transform(x))) ** 2).sum(axis=1)
    assert np.allclose(det.score(x, rows), ref_err, rtol=1e-4, atol=1e-10)


def test_variance_rule_picks_the_two_real_directions():
    x = correlated()
    det = PCADetector(window=1, variance=0.99).fit(x, np.arange(len(x)))
    assert det.k_ == 2


def test_a_row_that_breaks_the_relationship_scores_high():
    x = correlated()
    det = PCADetector(window=1, variance=0.99).fit(x, np.arange(len(x)))
    normal_row = x[:1]
    broken_row = normal_row + np.array([[0.0, 0.0, 3.0]])   # third column off its pattern
    normal, broken = det.score(np.vstack([normal_row, broken_row]), np.array([0, 1]))
    assert broken > 100 * normal


def test_all_directions_rebuild_everything_perfectly():
    # the "lenient guard": with every direction kept, even the broken row scores 0
    x = correlated()
    det = PCADetector(window=1, n_components=3).fit(x, np.arange(len(x)))
    broken = x[:1] + np.array([[0.0, 0.0, 3.0]])
    assert det.score(broken, np.array([0]))[0] < 1e-12


# --- windowed PCA ----------------------------------------------------------------


def test_windowed_equals_sklearn_on_explicitly_flattened_windows():
    x = correlated(n=600)
    window = 4
    anchors = np.arange(window - 1, len(x))
    flat = np.stack([x[i - window + 1:i + 1].ravel() for i in anchors])
    det = PCADetector(window=window, n_components=5).fit(x, anchors)
    ref = PCA(n_components=5).fit(flat)
    ref_err = ((flat - ref.inverse_transform(ref.transform(flat))) ** 2).sum(axis=1)
    assert np.allclose(det.score(x, anchors), ref_err, rtol=1e-4, atol=1e-10)


def test_batch_size_does_not_change_the_result():
    x = correlated(n=500)
    anchors = np.arange(9, len(x))
    a = PCADetector(window=10, n_components=4).fit(x, anchors, batch_size=7)
    b = PCADetector(window=10, n_components=4).fit(x, anchors, batch_size=10_000)
    assert np.allclose(a.score(x, anchors, batch_size=13), b.score(x, anchors))


def test_windowed_pca_catches_a_frozen_sensor_that_plain_pca_misses():
    # Two sensors that always move together. Freezing both at a normal value
    # keeps every single row normal; only the flat stretch over time is wrong.
    t = np.arange(4000)
    rng = np.random.default_rng(0)
    signal = np.sin(t / 20)
    x = np.column_stack([signal, signal]) + 0.01 * rng.normal(size=(4000, 2))
    train = np.arange(30, 3000)
    frozen = x.copy()
    frozen[3500:3700] = [0.5, 0.5]
    test = np.arange(3530, 3700)                     # windows fully inside the freeze

    plain = PCADetector(window=1, variance=0.99).fit(x, train)
    windowed = PCADetector(window=30, variance=0.99).fit(x, train)
    normal_rows = np.arange(3030, 3400)
    plain_ratio = plain.score(frozen, test).mean() / plain.score(x, normal_rows).mean()
    windowed_ratio = windowed.score(frozen, test).mean() / windowed.score(x, normal_rows).mean()
    # measured: plain about 0.0x (a frozen row is noise-free, so it fits even
    # better than usual), windowed about 20x
    assert plain_ratio < 1.5
    assert windowed_ratio > 5


def test_score_series_is_nan_where_not_scored():
    x = correlated(n=100)
    det = PCADetector(window=5, n_components=2).fit(x, np.arange(4, 100))
    series = det.score_series(x, np.arange(10, 100))
    assert np.isnan(series[:10]).all()
    assert not np.isnan(series[10:]).any()


def test_window_must_be_positive():
    with pytest.raises(ValueError, match="window"):
        PCADetector(window=0)


# --- on the real split -------------------------------------------------------------


def test_plain_pca_on_swat_keeps_18_directions(swat_split):
    det = PCADetector(window=1, variance=0.99)
    train, val = swat_split.anchors(det.history)
    det.fit(swat_split.normal, train)
    assert det.k_ == 18                              # also half of the 36 columns
    val_scores = det.score(swat_split.normal, val)
    assert np.isfinite(val_scores).all() and (val_scores >= 0).all()
