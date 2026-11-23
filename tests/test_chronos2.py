"""Tests for the Chronos-2 adapter with the real pinned model (Step B3).

Needs chronos-forecasting and the model (about 480 MB, downloaded once into
the Hugging Face cache); skipped if either is missing. About 30 s on the
laptop's GPU after the first download.
"""
import numpy as np
import pytest

pytest.importorskip("chronos")

from ics_tsfm.models.tsfm import ForecastDetector  # noqa: E402
from ics_tsfm.models.tsfm.chronos2 import Chronos2Forecaster  # noqa: E402


@pytest.fixture(scope="module")
def chronos():
    try:
        return Chronos2Forecaster()
    except OSError as err:   # offline and not cached
        pytest.skip(f"Chronos-2 not available: {err}")


def lagged_pair(rng, context=512, horizon=8, lag=8):
    """Sensor A is pure noise; sensor B is A delayed by `lag` rows. B's next
    `lag` values are already in A's context, so a model that reads across
    sensors can forecast B; one that sees B alone cannot."""
    a = rng.standard_normal(context + horizon + lag).astype(np.float32)
    b = np.r_[np.zeros(lag, np.float32), a[:-lag]]
    x = np.stack([a, b])[:, lag:]
    return x[:, :context], x[:, context:]


def test_forecasts_have_one_value_per_series_variate_and_step(chronos):
    contexts = np.random.default_rng(0).standard_normal((5, 3, 64)).astype(np.float32)
    out = chronos.predict(contexts, 4)
    assert out.shape == (5, 3, 4) and out.dtype == np.float32 and np.isfinite(out).all()


def test_joint_mode_reads_the_other_sensors(chronos):
    rng = np.random.default_rng(0)
    joint, alone = [], []
    for _ in range(5):
        context, future = lagged_pair(rng)
        joint.append(np.abs(chronos.predict(context[None], 8)[0, 1] - future[1]).mean())
        alone.append(np.abs(chronos.predict(context[:, None], 8)[1, 0] - future[1]).mean())
    # measured 2026-11-23: joint 0.14-0.22, alone 0.74-1.07 (noise has spread 1)
    assert np.mean(joint) < 0.5 * np.mean(alone)


def test_per_sensor_mode_ignores_the_other_sensors_through_the_detector(chronos):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 2)).astype(np.float32)
    changed = x.copy()
    changed[:, 0] = rng.standard_normal(300)   # replace sensor 0 entirely
    rows = np.arange(256, 300)
    for joint, should_change in ((False, False), (True, True)):
        det = ForecastDetector(chronos, context=256, horizon=1, joint=joint, verbose=False)
        before, after = det.predict(x, rows)[:, 1], det.predict(changed, rows)[:, 1]
        assert (np.abs(before - after).max() > 1e-3) == should_change


def test_a_flat_sensor_gets_a_flat_finite_forecast(chronos):
    t = np.arange(512)
    context = np.stack([np.sin(t / 10), np.full(512, 2.5)]).astype(np.float32)
    out = chronos.predict(context[None], 16)[0]
    assert np.isfinite(out).all() and np.allclose(out[1], 2.5)


def test_forecasts_are_repeatable_and_do_not_depend_on_the_batch_size(chronos):
    contexts = np.random.default_rng(2).standard_normal((40, 36, 128)).astype(np.float32)
    first, second = chronos.predict(contexts, 4), chronos.predict(contexts, 4)
    assert np.array_equal(first, second)
    chronos.batch_size = 36   # one origin (36 series) per forward pass instead of ~8
    try:
        assert np.allclose(first, chronos.predict(contexts, 4), atol=1e-4)   # measured max diff ~6e-7
    finally:
        chronos.batch_size = 256


def test_a_context_longer_than_the_model_reads_is_refused(chronos):
    with pytest.raises(ValueError, match="longer"):
        chronos.predict(np.zeros((1, 1, chronos.max_context + 1), np.float32), 1)


def test_the_real_model_copies_a_held_sensor_like_lowe_et_al_report(chronos):
    """Two noisy waves; from row 550 to 950 sensor 0 is held at 1.0 (an
    in-range value, like SWaT attack 27 in Lowe et al.'s Figure 1). The error
    spikes at the onset; once the 128-row context is filled with the held
    value, the model forecasts it perfectly and the attack scores LOWER than
    normal operation, which is why their no-fallback ROC-AUC is below 0.5."""
    rng = np.random.default_rng(3)
    t = np.arange(1100)
    x = np.column_stack([np.sin(t / 10), np.cos(t / 17)]) + 0.05 * rng.standard_normal((1100, 2))
    x = x.astype(np.float32)
    onset, end = 550, 950   # sin(55) is about -1, so the jump to 1.0 is large
    x[onset:end, 0] = 1.0
    rows = np.arange(128, 1100)
    scores = ForecastDetector(chronos, context=128, horizon=1, verbose=False).score_series(x, rows)
    normal = np.nanmean(scores[128:onset])
    assert scores[onset] > 5 * normal                           # onset: obvious
    assert np.nanmean(scores[onset + 128:end]) < normal         # held long enough: quieter than normal
