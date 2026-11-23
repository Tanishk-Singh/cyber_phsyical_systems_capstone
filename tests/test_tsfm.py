"""Tests for the model-independent TSFM plumbing (ForecastDetector) with fake
forecasters: no model download, CPU, under a second. The Chronos-2 adapter
gets its own tests in Step B3."""
import numpy as np
import pytest

from ics_tsfm.data.windows import contiguous_anchors
from ics_tsfm.models.tsfm import ForecastDetector, LastValueForecaster, origins


class OldestValueForecaster:
    """Predicts the OLDEST value in the context: catches off-by-one errors at
    the far end of the context, as LastValueForecaster does at the near end."""

    def predict(self, contexts, horizon):
        return np.repeat(contexts[..., :1], horizon, axis=-1)

    def config(self):
        return {"forecaster": "oldest_value"}


class MeanOfSensorsForecaster:
    """Predicts every variate as the mean of all variates' last values. It is
    the simplest model that looks ACROSS sensors, so joint and per-sensor runs
    must differ for it."""

    def predict(self, contexts, horizon):
        last = contexts[..., -1:].mean(axis=1, keepdims=True)
        return np.repeat(np.repeat(last, contexts.shape[1], axis=1), horizon, axis=-1)

    def config(self):
        return {"forecaster": "mean_of_sensors"}


class CountingForecaster(LastValueForecaster):
    """LastValueForecaster that counts how many series it was asked for."""

    def __init__(self):
        self.calls = 0

    def predict(self, contexts, horizon):
        self.calls += len(contexts)
        return super().predict(contexts, horizon)


def ramp(n=200, n_sensors=3):
    """Sensor j at row t = 100 * j + t: every value tells you its row."""
    return (np.arange(n)[:, None] + 100.0 * np.arange(n_sensors)).astype(np.float32)


# --- origins -------------------------------------------------------------------------


def test_origins_match_the_worked_example_in_the_docstring():
    rows = np.r_[10:17, 20:24]
    origin_idx, lead = origins(rows, 3)
    assert rows[origin_idx].tolist() == [10, 10, 10, 13, 13, 13, 16, 20, 20, 20, 23]
    assert lead.tolist() == [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0]


def test_with_horizon_1_every_row_is_its_own_origin():
    rows = np.r_[5:9, 12:15]
    origin_idx, lead = origins(rows, 1)
    assert (origin_idx == np.arange(len(rows))).all() and (lead == 0).all()


def test_origins_reject_unsorted_rows():
    with pytest.raises(ValueError):
        origins(np.array([3, 2, 4]), 1)


# --- forecasts land on the right rows ---------------------------------------------


def test_horizon_1_predicts_each_row_from_the_row_before():
    x = ramp()
    rows = np.arange(10, 200)
    pred = ForecastDetector(LastValueForecaster(), context=10, verbose=False).predict(x, rows)
    assert np.array_equal(pred, x[rows - 1])


def test_context_covers_exactly_the_rows_before_the_origin():
    x = ramp()
    rows = np.arange(10, 200)
    pred = ForecastDetector(OldestValueForecaster(), context=10, verbose=False).predict(x, rows)
    assert np.array_equal(pred, x[rows - 10])   # oldest context row of origin t is t - 10


def test_horizon_blocks_forecast_later_rows_from_older_information():
    x = ramp()
    rows = np.r_[10:17, 20:24]
    pred = ForecastDetector(LastValueForecaster(), context=5, horizon=3, verbose=False).predict(x, rows)
    origin_idx, _ = origins(rows, 3)
    # every row in a block gets the value of the row just before its origin
    assert np.array_equal(pred, x[rows[origin_idx] - 1])


def test_rows_without_a_full_context_are_refused():
    with pytest.raises(ValueError):
        ForecastDetector(LastValueForecaster(), context=10, verbose=False).predict(ramp(), np.arange(5, 50))


@pytest.mark.parametrize("horizon", [1, 4])
def test_batch_and_chunk_sizes_do_not_change_the_forecasts(horizon):
    x = np.random.default_rng(0).normal(size=(300, 4)).astype(np.float32)
    rows = np.arange(20, 300)
    ref = ForecastDetector(MeanOfSensorsForecaster(), 20, horizon, verbose=False).predict(x, rows)
    other = ForecastDetector(MeanOfSensorsForecaster(), 20, horizon, batch_size=7,
                             chunk_origins=11, verbose=False).predict(x, rows)
    assert np.array_equal(ref, other)


# --- joint vs one sensor at a time ----------------------------------------------------


def test_per_sensor_mode_hides_the_other_sensors():
    x = ramp()
    rows = np.arange(10, 200)
    joint = ForecastDetector(MeanOfSensorsForecaster(), 10, joint=True, verbose=False).predict(x, rows)
    alone = ForecastDetector(MeanOfSensorsForecaster(), 10, joint=False, verbose=False).predict(x, rows)
    # together: every sensor is predicted as the mean of the three last values
    assert np.allclose(joint, x[rows - 1].mean(axis=1, keepdims=True))
    # alone: the "mean of sensors" is just the sensor itself
    assert np.array_equal(alone, x[rows - 1])


def test_modes_agree_for_a_model_that_never_looks_across_sensors():
    x = ramp()
    rows = np.arange(10, 200)
    a = ForecastDetector(LastValueForecaster(), 10, horizon=4, joint=True, verbose=False).predict(x, rows)
    b = ForecastDetector(LastValueForecaster(), 10, horizon=4, joint=False, verbose=False).predict(x, rows)
    assert np.array_equal(a, b)


# --- scores ---------------------------------------------------------------------------


def test_score_is_the_mean_absolute_error_over_sensors():
    x = np.random.default_rng(1).normal(size=(100, 3)).astype(np.float32)
    rows = np.arange(5, 100)
    det = ForecastDetector(LastValueForecaster(), context=5, verbose=False)
    expected = np.abs(x[rows] - x[rows - 1]).mean(axis=1)
    assert np.allclose(det.score(x, rows), expected)
    series = det.score_series(x, rows)
    assert np.isnan(series[:5]).all() and np.allclose(series[rows], expected)


def test_a_copying_forecaster_loses_a_long_attack_after_its_first_step():
    """Lowe et al.'s failure in miniature: a sensor is pushed to a new value
    and held there. The error jumps at the first attacked row, then falls
    straight back to the normal level, because the forecaster copies the
    attacked values it now sees in its context."""
    rng = np.random.default_rng(2)
    x = (0.01 * rng.normal(size=(1000, 2))).astype(np.float32)
    x[500:900, 0] += 3.0   # attack: sensor 0 held 3 units higher for 400 rows
    rows = np.arange(50, 1000)
    scores = ForecastDetector(LastValueForecaster(), context=50, verbose=False).score_series(x, rows)
    normal = np.nanmax(scores[:500])
    assert scores[500] > 20 * normal           # onset: obvious (3 units / 2 sensors = 1.5, ~45x)
    assert np.nanmax(scores[502:900]) < 2 * normal   # during the attack: looks normal


def test_gapped_recording_rows_come_from_contiguous_anchors():
    t = np.arange(60).astype("datetime64[s]")
    t[30:] += 82   # an 82 s gap between rows 29 and 30, like the SWaT Attack file
    rows = np.flatnonzero(contiguous_anchors(t, 10))
    assert rows[0] == 10 and 30 not in rows and rows[rows > 29][0] == 40
    pred = ForecastDetector(LastValueForecaster(), 10, horizon=4, verbose=False).predict(ramp(60), rows)
    assert pred[rows.tolist().index(40)].tolist() == ramp(60)[39].tolist()   # block restarts after the gap


# --- resumable runs -------------------------------------------------------------------


def test_an_interrupted_run_recomputes_only_the_missing_chunk(tmp_path):
    x = np.random.default_rng(3).normal(size=(500, 3)).astype(np.float32)
    rows = np.arange(20, 500)
    full = ForecastDetector(LastValueForecaster(), 20, chunk_origins=100, verbose=False).predict(
        x, rows, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("chunk_*.npy"))) == 5
    (tmp_path / "chunk_00002.npy").unlink()   # pretend the run died here

    counter = CountingForecaster()
    again = ForecastDetector(counter, 20, chunk_origins=100, verbose=False).predict(x, rows, cache_dir=tmp_path)
    assert counter.calls == 100 and np.array_equal(full, again)


def test_a_cache_folder_refuses_a_different_run(tmp_path):
    x = np.random.default_rng(4).normal(size=(200, 2)).astype(np.float32)
    rows = np.arange(20, 200)
    ForecastDetector(LastValueForecaster(), 20, verbose=False).predict(x, rows, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="horizon"):
        ForecastDetector(LastValueForecaster(), 20, horizon=4, verbose=False).predict(x, rows, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="x_sha1"):
        ForecastDetector(LastValueForecaster(), 20, verbose=False).predict(x + 1, rows, cache_dir=tmp_path)
