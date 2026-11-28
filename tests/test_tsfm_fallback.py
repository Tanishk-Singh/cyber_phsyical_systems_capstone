"""Tests for Forecast Fallback (Step B5) with fake forecasters: no model
download, CPU, a few seconds."""
import time

import numpy as np
import pytest

from ics_tsfm.models.tsfm import (ForecastDetector, ForecastFallbackDetector, LastValueForecaster,
                                  TimeLimitReached, fallback_thresholds, origins, spread_blocks)


class ZeroForecaster:
    """Always predicts 0: its forecasts do not depend on the input, so which
    rows go into fallback follows from x alone and can be written down."""

    def __init__(self):
        self.contexts = []

    def predict(self, contexts, horizon):
        self.contexts += list(contexts)
        return np.zeros(contexts.shape[:2] + (horizon,), np.float32)

    def config(self):
        return {"forecaster": "zero"}


class MeanForecaster:
    """Predicts each variate as the mean of its context: a slow adapter. A
    step held for longer than the context is absorbed (the copy failure),
    but not within the first few rows, so the fallback trigger can fire."""

    def __init__(self):
        self.calls = 0

    def predict(self, contexts, horizon):
        self.calls += 1
        # contiguous rows: every row is summed the same way whatever the batch, so
        # runs that batch origins differently still agree to the last bit
        return np.repeat(np.ascontiguousarray(contexts).mean(axis=-1, keepdims=True), horizon, axis=-1)

    def config(self):
        return {"forecaster": "mean"}


def held_attack(n=1000, start=500, end=900, seed=0):
    """Two quiet sensors; sensor 0 is held 3 units higher on rows [start, end)."""
    rng = np.random.default_rng(seed)
    x = (0.01 * rng.normal(size=(n, 2))).astype(np.float32)
    x[start:end, 0] += 3.0
    return x


# --- the rule ------------------------------------------------------------------------


def test_infinite_thresholds_give_the_plain_forecasts():
    x = np.random.default_rng(1).normal(size=(400, 3)).astype(np.float32)
    rows = np.r_[20:180, 230:400]   # two runs
    for horizon in (1, 4):
        for joint in (True, False):
            plain = ForecastDetector(MeanForecaster(), 20, horizon, joint=joint, verbose=False).predict(x, rows)
            ff = ForecastFallbackDetector(MeanForecaster(), 20, np.full(3, np.inf), horizon, joint=joint,
                                          verbose=False)
            forecasts, fallback = ff.run(x, rows)
            assert np.array_equal(forecasts, plain) and not fallback.any()


@pytest.mark.parametrize("horizon", [1, 4])
def test_fallback_starts_at_the_third_exceeding_row_in_a_row_and_ends_at_the_first_quiet_one(horizon):
    x = np.zeros((60, 2), np.float32)
    big = [20, 21, 23, 24, 25, 26, 28]   # sensor 0 far from the forecast (0) on these rows
    x[big, 0] = 5.0
    rows = np.arange(10, 60)
    det = ForecastFallbackDetector(ZeroForecaster(), 10, np.ones(2), horizon, verbose=False)
    _, fallback = det.run(x, rows)
    # 20, 21: only two in a row; 23-26: the third (25) and fourth (26) are replaced; 28: alone
    assert rows[fallback[:, 0]].tolist() == [25, 26]
    assert not fallback[:, 1].any()


def test_the_count_starts_again_in_every_run():
    x = np.zeros((80, 1), np.float32)
    x[[38, 39, 50, 51, 52], 0] = 5.0
    rows = np.r_[10:40, 50:80]   # run 1 ends with two exceeding rows, run 2 starts with three
    _, fallback = ForecastFallbackDetector(ZeroForecaster(), 10, np.ones(1), verbose=False).run(x, rows)
    assert rows[fallback[:, 0]].tolist() == [52]


def test_replaced_values_are_what_later_contexts_read():
    x = np.random.default_rng(2).normal(size=(70, 2)).astype(np.float32)
    x[30:40, 1] = 9.0
    rows = np.arange(10, 70)
    spy = ZeroForecaster()
    _, fallback = ForecastFallbackDetector(spy, 10, np.full(2, 5.0), verbose=False).run(x, rows)
    assert rows[fallback[:, 1]].tolist() == list(range(32, 40)) and not fallback[:, 0].any()
    expected_input = x.copy()
    expected_input[32:40, 1] = 0.0   # the forecast replaced the observation
    for t, context in zip(rows, spy.contexts):   # horizon 1, one run: one origin per call, in order
        assert np.array_equal(context, expected_input[t - 10:t].T)


def test_scores_are_against_the_real_values():
    x = held_attack()
    rows = np.arange(50, 1000)
    det = ForecastFallbackDetector(MeanForecaster(), 50, np.full(2, 0.1), verbose=False)
    forecasts, _ = det.run(x, rows)
    assert np.allclose(det.score(x, rows), np.abs(forecasts - x[rows]).mean(axis=1))
    assert np.allclose(det.errors(x, rows), np.abs(forecasts - x[rows]))


def test_fallback_keeps_a_held_attack_visible_where_the_plain_forecaster_loses_it():
    """Lowe et al.'s Fig. 3 in miniature. Thresholds come from benign rows,
    as in the run script."""
    x = held_attack()
    benign = np.arange(50, 450)
    thresholds = fallback_thresholds(ForecastDetector(MeanForecaster(), 50, verbose=False).errors(x, benign))
    rows = np.arange(50, 1000)
    plain = ForecastDetector(MeanForecaster(), 50, verbose=False).score_series(x, rows)
    ff_det = ForecastFallbackDetector(MeanForecaster(), 50, thresholds, verbose=False)
    ff = ff_det.score_series(x, rows)
    normal = np.nanmax(plain[:500])
    assert np.nanmax(plain[560:900]) < 2 * normal      # plain: the attack has filled the context
    assert np.nanmin(ff[502:900]) > 20 * normal        # fallback: visible for the whole attack
    _, fallback = ff_det.run(x, rows)
    assert fallback[(rows >= 502) & (rows < 900), 0].all()


# --- lockstep, sessions, cache ---------------------------------------------------------


@pytest.mark.parametrize("horizon", [1, 3])
def test_runs_in_lockstep_give_the_same_result_as_each_run_on_its_own(horizon):
    x = np.concatenate([held_attack(600, 300, 450, seed=3), held_attack(700, 200, 500, seed=4)])
    rows = np.r_[50:600, 700:1300]   # two runs, each with an attack
    thresholds = np.full(2, 0.05)
    det = ForecastFallbackDetector(MeanForecaster(), 50, thresholds, horizon, verbose=False)
    both = det.run(x, rows)
    first, second = det.run(x, rows[rows < 600]), det.run(x, rows[rows >= 700])
    assert both[1].any()
    assert np.array_equal(both[0], np.concatenate([first[0], second[0]]))
    assert np.array_equal(both[1], np.concatenate([first[1], second[1]]))


def test_the_runs_share_forward_passes():
    x = np.zeros((300, 1), np.float32)
    rows = np.r_[20:100, 150:230]   # two runs of 80 rows
    counter = MeanForecaster()
    ForecastFallbackDetector(counter, 20, np.ones(1), verbose=False).run(x, rows)
    assert counter.calls == 80


def test_a_run_whose_context_reaches_an_earlier_run_is_refused():
    det = ForecastFallbackDetector(ZeroForecaster(), 10, np.ones(1), verbose=False)
    with pytest.raises(ValueError, match="reaches back"):
        det.run(np.zeros((60, 1)), np.r_[10:20, 25:40])   # row 25 reads rows 15-24
    det.run(np.zeros((60, 1)), np.r_[10:20, 30:40])        # row 30 reads rows 20-29: fine


def test_a_time_limit_stops_between_steps_and_the_run_continues_to_the_same_result(tmp_path):
    x = held_attack()
    rows = np.arange(50, 1000)
    thresholds = np.full(2, 0.1)
    full = ForecastFallbackDetector(MeanForecaster(), 50, thresholds, verbose=False).run(x, rows)

    class DeadlineAfter(MeanForecaster):
        def predict(self, contexts, horizon):
            if self.calls == 599:   # the limit passes during step 600, inside the attack
                det.deadline = time.time() - 1
            return super().predict(contexts, horizon)

    det = ForecastFallbackDetector(DeadlineAfter(), 50, thresholds, verbose=False)
    with pytest.raises(TimeLimitReached, match="600 of 950"):
        det.run(x, rows, cache_dir=tmp_path)
    assert full[1][599].any()   # fallback is active where the first session stopped

    counter = MeanForecaster()   # next session, no limit
    again = ForecastFallbackDetector(counter, 50, thresholds, verbose=False).run(x, rows, cache_dir=tmp_path)
    assert counter.calls == 350
    assert np.array_equal(full[0], again[0]) and np.array_equal(full[1], again[1])
    # a finished run loads from its saved state without forecasting anything
    idle = MeanForecaster()
    loaded = ForecastFallbackDetector(idle, 50, thresholds, verbose=False).run(x, rows, cache_dir=tmp_path)
    assert idle.calls == 0 and np.array_equal(full[0], loaded[0])


def test_a_crash_loses_only_the_steps_since_the_last_checkpoint(tmp_path):
    x = held_attack()
    rows = np.arange(50, 1000)
    thresholds = np.full(2, 0.1)
    full = ForecastFallbackDetector(MeanForecaster(), 50, thresholds, verbose=False).run(x, rows)

    class Crash(Exception):
        pass

    class CrashAt(MeanForecaster):
        def predict(self, contexts, horizon):
            if self.calls == 700:
                raise Crash   # like a Ctrl+C or a power cut during step 701
            return super().predict(contexts, horizon)

    with pytest.raises(Crash):   # checkpoint after every step
        ForecastFallbackDetector(CrashAt(), 50, thresholds, checkpoint_minutes=0,
                                 verbose=False).run(x, rows, cache_dir=tmp_path)
    counter = MeanForecaster()
    again = ForecastFallbackDetector(counter, 50, thresholds, verbose=False).run(x, rows, cache_dir=tmp_path)
    assert counter.calls == 250
    assert np.array_equal(full[0], again[0]) and np.array_equal(full[1], again[1])


def test_a_cache_folder_refuses_other_thresholds_trigger_or_data(tmp_path):
    x = held_attack()
    rows = np.arange(50, 300)
    ForecastFallbackDetector(MeanForecaster(), 50, np.full(2, 0.1), verbose=False).run(x, rows, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="fallback_thresholds_sha1"):
        ForecastFallbackDetector(MeanForecaster(), 50, np.full(2, 0.2), verbose=False).run(
            x, rows, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="fallback_trigger"):
        ForecastFallbackDetector(MeanForecaster(), 50, np.full(2, 0.1), trigger=2, verbose=False).run(
            x, rows, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="x_sha1"):
        ForecastFallbackDetector(MeanForecaster(), 50, np.full(2, 0.1), verbose=False).run(
            x + 1, rows, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="fallback"):   # a plain run cannot use a fallback folder
        ForecastDetector(MeanForecaster(), 50, verbose=False).predict(x, rows, cache_dir=tmp_path)


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="thresholds"):
        ForecastFallbackDetector(ZeroForecaster(), 10, np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="trigger"):
        ForecastFallbackDetector(ZeroForecaster(), 10, np.ones(2), trigger=0)
    det = ForecastFallbackDetector(ZeroForecaster(), 10, np.ones(3), verbose=False)
    with pytest.raises(ValueError, match="3 thresholds for 2 sensors"):
        det.run(np.zeros((50, 2)), np.arange(10, 50))
    with pytest.raises(ValueError, match="fewer than context"):
        det.run(np.zeros((50, 3)), np.arange(5, 50))


# --- thresholds and the training-row sample -------------------------------------------


def test_thresholds_are_per_sensor_percentiles_with_a_floor():
    errors = np.c_[np.arange(1000.0), np.zeros(1000)]
    thresholds = fallback_thresholds(errors)
    assert thresholds[0] == pytest.approx(np.percentile(np.arange(1000.0), 99.5))
    assert thresholds[1] == 1e-6   # a sensor that never moved
    with pytest.raises(ValueError):
        fallback_thresholds(np.array([[1.0, np.nan]]))


@pytest.mark.parametrize("horizon", [1, 4])
def test_the_training_sample_is_whole_blocks_forecast_exactly_as_in_a_full_run(horizon):
    x = np.random.default_rng(5).normal(size=(3000, 3)).astype(np.float32)
    rows = np.r_[30:1500, 1530:3000]
    sample = spread_blocks(rows, horizon, 200)
    assert abs(len(sample) - 200) <= horizon and np.isin(sample, rows).all()
    assert sample[0] == rows[0] and sample[-1] > 2500   # spread over all rows
    full = ForecastDetector(MeanForecaster(), 30, horizon, verbose=False).predict(x, rows)
    part = ForecastDetector(MeanForecaster(), 30, horizon, verbose=False).predict(x, sample)
    assert np.array_equal(part, full[np.isin(rows, sample)])
    assert (origins(sample, horizon)[1] == origins(rows, horizon)[1][np.isin(rows, sample)]).all()
    assert np.array_equal(spread_blocks(rows, horizon, 10**6), rows)


def test_last_value_forecaster_never_triggers_on_a_held_step():
    """Why the trigger needs a forecaster that adapts gradually: one that
    copies the last value has caught up by the second attacked row, so three
    exceeding rows in a row never happen. Documents a limit of the rule."""
    x = held_attack()
    rows = np.arange(50, 1000)
    _, fallback = ForecastFallbackDetector(LastValueForecaster(), 50, np.full(2, 0.1), verbose=False).run(x, rows)
    assert not fallback.any()
