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


# --- additions A (per-lead thresholds) and B (bounded fallback) -----------------------


class TrendForecaster:
    """Continues the straight line through the first and last context value:
    the forecaster that runs away when it reads only its own forecasts."""

    def predict(self, contexts, horizon):
        slope = (contexts[..., -1:] - contexts[..., :1]) / (contexts.shape[-1] - 1)
        return contexts[..., -1:] + slope * np.arange(1, horizon + 1)

    def config(self):
        return {"forecaster": "trend"}


def triangle(n=1500, period=100):
    """One sensor ramping up and down between 0 and 1, like a tank level."""
    t = np.arange(n)
    return (np.abs((t % period) / (period / 2) - 1)[:, None]).astype(np.float32)


def test_each_lead_is_compared_with_its_own_threshold():
    x = np.zeros((60, 1), np.float32)
    x[:, 0] = 5.0   # forecast is always 0, so every row's error is 5
    rows = np.arange(10, 60)
    shared = ForecastFallbackDetector(ZeroForecaster(), 10, np.ones(1), horizon=2, verbose=False)
    per_lead = ForecastFallbackDetector(ZeroForecaster(), 10, np.array([[1.0], [10.0]]), horizon=2, verbose=False)
    assert shared.run(x, rows)[1][2:].all()   # every row exceeds: fallback from the third row on
    assert not per_lead.run(x, rows)[1].any()   # lead-1 rows stay under 10: never 3 in a row


def test_per_lead_thresholds_are_percentiles_at_each_lead():
    errors = np.c_[np.r_[np.arange(100.0), 1000 + np.arange(100.0)]]
    leads = np.r_[np.zeros(100, int), np.ones(100, int)]
    thresholds = fallback_thresholds(errors, leads, horizon=2)
    assert thresholds.shape == (2, 1)
    assert thresholds[0, 0] == pytest.approx(np.percentile(np.arange(100.0), 99.5))
    assert thresholds[1, 0] == pytest.approx(np.percentile(1000 + np.arange(100.0), 99.5))
    with pytest.raises(ValueError, match="missing"):
        fallback_thresholds(errors, np.zeros(200, int), horizon=2)


def test_per_lead_thresholds_must_match_the_horizon():
    with pytest.raises(ValueError, match="3 leads for horizon 2"):
        ForecastFallbackDetector(ZeroForecaster(), 10, np.ones((3, 1)), horizon=2)
    with pytest.raises(ValueError, match="max_fallback"):
        ForecastFallbackDetector(ZeroForecaster(), 10, np.ones(1), max_fallback=0)


def test_a_bounded_fallback_releases_and_starts_counting_again():
    x = np.zeros((70, 1), np.float32)
    x[20:60, 0] = 5.0   # far from the forecast (0) for 40 rows
    rows = np.arange(10, 70)
    _, fallback = ForecastFallbackDetector(ZeroForecaster(), 10, np.ones(1), max_fallback=5,
                                           verbose=False).run(x, rows)
    # 3 rows to trigger, 5 rows in fallback, released, 3 rows to trigger again, ...
    expected = [r for start in (22, 30, 38, 46, 54) for r in range(start, start + 5)]
    assert rows[fallback[:, 0]].tolist() == expected


def test_a_release_puts_the_real_values_back_into_the_input():
    x = np.zeros((40, 1), np.float32)
    x[20:40, 0] = 5.0
    rows = np.arange(10, 40)
    spy = ZeroForecaster()
    ForecastFallbackDetector(spy, 10, np.ones(1), max_fallback=5, verbose=False).run(x, rows)
    context_of = dict(zip(rows, spy.contexts))
    assert context_of[27][0, -5:].tolist() == [0.0] * 5   # rows 22-26 still replaced by the forecast
    assert context_of[28][0, -6:].tolist() == [5.0] * 6   # released at 27: rows 22-27 real again


def test_a_bounded_fallback_stops_a_forecast_running_away():
    """Lowe's rule with a trend-following forecaster on a tank-like ramp: at
    the first turn the sensor enters fallback, the forecast keeps climbing,
    its error never drops back, and the forecast runs off. Bounded, it
    re-syncs after at most `max_fallback` rows."""
    x = triangle()
    rows = np.arange(20, 1500)
    thresholds = np.full(1, 0.01)
    literal, literal_fb = ForecastFallbackDetector(TrendForecaster(), 20, thresholds, verbose=False).run(x, rows)
    bounded, _ = ForecastFallbackDetector(TrendForecaster(), 20, thresholds, max_fallback=20,
                                          verbose=False).run(x, rows)
    assert literal_fb[-500:].all() and np.abs(literal).max() > 10     # stuck in fallback, running off
    assert np.abs(bounded).max() < 2                                   # never far outside the 0-1 range


@pytest.mark.parametrize("horizon", [1, 3])
def test_lockstep_matches_runs_on_their_own_with_bounded_fallback(horizon):
    x = np.concatenate([held_attack(600, 300, 450, seed=3), held_attack(700, 200, 500, seed=4)])
    rows = np.r_[50:600, 700:1300]
    det = ForecastFallbackDetector(MeanForecaster(), 50, np.full((horizon, 2), 0.05), horizon,
                                   max_fallback=40, verbose=False)
    both = det.run(x, rows)
    first, second = det.run(x, rows[rows < 600]), det.run(x, rows[rows >= 700])
    assert np.array_equal(both[0], np.concatenate([first[0], second[0]]))
    assert np.array_equal(both[1], np.concatenate([first[1], second[1]]))


def test_a_bounded_run_resumes_to_the_same_result(tmp_path):
    x = held_attack()
    rows = np.arange(50, 1000)
    kwargs = dict(thresholds=np.full(2, 0.1), max_fallback=30, verbose=False)
    full = ForecastFallbackDetector(MeanForecaster(), 50, **kwargs).run(x, rows)
    attack = (rows >= 502) & (rows < 900)
    assert full[1][attack, 0].any() and not full[1][attack, 0].all()   # releases do happen

    class DeadlineAfter(MeanForecaster):
        def predict(self, contexts, horizon):
            if self.calls == 520:   # inside the attack, between releases
                det.deadline = time.time() - 1
            return super().predict(contexts, horizon)

    det = ForecastFallbackDetector(DeadlineAfter(), 50, **kwargs)
    with pytest.raises(TimeLimitReached):
        det.run(x, rows, cache_dir=tmp_path)
    again = ForecastFallbackDetector(MeanForecaster(), 50, **kwargs).run(x, rows, cache_dir=tmp_path)
    assert np.array_equal(full[0], again[0]) and np.array_equal(full[1], again[1])
    with pytest.raises(ValueError, match="fallback_max_rows"):
        ForecastFallbackDetector(MeanForecaster(), 50, np.full(2, 0.1), max_fallback=31,
                                 verbose=False).run(x, rows, cache_dir=tmp_path)


# --- addition C (clamped surrogates) ---------------------------------------------------


class HundredForecaster(ZeroForecaster):
    """Always predicts 100, far outside the data's range."""

    def predict(self, contexts, horizon):
        return super().predict(contexts, horizon) + 100.0


def test_clamped_surrogates_stay_inside_the_range_but_forecasts_do_not():
    x = np.zeros((40, 1), np.float32)
    rows = np.arange(10, 40)
    spy = HundredForecaster()
    det = ForecastFallbackDetector(spy, 10, np.ones(1), clamp=(np.array([-1.0]), np.array([1.0])), verbose=False)
    forecasts, fallback = det.run(x, rows)
    assert rows[fallback[:, 0]][0] == 12 and fallback[2:, 0].all()   # 3 rows in a row, then stays
    assert (forecasts == 100).all()                                   # scores use the real forecast
    context_of = dict(zip(rows, spy.contexts))
    assert context_of[20][0, -8:].tolist() == [1.0] * 8               # rows 12-19 stand in as 1, not 100


def test_clamping_stops_the_runaway_and_keeps_the_frozen_reference():
    x = triangle()
    rows = np.arange(20, 1500)
    clamped, clamped_fb = ForecastFallbackDetector(TrendForecaster(), 20, np.full(1, 0.01),
                                                   clamp=(np.zeros(1), np.ones(1)), verbose=False).run(x, rows)
    assert np.abs(clamped).max() < 2          # bounded by the range plus at most one step of trend
    # no release by time: the frozen value sits at the bottom edge of the range, and the sensor
    # leaves fallback only where the real ramp comes back down to it
    released = ~clamped_fb[-500:, 0]
    assert clamped_fb[-500:, 0].mean() > 0.9 and (x[rows[-500:]][released, 0] < 0.35).all()


def test_a_clamped_run_resumes_to_the_same_result(tmp_path):
    x = held_attack()
    rows = np.arange(50, 1000)
    clamp = (np.full(2, -0.05), np.full(2, 0.05))   # binds: surrogates during the attack are about 0.12
    kwargs = dict(thresholds=np.full(2, 0.1), clamp=clamp, verbose=False)
    full = ForecastFallbackDetector(MeanForecaster(), 50, **kwargs).run(x, rows)
    unclamped = ForecastFallbackDetector(MeanForecaster(), 50, np.full(2, 0.1), verbose=False).run(x, rows)
    assert not np.array_equal(full[0], unclamped[0])   # the clamp changed later forecasts

    class DeadlineAfter(MeanForecaster):
        def predict(self, contexts, horizon):
            if self.calls == 600:
                det.deadline = time.time() - 1
            return super().predict(contexts, horizon)

    det = ForecastFallbackDetector(DeadlineAfter(), 50, **kwargs)
    with pytest.raises(TimeLimitReached):
        det.run(x, rows, cache_dir=tmp_path)
    again = ForecastFallbackDetector(MeanForecaster(), 50, **kwargs).run(x, rows, cache_dir=tmp_path)
    assert np.array_equal(full[0], again[0]) and np.array_equal(full[1], again[1])
    with pytest.raises(ValueError, match="fallback_clamp_sha1"):
        ForecastFallbackDetector(MeanForecaster(), 50, np.full(2, 0.1), clamp=(clamp[0], clamp[1] * 2),
                                 verbose=False).run(x, rows, cache_dir=tmp_path)


def test_bad_clamps_are_refused():
    for clamp in ((np.ones(2), np.zeros(2)), (np.zeros(3), np.ones(3)), (np.zeros(2), np.array([1.0, np.inf]))):
        with pytest.raises(ValueError, match="clamp"):
            ForecastFallbackDetector(ZeroForecaster(), 10, np.ones(2), clamp=clamp)
