"""Forecast Fallback (Lowe et al., ICML 2026 FMSD workshop, Sec. 3.4) (Phase B, Step B5).

The problem it fixes
--------------------
A pretrained forecaster copies what it reads. Once an attacker's value has
filled the context, the forecast follows it and the error drops back to the
normal level, so a long attack looks normal (Step B4 reproduced this on SWaT:
ROC-AUC 0.20-0.29). Forecast Fallback stops feeding the model values it has
reason to distrust and feeds it its own forecasts instead.

The rule as the paper states it
-------------------------------
"we use a 99.5th-percentile per-feature threshold on the MAE for FF and
trigger the fallback process if three consecutive time-steps of a feature
exceed this threshold". Once triggered, the forecaster's prediction replaces
the observation "as a surrogate input for that time step", and forecasting
continues "on the reconstructed input sequence".

How it is implemented here (points the paper leaves open are marked *)
---------------------------------------------------------------------
For every scored row t, in time order, and every sensor j:

    error_tj     = |x_tj - forecast_tj|            always against the REAL value
    count_j      = count_j + 1 if error_tj > threshold_j else 0
    fallback_tj  = count_j >= trigger              trigger = 3
    input_tj     = forecast_tj if fallback_tj else x_tj

Later forecasts read `input`, never x. Consequences:
  - The first trigger - 1 exceeding rows still enter the input (*: the paper
    does not say they are replaced afterwards; here they are not).
  - A sensor leaves fallback at the first row whose error is back under its
    threshold (*: the paper gives no release rule; this is the per-step
    reading of "for that time step"). It then needs `trigger` exceeding rows
    in a row again to re-enter.
  - The anomaly score is unchanged: mean absolute error over the sensors
    against the real values. A sensor in fallback keeps a large error for as
    long as the attack holds it away from the forecast (Lowe et al. Fig. 3).
  - threshold_j (*: the paper does not say which benign data) is the 99.5th
    percentile of sensor j's plain (no fallback) forecast error on benign
    rows; see `fallback_thresholds`. The run script uses Normal-file
    TRAINING rows, so the detection threshold tau can still come from the
    validation rows, as for every other model.

Horizon H > 1: one forecast covers H rows. The rule runs row by row through
them, and the replaced values enter the context of the next origin. Rows of
the same block are already forecast, so replacing one cannot change the
others in its block.

Why forecasts are made one origin at a time
-------------------------------------------
Every forecast depends on which earlier rows were replaced, so the origins of
one stretch of consecutive rows (a "run") must be forecast in order. Separate
runs are independent: the fallback state starts fresh at the first row of
each run, whose context is all real data. So all runs advance together (in
"lockstep") and their next origins share one forward pass. The SWaT Attack
file has 2 runs (the 82 s gap splits it); the validation rows have about 20
(one per stretch of validation blocks). A run whose context would reach back
into an earlier run is refused, because it would read replaced values.

Long runs
---------
With `cache_dir` the whole state (forecasts so far, which inputs were
replaced, the counters, how many lockstep steps are done) is saved every
`checkpoint_minutes`, and also when `deadline` (a time.time() value) passes,
in which case TimeLimitReached is raised. The same call later loads the state
and continues; the result is the same as an uninterrupted run, because the
saved state is everything the remaining forecasts depend on. A hard stop
(Ctrl+C, shutdown) loses at most the work since the last checkpoint. A
manifest refuses a folder that belongs to a different run (other thresholds,
trigger, rows, data or model).

Repeatability
-------------
Chronos-2's forecasts move by about 1e-6 when the same origin shares a
forward pass with different other origins (GPU rounding). Without fallback
that is all. A sensor in fallback, though, reads its own forecasts, and the
model standardises every series by its context's own spread, so such a tiny
difference can grow: measured on SWaT (context 32, attack 28's onset), from
1e-7 to 5e-4 over 24 steps, on the fallback sensors only, with identical
fallback decisions; with one origin per pass the two runs were bit for bit
equal. 5e-4 is negligible next to scores of 0.1-1 during an attack, but it
means fallback runs repeat bit for bit only with the same batches: same
rows, same batch_size, same GPU. A resumed run uses the same batches.
"""
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from ics_tsfm.models.tsfm.forecast import ForecastDetector, Forecaster, TimeLimitReached, origins

FF_PERCENTILE = 99.5
FF_TRIGGER = 3
# Guards a column that never moves in the benign rows (seven on/off columns
# are constant in SWaT's training blocks, e.g. P402, P501 and UV401): its
# benign error is about 0, and a threshold of 0 would put it in fallback on
# any float rounding. 1e-6 in scaled units is far below any real change.
MIN_THRESHOLD = 1e-6
STATE_FILE = "state.npz"


def fallback_thresholds(errors: np.ndarray, percentile: float = FF_PERCENTILE,
                        floor: float = MIN_THRESHOLD) -> np.ndarray:
    """Per-sensor fallback thresholds from benign forecast errors.

    errors: (n_rows, n_sensors) absolute errors of the PLAIN forecaster (no
    fallback) on benign rows. Returns (n_sensors,): each sensor's
    `percentile`-th percentile, but at least `floor`.
    """
    errors = np.asarray(errors, float)
    if errors.ndim != 2 or len(errors) == 0:
        raise ValueError(f"errors must be (n_rows, n_sensors) with at least one row, got {errors.shape}")
    if not np.isfinite(errors).all():
        raise ValueError("errors contain NaN or inf")
    return np.maximum(np.percentile(errors, percentile, axis=0), floor)


def spread_blocks(rows: np.ndarray, horizon: int, n_rows: int) -> np.ndarray:
    """About `n_rows` of `rows`, made of whole forecast blocks whose origins
    are spread evenly over all of `rows`.

    Used to estimate the fallback thresholds from a sample of the training
    rows: whole blocks keep the same mix of leads (1 to H) as a full run, and
    the origins of the sample are origins of the full rows too, so each
    sampled row gets exactly the forecast it would get in a full run. All
    rows come back if there are no more than n_rows.
    """
    rows = np.asarray(rows, np.int64)
    origin_idx, lead = origins(rows, horizon)
    starts = np.flatnonzero(lead == 0)
    n_blocks = max(1, -(-n_rows // horizon))
    if len(rows) <= n_rows or n_blocks >= len(starts):
        return rows
    # spacing (len(starts) - 1) / (n_blocks - 1) >= 1, so the rounded picks are distinct
    keep = starts[np.linspace(0, len(starts) - 1, n_blocks).round().astype(np.int64)]
    return rows[np.isin(origin_idx, keep)]


def _save_atomic(path: Path, **arrays) -> None:
    """Write then rename: a killed run never leaves half a file."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


class ForecastFallbackDetector(ForecastDetector):
    """ForecastDetector with Forecast Fallback. Usage:

        thresholds = fallback_thresholds(plain_detector.errors(x_normal, train_rows))
        det = ForecastFallbackDetector(forecaster, context=128, thresholds=thresholds, horizon=32)
        forecasts, fallback = det.run(split.test, rows, cache_dir=out / "state_test")
        scores = np.abs(forecasts - split.test[rows]).mean(axis=1)

    predict / errors / score / score_series work as in ForecastDetector and
    return the fallback forecasts; `run` also returns which inputs were
    replaced. batch_size counts origins per forecaster call (all runs'
    current origins go into one call, split into batches of this size).
    """

    def __init__(self, forecaster: Forecaster, context: int, thresholds: np.ndarray, horizon: int = 1,
                 joint: bool = True, trigger: int = FF_TRIGGER, batch_size: int = 256,
                 checkpoint_minutes: float = 5.0, deadline: float | None = None, verbose: bool = True):
        super().__init__(forecaster, context, horizon, joint, batch_size, deadline=deadline, verbose=verbose)
        thresholds = np.asarray(thresholds, float)
        if thresholds.ndim != 1 or np.isnan(thresholds).any() or (thresholds < 0).any():
            raise ValueError("thresholds must be a 1-D array of non-negative numbers (one per sensor)")
        if trigger < 1:
            raise ValueError("trigger must be at least 1")
        self.thresholds, self.trigger = thresholds, trigger
        self.checkpoint_minutes = checkpoint_minutes

    def config(self) -> dict:
        return {**super().config(), "fallback_trigger": self.trigger,
                "fallback_thresholds_sha1": hashlib.sha1(np.ascontiguousarray(self.thresholds)).hexdigest()}

    def _manifest(self, x: np.ndarray, rows: np.ndarray) -> dict:
        manifest = super()._manifest(x, rows)
        del manifest["chunk_origins"]   # fallback runs save a state, not chunks
        return manifest

    def predict(self, x: np.ndarray, rows: np.ndarray, cache_dir=None) -> np.ndarray:
        """(len(rows), n_sensors) fallback forecast for each row, in the order given."""
        return self.run(x, rows, cache_dir)[0]

    def run(self, x: np.ndarray, rows: np.ndarray, cache_dir=None) -> tuple[np.ndarray, np.ndarray]:
        """(forecasts, fallback), both (len(rows), n_sensors): the forecast of
        each row, and True where that row's value was replaced by its
        forecast in the input that later forecasts read."""
        rows = np.asarray(rows, np.int64)
        n_rows, n_sensors = len(rows), x.shape[1]
        if len(self.thresholds) != n_sensors:
            raise ValueError(f"{len(self.thresholds)} thresholds for {n_sensors} sensors")
        if n_rows and rows[0] < self.context:
            raise ValueError(f"row {rows[0]} has fewer than context={self.context} rows before it")
        origin_idx, lead = origins(rows, self.horizon)   # also checks rows are sorted and unique

        # ---- runs of consecutive rows, and the lockstep schedule over them
        starts_run = np.r_[True, np.diff(rows) != 1] if n_rows else np.zeros(0, bool)
        run_of_row = np.cumsum(starts_run) - 1
        first, n_runs = np.flatnonzero(starts_run), int(starts_run.sum())
        last = np.r_[first[1:] - 1, n_rows - 1] if n_runs else first
        if n_runs > 1 and np.any(rows[first[1:]] - self.context <= rows[last[:-1]]):
            raise ValueError(f"a run's context (context={self.context}) reaches back into an earlier run, "
                             "so it would read that run's replaced values; separate the runs further")
        is_origin = lead == 0
        origin_pos = np.flatnonzero(is_origin)           # positions in `rows` of every origin
        origin_number = np.cumsum(is_origin) - 1         # global number of each row's origin
        run_of_origin = run_of_row[origin_pos]
        # lockstep step = the origin's number inside its own run (each run starts with an origin)
        step_of_origin = origin_number[origin_pos] - origin_number[first[run_of_origin]]
        n_steps = int(step_of_origin.max()) + 1 if n_rows else 0
        schedule = np.lexsort((run_of_origin, step_of_origin))   # by step, then by run
        step_bounds = np.searchsorted(step_of_origin[schedule], np.arange(n_steps + 1))
        block_len = np.bincount(origin_idx, minlength=n_rows)    # rows forecast from each origin
        step_of_row = step_of_origin[origin_number]

        # ---- state: fresh, or loaded from the last checkpoint
        forecasts = np.zeros((n_rows, n_sensors), np.float32)
        fallback = np.zeros((n_rows, n_sensors), bool)
        counts = np.zeros((n_runs, n_sensors), np.int64)
        done_steps = 0
        state_path = None
        if cache_dir is not None:
            state_path = self._open_cache(cache_dir, x, rows) / STATE_FILE
            if state_path.exists():
                with np.load(state_path) as saved:
                    forecasts, fallback, counts = saved["forecasts"], saved["fallback"], saved["counts"]
                    done_steps = int(saved["done_steps"])
                if forecasts.shape != (n_rows, n_sensors) or counts.shape != (n_runs, n_sensors):
                    raise ValueError(f"{state_path} does not fit these rows; delete the folder")
                if self.verbose:
                    print(f"    resuming at step {done_steps:,} of {n_steps:,}", flush=True)
        # the input the forecaster reads: real data, with the replaced values put back
        x_in = np.array(x, dtype=np.float32)
        done = step_of_row < done_steps
        x_in[rows[done]] = np.where(fallback[done], forecasts[done], x_in[rows[done]])

        def save(steps):
            _save_atomic(state_path, forecasts=forecasts, fallback=fallback, counts=counts,
                         done_steps=np.int64(steps))

        start = last_save = time.perf_counter()
        for step in range(done_steps, n_steps):
            if self.deadline is not None and time.time() >= self.deadline:
                if state_path is not None:
                    save(step)
                raise TimeLimitReached(f"time limit reached with {step:,} of {n_steps:,} steps done")
            batch = schedule[step_bounds[step]:step_bounds[step + 1]]
            positions = origin_pos[batch]
            predicted = self._forecast(x_in, rows[positions])   # (n_origins, n_sensors, horizon)
            for p, run, forecast in zip(positions, run_of_origin[batch], predicted):
                count = counts[run].copy()
                for k in range(block_len[p]):
                    i = p + k
                    guess = forecast[:, k]
                    count = np.where(np.abs(x[rows[i]] - guess) > self.thresholds, count + 1, 0)
                    replace = count >= self.trigger
                    x_in[rows[i], replace] = guess[replace]
                    forecasts[i], fallback[i] = guess, replace
                counts[run] = count
            now = time.perf_counter()
            if state_path is not None and now - last_save >= 60 * self.checkpoint_minutes:
                save(step + 1)
                last_save = now
                if self.verbose:
                    spent = now - start
                    left = spent / (step + 1 - done_steps) * (n_steps - step - 1)
                    print(f"    step {step + 1:,}/{n_steps:,} ({100 * (step + 1) / n_steps:.0f}%), "
                          f"{spent / 60:.1f} min so far, about {left / 60:.0f} min left", flush=True)
        if state_path is not None and done_steps < n_steps:
            save(n_steps)
        return forecasts, fallback
