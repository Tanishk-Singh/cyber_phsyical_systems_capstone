"""Forecast-error anomaly detection with a pretrained forecaster (Phase B, Step B2).

How it detects anomalies
------------------------
A forecaster reads the `context` rows before an origin row o, [o - context, o),
and predicts the next `horizon` rows, [o, o + horizon). Each predicted row is
compared with what actually happened; the score of row t is the mean absolute
error over the sensors, as in Lowe et al. (ICML 2026 FMSD workshop):

    score_t = sum_j |forecast_tj - x_tj| / n_sensors

(The CNN baseline uses the mean SQUARED error, following Fung et al.; the
TSFM follows Lowe et al. because their Forecast Fallback trigger is defined on
the per-sensor absolute error, which `errors` returns.)

Origins and horizon
-------------------
With horizon 1 every row is an origin: row t is predicted from the context
just before it. With horizon H > 1 the rows are covered in blocks: origins
every H rows, each forecast covering the next H rows, so every row gets
exactly one forecast and rows later in a block are predicted from older
information (lead 1 to H). This is the setup in Lowe et al.'s Figure 2.

Worked example, horizon 3, rows to score 10-16 and 20-23 (gap in between):
    row     10 11 12 13 14 15 16 | 20 21 22 23
    origin  10 10 10 13 13 13 16 | 20 20 20 23
    lead     0  1  2  0  1  2  0 |  0  1  2  0
A block never crosses a break in the rows: it restarts at row 20.

Which rows to pass
------------------
Pass the rows the model can score on its own: every row whose `context` rows
before it contain no recording gap, i.e.
`np.flatnonzero(contiguous_anchors(timestamps, context))`. Comparisons with
other models are then made on the shared rows by masking the saved scores
(windows.scored_rows), never by re-running with fewer rows, because with
horizon > 1 the rows passed decide where the blocks start.

Zero-shot models learn nothing from the plant's data, so the Phase A blocked
split's purge (SwatSplit.anchors) is not needed: a context may run across
training and validation rows of the Normal file. The split only decides
which scores go to which pile (e.g. thresholds from validation rows).

All sensors together or one at a time
-------------------------------------
`joint=True` hands the forecaster each origin as one multivariate series,
shape (n_sensors, context), so a multivariate model such as Chronos-2 can use
the other sensors. `joint=False` hands it every sensor as its own series,
shape (1, context), so no model can see across sensors. This is the
joint-vs-per-sensor ablation, and it works the same for any forecaster.

Long runs
---------
Origins are processed in chunks; with `cache_dir` each finished chunk is
saved, so an interrupted run picks up where it stopped. A manifest in the
folder records the settings and refuses to mix chunks from different runs.
With `deadline` (a time.time() value) no new chunk is started once it has
passed: predict raises TimeLimitReached with every finished chunk saved, and
the same call later continues from there. This is how runs are split into
sessions on a shared lab PC.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np


class TimeLimitReached(Exception):
    """The deadline passed between two chunks; the finished chunks are saved."""


class Forecaster(Protocol):
    """What ForecastDetector needs from a model (Chronos-2, Moirai, a fake)."""

    def predict(self, contexts: np.ndarray, horizon: int) -> np.ndarray:
        """contexts: (n_series, n_variates, context) float32, oldest value first.
        Returns point forecasts (n_series, n_variates, horizon)."""
        ...

    def config(self) -> dict:
        """Settings that change the forecasts (model id, revision, ...)."""
        ...


class LastValueForecaster:
    """Predicts that every sensor stays at its last observed value.

    The 'copy the input' behaviour in its purest form: no model can follow the
    plant more closely from one step to the next, and none is blinder to an
    attack that holds a sensor at a fixed value. Used in tests, and the same
    as Phase A's persistence check (experiments/analysis/persistence_check.py).
    """

    def predict(self, contexts: np.ndarray, horizon: int) -> np.ndarray:
        return np.repeat(contexts[..., -1:], horizon, axis=-1)

    def config(self) -> dict:
        return {"forecaster": "last_value"}


def origins(rows: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """For each row: the index (into `rows`) of the origin it is forecast from,
    and its lead (0 = the origin row itself). See the worked example above.

    Same counting trick as windows.contiguous_anchors: number the runs of
    consecutive rows, then count each row's position inside its run.
    """
    rows = np.asarray(rows)
    if len(rows) and np.any(np.diff(rows) <= 0):
        raise ValueError("rows must be sorted and unique")
    starts_run = np.r_[True, np.diff(rows) != 1]
    run_start = np.maximum.accumulate(np.where(starts_run, np.arange(len(rows)), 0))
    lead = (np.arange(len(rows)) - run_start) % horizon
    return np.arange(len(rows)) - lead, lead


class ForecastDetector:
    """Usage:

        det = ForecastDetector(forecaster, context=128, horizon=1)
        rows = np.flatnonzero(contiguous_anchors(split.test_timestamps, det.history))
        err = det.errors(split.test, rows, cache_dir=out / "test_cache")   # (n_rows, n_sensors)
        scores = err.mean(axis=1)
    """

    def __init__(self, forecaster: Forecaster, context: int, horizon: int = 1,
                 joint: bool = True, batch_size: int = 256, chunk_origins: int = 8192,
                 deadline: float | None = None, verbose: bool = True):
        if context < 1 or horizon < 1:
            raise ValueError("context and horizon must be at least 1")
        self.forecaster = forecaster
        self.context, self.horizon, self.joint = context, horizon, joint
        self.batch_size, self.chunk_origins = batch_size, chunk_origins
        self.deadline = deadline
        self.verbose = verbose

    @property
    def history(self) -> int:
        """Rows read before a scored row (same meaning as for PCA and the CNN)."""
        return self.context

    def config(self) -> dict:
        return {"context": self.context, "horizon": self.horizon, "joint": self.joint,
                **self.forecaster.config()}

    # ------------------------------------------------------------- forecasting
    def _forecast(self, x: np.ndarray, origin_rows: np.ndarray) -> np.ndarray:
        """(n_origins, n_sensors, horizon) forecasts, built a batch at a time."""
        offsets = np.arange(-self.context, 0)
        n_sensors = x.shape[1]
        out = np.empty((len(origin_rows), n_sensors, self.horizon), np.float32)
        for start in range(0, len(origin_rows), self.batch_size):
            batch = origin_rows[start:start + self.batch_size]
            contexts = x[batch[:, None] + offsets].transpose(0, 2, 1).astype(np.float32)
            if not self.joint:   # every sensor becomes its own one-variate series
                contexts = contexts.reshape(-1, 1, self.context)
            pred = np.asarray(self.forecaster.predict(contexts, self.horizon), np.float32)
            out[start:start + len(batch)] = pred.reshape(len(batch), n_sensors, self.horizon)
        return out

    def _manifest(self, x: np.ndarray, rows: np.ndarray) -> dict:
        return {**self.config(), "chunk_origins": self.chunk_origins, "n_rows": int(len(rows)),
                "rows_sha1": hashlib.sha1(np.ascontiguousarray(rows, np.int64)).hexdigest(),
                "x_shape": list(x.shape),
                "x_sha1": hashlib.sha1(np.ascontiguousarray(x, np.float32)).hexdigest()}

    def predict(self, x: np.ndarray, rows: np.ndarray, cache_dir=None) -> np.ndarray:
        """(len(rows), n_sensors) forecast for each row, in the order given."""
        rows = np.asarray(rows, np.int64)
        if len(rows) and rows[0] < self.context:
            raise ValueError(f"row {rows[0]} has fewer than context={self.context} rows before it")
        _, lead = origins(rows, self.horizon)
        is_origin = lead == 0
        origin_rows = rows[is_origin]
        # Row k's forecast sits in the forecast of origin number `which[k]`, at `lead[k]`.
        which = np.cumsum(is_origin) - 1

        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            manifest = self._manifest(x, rows)
            path = cache_dir / "manifest.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved != manifest:
                    changed = sorted(k for k in manifest if saved.get(k) != manifest[k])
                    raise ValueError(f"{cache_dir} holds a different run (differs in {changed}); "
                                     "delete it or pass another cache_dir")
            else:
                path.write_text(json.dumps(manifest, indent=2))

        chunks = []
        n_chunks = -(-len(origin_rows) // self.chunk_origins)
        start_time, computed = time.perf_counter(), 0
        for c in range(n_chunks):
            part = origin_rows[c * self.chunk_origins:(c + 1) * self.chunk_origins]
            file = cache_dir / f"chunk_{c:05d}.npy" if cache_dir is not None else None
            if file is not None and file.exists():
                chunks.append(np.load(file))
                if self.verbose:
                    print(f"    chunk {c + 1}/{n_chunks} already saved, skipped", flush=True)
                continue
            if self.deadline is not None and time.time() >= self.deadline:
                raise TimeLimitReached(f"time limit reached with {c} of {n_chunks} chunks saved")
            forecast = self._forecast(x, part)
            if file is not None:   # write then rename: a killed run never leaves half a file
                tmp = file.with_suffix(".tmp.npy")
                np.save(tmp, forecast)
                os.replace(tmp, file)
            chunks.append(forecast)
            computed += 1
            if self.verbose:
                spent = time.perf_counter() - start_time
                left = spent / computed * (n_chunks - c - 1)   # from the chunks computed in this run
                print(f"    chunk {c + 1}/{n_chunks} ({100 * (c + 1) / n_chunks:.0f}%), "
                      f"{spent / 60:.1f} min so far, about {left / 60:.0f} min left", flush=True)
        forecasts = np.concatenate(chunks) if chunks else np.empty((0, x.shape[1], self.horizon))
        return forecasts[which, :, lead]

    def errors(self, x: np.ndarray, rows: np.ndarray, cache_dir=None) -> np.ndarray:
        """(len(rows), n_sensors) absolute forecast error per row and sensor."""
        rows = np.asarray(rows, np.int64)
        return np.abs(self.predict(x, rows, cache_dir) - x[rows])

    def score(self, x: np.ndarray, rows: np.ndarray, cache_dir=None) -> np.ndarray:
        """Mean absolute error over the sensors, for each row in the order given."""
        return self.errors(x, rows, cache_dir).mean(axis=1)

    def score_series(self, x: np.ndarray, rows: np.ndarray, cache_dir=None) -> np.ndarray:
        """One score per row of `x`: the score at each row given, NaN elsewhere."""
        series = np.full(len(x), np.nan)
        series[rows] = self.score(x, rows, cache_dir)
        return series
