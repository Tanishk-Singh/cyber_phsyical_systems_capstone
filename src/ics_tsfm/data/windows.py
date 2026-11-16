"""Windowing (Phase A, Step 4): turning anchor rows into model samples.

A sample is named by its anchor row i. A windowed model reads the `history`
rows before it, [i - history, i), and is scored at row i; so the sample
touches rows [i - history, i]. Point-wise models (plain PCA) use history = 0:
the sample is row i alone.

Two jobs, shared by every model so they all see exactly the same samples:

1. Which rows can be scored. A sample may not cross a gap in the recording
   (the SWaT Attack file jumps 82 s between rows 298,836 and 298,837). Reading
   across it would make the plant look like it moved 82 s worth in one second.
   The rows right after a gap are treated like the first rows of the file:
   they wait until `history` clean rows have built up.

   All models are compared on the same rows: the ones the model with the
   LONGEST history can score. Each model saves a score for every row it can
   score; the metrics then use `scored_rows(timestamps, histories)`.

2. Cutting the windows, a batch at a time. Storing every window up front
   would take about 3.5 GB for SWaT at history 50, because neighbouring windows
   share all but one row; batches keep memory small.
"""
from collections.abc import Iterator

import numpy as np

ONE_SECOND = np.timedelta64(1, "s")


def contiguous_anchors(timestamps: np.ndarray, history: int,
                       step: np.timedelta64 = ONE_SECOND) -> np.ndarray:
    """True for every row i whose rows [i - history, i] contain no gap.

    Same counting trick as split_anchors: number the stretches between gaps,
    and a sample is clean when its first and last row are in the same stretch.

    Worked example, history = 2, a gap between rows 4 and 5:
        row        0  1  2  3  4 | 5  6  7  8
        stretch    0  0  0  0  0 | 1  1  1  1
        clean      F  F  T  T  T   F  F  T  T
    Rows 0-1 lack a full history (start of file); rows 5-6 lack one after the
    gap (row 5's sample would read rows 3-5, across the gap).
    """
    timestamps = np.asarray(timestamps)
    is_gap = np.r_[False, np.diff(timestamps) != step]
    stretch = np.cumsum(is_gap)
    clean = np.zeros(len(timestamps), bool)
    rows = np.arange(history, len(timestamps))
    clean[rows] = stretch[rows] == stretch[rows - history]
    return clean


def scored_rows(timestamps: np.ndarray, histories) -> np.ndarray:
    """Row numbers every model in the comparison can score.

    A clean sample at the longest history is clean at every shorter one, so
    this is simply the clean rows at max(histories). Example: comparing PCA
    (history 0) with the CNN (history 50) on the SWaT Attack file skips the
    first 50 rows of the file and the first 50 rows after the gap.
    """
    return np.flatnonzero(contiguous_anchors(timestamps, max(histories)))


def window_batches(x: np.ndarray, anchors: np.ndarray, history: int,
                   batch_size: int = 4096) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (windows, targets, anchors) in batches, in the order of `anchors`.

    windows: (batch, history, n_columns), rows [i - history, i) for each anchor i
    targets: (batch, n_columns), row i
    The caller is responsible for passing only valid anchors (from
    SwatSplit.anchors or scored_rows); this function does not re-check them.

    Example with one column x = [10, 11, 12, 13, 14], history 3, anchors [3, 4]:
        windows [[10, 11, 12], [11, 12, 13]], targets [13, 14]
    """
    offsets = np.arange(-history, 0)
    for start in range(0, len(anchors), batch_size):
        batch = np.asarray(anchors[start:start + batch_size])
        yield x[batch[:, None] + offsets], x[batch], batch
