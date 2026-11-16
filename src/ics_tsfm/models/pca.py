"""PCA and windowed-PCA anomaly detectors (Phase A, Step 5).

How PCA detects anomalies
-------------------------
Normal rows follow a few shared patterns (pump on -> flow up -> tank fills).
PCA finds them as directions, sorted by how much of the normal variation each
explains, and keeps the top k. To score a sample it projects the sample onto
those k directions, rebuilds it, and measures what is left over:

    score = sum of squared differences between the sample and its rebuild

A sample that breaks a normal relationship (high flow, tank not filling) has
movement in directions normal data almost never uses, so the leftover is big.

Windowed PCA does the same on a window of the last `window` rows flattened
into one long row (window x n_columns numbers), so the patterns also include
how readings change over time; plain PCA is window = 1.

Choosing k
----------
k is the smallest number of directions that explains `variance` (default 99%)
of the training variation. It cannot be tuned on validation error: validation
holds only normal rows, and more directions always rebuild normal rows better,
so that would always pick every direction, which rebuilds attacks perfectly
too. `n_components` overrides the rule, for the sensitivity table.

Why not sklearn's PCA
---------------------
Holding every training window at once would take about 5.5 GB at window 50.
Here the covariance matrix is added up batch by batch instead, and PCA is the
eigen-decomposition of that matrix; the tests check the result against
sklearn on small data.
"""
import numpy as np

from ics_tsfm.data.windows import window_batches


def _samples(x: np.ndarray, anchors: np.ndarray, window: int, batch_size: int):
    """Flattened samples: rows [i - window + 1, i] for each anchor i, as one
    row of window * n_columns numbers. Yields (flat_batch, anchor_batch)."""
    for windows, targets, batch in window_batches(x, anchors, window - 1, batch_size):
        sample = np.concatenate([windows, targets[:, None, :]], axis=1)
        yield sample.reshape(len(batch), -1), batch


class PCADetector:
    """Reconstruction-error detector. Usage:

        det = PCADetector(window=50).fit(split.normal, train_anchors)
        val_scores = det.score(split.normal, val_anchors)

    The score at anchor i covers the whole window ending at i, so an anomaly
    keeps raising scores for up to `window - 1` rows after it ends.
    """

    def __init__(self, window: int = 1, variance: float = 0.99, n_components: int | None = None):
        if window < 1:
            raise ValueError("window must be at least 1 (1 = plain PCA)")
        self.window = window
        self.variance = variance
        self.n_components = n_components

    @property
    def history(self) -> int:
        """Rows needed before the anchor row; use with SwatSplit.anchors."""
        return self.window - 1

    def fit(self, x: np.ndarray, anchors: np.ndarray, batch_size: int = 4096) -> "PCADetector":
        """Learn the mean and the top-k directions from the training samples.

        Covariance from running sums: cov = E[z z^T] - mean mean^T, where z is
        a flattened sample. The data is already standardized (mean near 0), so
        this one-pass formula loses no meaningful precision.
        """
        dim = self.window * x.shape[1]
        total = np.zeros(dim)
        outer = np.zeros((dim, dim))
        n = 0
        for flat, _ in _samples(x, anchors, self.window, batch_size):
            total += flat.sum(axis=0)
            outer += flat.T @ flat
            n += len(flat)
        self.mean_ = total / n
        cov = outer / n - np.outer(self.mean_, self.mean_)

        eigvals, eigvecs = np.linalg.eigh(cov)          # ascending order
        eigvals = np.clip(eigvals[::-1], 0, None)       # largest first; tiny negatives are rounding
        eigvecs = eigvecs[:, ::-1]
        self.explained_variance_ratio_ = eigvals / eigvals.sum()

        if self.n_components is not None:
            k = self.n_components
        else:
            # smallest k whose top-k directions reach the variance share
            k = int(np.searchsorted(np.cumsum(self.explained_variance_ratio_), self.variance) + 1)
        self.k_ = min(k, dim)
        self.components_ = eigvecs[:, :self.k_]         # (dim, k), one direction per column
        return self

    def score(self, x: np.ndarray, anchors: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Reconstruction error for each anchor, in the order given.

        For a centred sample c: rebuild = (c P) P^T, where P holds the k
        directions; score = |c - rebuild|^2.
        """
        scores = np.empty(len(anchors))
        done = 0
        for flat, batch in _samples(x, anchors, self.window, batch_size):
            centred = flat - self.mean_
            leftover = centred - (centred @ self.components_) @ self.components_.T
            scores[done:done + len(batch)] = (leftover ** 2).sum(axis=1)
            done += len(batch)
        return scores

    def score_series(self, x: np.ndarray, anchors: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """One score per row of `x`: the score at each anchor, NaN elsewhere.
        This is the form Steps 7-8 consume (rows a model cannot score stay NaN)."""
        series = np.full(len(x), np.nan)
        series[anchors] = self.score(x, anchors, batch_size)
        return series
