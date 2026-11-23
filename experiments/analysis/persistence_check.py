"""Does a drift-proof forecaster avoid the drift false alarms, and at what cost?

Why this exists
---------------
The CNN's false alarms come from sensors drifting beyond anything seen in
training. One proposed remedy is to make the forecaster look only at changes
relative to the recent past (per-window normalisation, as foundation models
do). The extreme version is the "persistence" forecaster: predict that row i
equals row i-1. It is drift-invariant by construction, so it shows the cost of
full drift-invariance.

Same placeholder thresholds as the CNN sanity check (99.5th / 99.9th
percentile of validation scores, every row judged on its own), on the shared
test rows.

Result (2026-11-17): no drift problem (false alarms 0.60% at the 99.5th
percentile, close to validation's 0.5%), but it catches almost no attacks
(0.5% of attack rows). SWaT's attacks are long and steady, so a model that
only reacts to change misses them: the same "collapse" Lowe et al. report for
foundation models. Normalising drift away trades false alarms for recall.

Loads the data (about 2 minutes):

    python experiments/analysis/persistence_check.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

from ics_tsfm.data.swat import load_swat_attack, load_swat_normal
from ics_tsfm.data.swat_preprocess import prepare_swat

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "codebase"))
from swat_cycle_analysis import SWAT_DIR  # noqa: E402


def persistence_error(x: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Squared error of predicting row i as row i-1, averaged over columns."""
    return ((x[anchors] - x[anchors - 1]) ** 2).mean(axis=1)


def main():
    warnings.simplefilter("ignore")
    split = prepare_swat(load_swat_normal(SWAT_DIR), load_swat_attack(SWAT_DIR))
    _, val = split.anchors(200)
    rows = split.test_rows([0, 49, 200])                  # the rows every model is scored on
    y = split.test_is_attack[rows]
    v, t = persistence_error(split.normal, val), persistence_error(split.test, rows)
    for q in (99.5, 99.9):
        tau = np.percentile(v, q)
        recall, fa = (t[y] > tau).mean(), (t[~y] > tau).mean()
        tp, fp = recall * y.sum(), fa * (~y).sum()
        precision = tp / (tp + fp)
        print(f"persistence, threshold p{q}: attack rows caught {recall * 100:.1f}%, false alarms "
              f"{fa * 100:.2f}% of normal test rows, point-F1 {2 * precision * recall / (precision + recall):.3f}")
    j = split.columns.index("AIT504")
    normal_test = rows[~y]
    print(f"AIT504: training max {split.normal[~split.is_val, j].max():.2f}, normal-test 99th percentile "
          f"{np.percentile(split.test[normal_test, j], 99):.2f} (level drifts), but one-second change "
          f"99.9th percentile: validation {np.percentile(np.abs(np.diff(split.normal[val, j])), 99.9):.3f} vs "
          f"normal test {np.percentile(np.abs(split.test[normal_test, j] - split.test[normal_test - 1, j]), 99.9):.3f} "
          f"(changes do not)")


if __name__ == "__main__":
    main()
