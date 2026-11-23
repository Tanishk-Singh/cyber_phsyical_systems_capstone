"""Seam leak check: do validation samples near training data score unusually low?

Why this exists
---------------
After training, the CNN's median error on training samples equals its median
error on validation samples (ratio 1.00). Equal errors can mean good
generalisation, or a leak: validation samples that are near-copies of
training ones across the seams of the 75-minute block split (Step 3). If near-
copies leaked, validation samples close in time to training data would be
predicted unusually well. This script tests exactly that, for every model.

Two tests
---------
  1. Median validation error by distance (in seconds) to the nearest training
     row, relative to the median over all validation samples (1.00 = same).
     A leak shows as values clearly below 1.00 at short distances.
  2. Inside each validation block, compare samples within 10 minutes of
     training data ("near") with the rest of the same block ("far"). This
     controls for block-to-block differences. A leak shows as near < far in
     most blocks; chance gives about half.

Result (2026-11-21): no leak. Near-seam samples are not predicted better
(ratios 0.96-1.14 at short distances); near < far in 8-12 of 21 blocks; median
near/far ratio 0.96-1.05. So the Step 3 seam buffer ("embargo") is not needed.

Reads saved validation scores only (run experiments/pca/run_swat_pca.py and
experiments/cnn/run_swat_cnn.py first). Runs in seconds:

    python experiments/analysis/seam_leak_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ics_tsfm.data.split import assign_blocks
from ics_tsfm.data.swat_preprocess import BLOCK_ROWS, STARTUP_CUTOFF_ROWS, VAL_SHARE

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
N_NORMAL_ROWS = 495_000 - STARTUP_CUTOFF_ROWS
SPLIT_SEED = 0
MODELS = {
    "Plain PCA": RESULTS / "pca" / "plain",
    "Windowed PCA": RESULTS / "pca" / "windowed_w50",
    "CNN seed 0": RESULTS / "cnn" / "seed0",
    "CNN seed 1": RESULTS / "cnn" / "seed1",
    "CNN seed 2": RESULTS / "cnn" / "seed2",
}
DISTANCE_BINS = [(0, 60), (60, 200), (200, 400), (400, 900), (900, 1800), (1800, 4500)]
NEAR_SECONDS = 600


def distance_to_training(is_val: np.ndarray) -> np.ndarray:
    """Seconds from every row to the nearest training row (0 for training rows)."""
    train_rows = np.flatnonzero(~is_val)
    idx = np.arange(len(is_val))
    k = np.searchsorted(train_rows, idx)
    before = np.where(k > 0, idx - train_rows[np.clip(k - 1, 0, None)], np.iinfo(np.int64).max)
    after = np.where(k < len(train_rows), train_rows[np.clip(k, 0, len(train_rows) - 1)] - idx,
                     np.iinfo(np.int64).max)
    return np.minimum(before, after)


def label(lo, hi):
    return f"{lo}-{hi} s" if hi <= 200 else f"{lo // 60}-{hi // 60} min"


def main():
    is_val = assign_blocks(N_NORMAL_ROWS, BLOCK_ROWS, VAL_SHARE, SPLIT_SEED)
    dist = distance_to_training(is_val)
    block = np.arange(N_NORMAL_ROWS) // BLOCK_ROWS

    by_distance, by_block = {}, []
    for name, folder in MODELS.items():
        val = np.load(folder / "val_scores.npy")
        scored = ~np.isnan(val)
        overall = np.median(val[scored])
        by_distance[name] = {label(lo, hi): np.median(val[scored & (dist >= lo) & (dist < hi)]) / overall
                             for lo, hi in DISTANCE_BINS}
        ratios = []
        for b in np.unique(block[scored]):
            inside = scored & (block == b)
            near, far = inside & (dist < NEAR_SECONDS), inside & (dist >= NEAR_SECONDS)
            if near.sum() > 60 and far.sum() > 60:
                ratios.append(np.median(val[near]) / np.median(val[far]))
        ratios = np.array(ratios)
        by_block.append({"model": name, "blocks compared": len(ratios),
                         "blocks where near < far": int((ratios < 1).sum()),
                         "median near/far ratio": float(np.median(ratios))})

    pd.set_option("display.width", 160)
    print("Test 1: median validation error by distance to the nearest training row "
          "(relative to all validation samples; a leak would show values clearly below 1.00 at the top)")
    print(pd.DataFrame(by_distance).round(2).to_string())
    print(f"\nTest 2: inside each validation block, near (< {NEAR_SECONDS // 60} min from training data) vs far "
          "(a leak would show near < far in most blocks)")
    print(pd.DataFrame(by_block).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
