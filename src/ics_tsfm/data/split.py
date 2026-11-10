"""Leak-free train/validation split of benign data (Phase A, Step 3).

Dataset-agnostic: works on any benign time series sampled at a fixed rate.
The SWaT-specific choices (block size, startup cutoff, columns) live in
swat_preprocess.py.

Two ideas
---------
1. Blocks. The benign rows are cut into contiguous blocks and whole blocks are
   assigned to training or validation at random. Random, so validation covers
   all days of the file (the plant drifts over the days); whole blocks, so the
   two sets do not interleave second by second.

2. Anchors. A model sample is named by its anchor row i: the input window is
   rows [i - history, i) and the target is row i, so the sample touches the
   rows [i - history, i]. A sample is kept only if every row it touches is on
   the same side. At a seam where training meets validation that drops exactly
   `history` anchors, all at the start of the later block; seams between two
   blocks on the same side drop nothing.

Why not Fung et al.'s split: they shuffle anchors (not blocks) 80/20, so each
row sits in `history` windows that land on both sides, and on SWaT with
history = 50 every validation window shares rows with training.
"""
import numpy as np


def assign_blocks(n_rows: int, block_rows: int, val_share: float, seed: int) -> np.ndarray:
    """True for every row that belongs to a validation block.

    The last block may be shorter than `block_rows`. Exactly
    round(n_blocks * val_share) blocks go to validation, picked at random.

    Example: n_rows=10, block_rows=3 -> blocks [0-2] [3-5] [6-8] [9];
    with 1 validation block picked as block 1, the result is
    [F, F, F, T, T, T, F, F, F, F].
    """
    n_blocks = -(-n_rows // block_rows)          # ceiling division
    n_val = round(n_blocks * val_share)
    rng = np.random.default_rng(seed)
    is_val_block = np.zeros(n_blocks, bool)
    is_val_block[rng.choice(n_blocks, size=n_val, replace=False)] = True
    return np.repeat(is_val_block, block_rows)[:n_rows]


def split_anchors(is_val: np.ndarray, history: int) -> tuple[np.ndarray, np.ndarray]:
    """(train_anchors, val_anchors): the anchors whose rows all sit on one side.

    How the check works: count, for every row, how many side changes have
    happened up to and including it. A sample touching rows [i - history, i]
    crosses a seam exactly when that count differs between its first and last
    row.

    Worked example, history = 2:
        row       0  1  2  3  4  5  6  7  8
        is_val    F  F  F  T  T  T  F  F  F
        changes   0  0  0  1  1  1  2  2  2
    Anchors start at row 2 (rows 0 and 1 have no full history).
        anchor 2: rows 0-2, counts 0 and 0 -> train
        anchor 3: rows 1-3, counts 0 and 1 -> crosses the seam, dropped
        anchor 4: rows 2-4, counts 0 and 1 -> dropped
        anchor 5: rows 3-5, counts 1 and 1 -> validation
        anchors 6, 7 dropped (seam at row 6); anchor 8 -> train
    Result: train [2, 8], validation [5]. Two anchors (= history) dropped at
    each seam, both at the start of the later block.

    With history = 0 (point-wise models such as plain PCA) nothing is dropped.
    """
    is_val = np.asarray(is_val, bool)
    changed = np.r_[False, is_val[1:] != is_val[:-1]]
    n_changes = np.cumsum(changed)
    anchors = np.arange(history, len(is_val))
    one_side = n_changes[anchors] == n_changes[anchors - history]
    kept = anchors[one_side]
    return kept[~is_val[kept]], kept[is_val[kept]]
