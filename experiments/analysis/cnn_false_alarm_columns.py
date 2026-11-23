"""Which columns cause the CNN's false alarms on the Attack file's normal rows?

Why this exists
---------------
With a threshold at the 99.5th percentile of validation scores, the CNN
flagged about half of all normal test rows. This script breaks the CNN's
prediction error down by column to find out why.

What it prints, per seed
------------------------
  - mean squared error per column on validation vs on normal test rows, for
    the columns with the largest test error (and the ratio between them);
  - in each false-alarm row, which column has the largest error;
  - how many false alarms would remain if the top column were ignored;
  - for a few suspect columns: the scaled range seen in training vs the range
    of the normal test rows.

Result (2026-11-17): drift, not P301. AIT504 has about 550x its validation
error on normal test rows and is the largest error in 58-78% of false-alarm
rows; its normal test values reach about 6 spreads while training never went
above 2.7. Next come the drifting stage-4/5 flows (FIT504, FIT503, FIT401,
FIT501: about 60-100x) and AIT202 (about 50x). These are exactly the columns
Step 3's drift analysis flagged as drifting within the Normal file itself.

Reads results/cnn/seed{0,1,2}/model.pt (experiments/cnn/run_swat_cnn.py) and
loads the data (about 2 minutes):

    python experiments/analysis/cnn_false_alarm_columns.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from ics_tsfm.data.swat import load_swat_attack, load_swat_normal
from ics_tsfm.data.swat_preprocess import prepare_swat
from ics_tsfm.models.cnn import CNNDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "codebase"))
from swat_cycle_analysis import SWAT_DIR  # noqa: E402

SEEDS = (0, 1, 2)
THRESHOLD_PERCENTILE = 99.5
SUSPECTS = ("AIT504", "AIT202", "FIT504", "FIT601", "P301")


@torch.no_grad()
def per_column_errors(det: CNNDetector, x: np.ndarray, anchors: np.ndarray, batch_size: int = 8192):
    """Squared prediction error per anchor AND per column (not averaged)."""
    det.model_.eval()
    x_dev = det._to_device(x)
    a_dev = torch.as_tensor(anchors, device=det.device)
    out = []
    for start in range(0, len(a_dev), batch_size):
        windows, targets = det._gather(x_dev, a_dev[start:start + batch_size])
        out.append(((det.model_(windows) - targets) ** 2).cpu().numpy())
    return np.concatenate(out)


def main():
    warnings.simplefilter("ignore")
    split = prepare_swat(load_swat_normal(SWAT_DIR), load_swat_attack(SWAT_DIR))
    cols = split.columns
    _, val = split.anchors(200)
    rows = split.test_rows([200])
    normal_rows = rows[~split.test_is_attack[rows]]

    for seed in SEEDS:
        det = CNNDetector.load(REPO_ROOT / "results" / "cnn" / f"seed{seed}" / "model.pt", len(cols))
        ev = per_column_errors(det, split.normal, val)
        et = per_column_errors(det, split.test, normal_rows)
        tau = np.percentile(ev.mean(axis=1), THRESHOLD_PERCENTILE)
        fa = et.mean(axis=1) > tau
        print(f"--- CNN seed {seed}: threshold {tau:.3f} (validation p{THRESHOLD_PERCENTILE}); "
              f"false alarms on {fa.mean() * 100:.1f}% of normal test rows")
        print("    mean squared error per column, validation vs normal test rows (top 8 by test):")
        for j in np.argsort(-et.mean(axis=0))[:8]:
            v, t = ev[:, j].mean(), et[:, j].mean()
            print(f"      {cols[j]:8s} validation {v:8.4f}   test {t:9.4f}   (x{t / max(v, 1e-9):.0f})")
        top, counts = np.unique(np.argmax(et[fa], axis=1), return_counts=True)
        ranked = sorted(zip(top, counts), key=lambda z: -z[1])[:5]
        print("    column with the largest error in each false-alarm row:",
              ", ".join(f"{cols[k]} {n / fa.sum() * 100:.0f}%" for k, n in ranked))
        j = ranked[0][0]
        keep = [k for k in range(len(cols)) if k != j]
        tau_keep = np.percentile(ev[:, keep].mean(axis=1), THRESHOLD_PERCENTILE)
        print(f"    if {cols[j]} were ignored: false alarms on {(et[:, keep].mean(axis=1) > tau_keep).mean() * 100:.1f}% "
              f"of normal test rows")

    print("\nScaled range in training vs normal test rows (1st to 99th percentile):")
    for col in SUSPECTS:
        j = cols.index(col)
        train = split.normal[~split.is_val, j]
        test = split.test[normal_rows, j]
        print(f"  {col:7s} training {train.min():7.2f} .. {train.max():7.2f}   "
              f"normal test {np.percentile(test, 1):7.2f} .. {np.percentile(test, 99):7.2f}")


if __name__ == "__main__":
    main()
