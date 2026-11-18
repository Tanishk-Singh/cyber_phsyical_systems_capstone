"""Phase A Step 5 run: fit plain and windowed PCA on SWaT and save their scores.

Writes the same files as experiments/cnn/run_swat_cnn.py, so the detection
step (experiments/detection/) treats every model the same way
(results/ is git-ignored):

  results/pca/{plain,windowed_w50}/val_scores.npy    score per Normal-file row (NaN = not scored)
  results/pca/{plain,windowed_w50}/test_scores.npy   score per Attack-file row (NaN = not scored)
  results/pca/summary.json                           k, explained variance, timing

Run from the repo root (about 2 minutes to load the data, then about 1.5
minutes for both models):

    python experiments/pca/run_swat_pca.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from ics_tsfm.data.swat import load_swat_attack, load_swat_normal
from ics_tsfm.data.swat_preprocess import prepare_swat
from ics_tsfm.models.pca import PCADetector

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "codebase"))
from swat_cycle_analysis import SWAT_DIR  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "pca"
MODELS = {"plain": 1, "windowed_w50": 50}      # name -> window (history = window - 1)


def main():
    warnings.simplefilter("ignore")
    print("Loading SWaT and running Step 3 preprocessing...")
    split = prepare_swat(load_swat_normal(SWAT_DIR), load_swat_attack(SWAT_DIR))
    summary = {}
    for name, window in MODELS.items():
        det = PCADetector(window=window)
        train, val = split.anchors(det.history)
        t0 = time.time()
        det.fit(split.normal, train)
        fit_s = time.time() - t0
        out = OUT_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "val_scores.npy", det.score_series(split.normal, val))
        np.save(out / "test_scores.npy", det.score_series(split.test, split.test_rows([det.history])))
        summary[name] = {"window": window, "history": det.history, "k": det.k_,
                         "explained_variance": float(np.cumsum(det.explained_variance_ratio_)[det.k_ - 1]),
                         "fit_seconds": fit_s}
        print(f"  {name}: window {window}, k {det.k_}, fit {fit_s:.1f} s")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
