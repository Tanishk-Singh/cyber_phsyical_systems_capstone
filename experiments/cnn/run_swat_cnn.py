"""Phase A Step 6: train the CNN baseline on SWaT (3 seeds) and sanity-check it.

Config (locked 2026-11-16): Fung et al. / Kravchik & Shabtai's SWaT CNN,
8 conv layers x 32 filters, kernel 3, history 200, one-step-ahead prediction,
early stopping on our leak-free validation blocks, seeds 0, 1, 2.

What it writes (results/ is git-ignored):
  results/cnn/seed{n}/model.pt            weights + config + loss history
  results/cnn/seed{n}/val_scores.npy      score per Normal-file row (NaN = not scored)
  results/cnn/seed{n}/test_scores.npy     score per Attack-file row (NaN = not scored)
  results/cnn/training_curves.png         train / validation loss per epoch, all seeds
  results/cnn/summary.json                everything printed below

The sanity check is NOT the official evaluation (that is Steps 7-8). It uses
a placeholder threshold, the 99.5th / 99.9th percentile of validation scores,
judges each row on its own, and puts the CNN next to plain and windowed PCA
on exactly the same test rows (those every model can score).

Resumable: a seed whose model and scores are already saved is loaded, not
retrained, so an interrupted run can simply be started again.

Run from the repo root (about 2 minutes to load the data, then a few minutes
per seed on a GTX 1660 Ti):

    python experiments/cnn/run_swat_cnn.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from ics_tsfm.data.swat import load_swat_attack, load_swat_normal
from ics_tsfm.data.swat_preprocess import prepare_swat
from ics_tsfm.models.cnn import CNNDetector
from ics_tsfm.models.pca import PCADetector

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "codebase"))
from swat_cycle_analysis import BLUE, INK2, ORANGE, SWAT_DIR, apply_style, new_figure  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "cnn"
SEEDS = (0, 1, 2)
PERCENTILES = (99.5, 99.9)


def sanity(val_scores, test_scores, is_attack):
    """Placeholder-threshold numbers: per row, no consecutive-row rule."""
    out = {"roc_auc": float(roc_auc_score(is_attack, test_scores))}
    for q in PERCENTILES:
        tau = np.percentile(val_scores, q)
        recall = float((test_scores[is_attack] > tau).mean())
        false_alarms = int((test_scores[~is_attack] > tau).sum())
        caught = recall * is_attack.sum()
        precision = caught / (caught + false_alarms) if caught + false_alarms else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"p{q}"] = {
            "threshold": float(tau),
            "attack_rows_flagged_pct": 100 * recall,
            "normal_test_rows_flagged_pct": 100 * false_alarms / (~is_attack).sum(),
            "precision": precision, "point_f1": f1,
        }
    return out


def print_row(name, s):
    a, b = s["p99.5"], s["p99.9"]
    print(f"  {name:22s} ROC-AUC {s['roc_auc']:.3f} | p99.5: attack {a['attack_rows_flagged_pct']:5.1f}%  "
          f"false alarms {a['normal_test_rows_flagged_pct']:.2f}%  F1 {a['point_f1']:.3f} | "
          f"p99.9: attack {b['attack_rows_flagged_pct']:5.1f}%  false alarms "
          f"{b['normal_test_rows_flagged_pct']:.2f}%  F1 {b['point_f1']:.3f}")


def plot_curves(histories):
    apply_style()
    fig, axes = new_figure(
        11, 4.6,
        "The CNN stops improving within a few epochs; early stopping keeps the best one",
        "Mean squared prediction error per epoch, for each seed. Left: training samples; right: our leak-free "
        "validation blocks.\nThe marker shows the epoch whose weights were kept (validation loss lowest, then "
        "3 epochs without improvement).",
        ncols=2, sharey=False,
    )
    colors = [BLUE, ORANGE, INK2]
    for (seed, hist, best), color in zip(histories, colors):
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], color=color, label=f"seed {seed}")
        axes[1].plot(ep, [h["val_loss"] for h in hist], color=color, label=f"seed {seed}")
        axes[1].scatter([best], [hist[best - 1]["val_loss"]], color=color, s=40, zorder=4)
    for ax, title in zip(axes, ("Training loss", "Validation loss")):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.legend()
    axes[0].set_ylabel("Mean squared error (log scale)")
    fig.savefig(OUT_DIR / "training_curves.png")
    plt.close(fig)


def main():
    warnings.simplefilter("ignore")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading SWaT and running Step 3 preprocessing...")
    split = prepare_swat(load_swat_normal(SWAT_DIR), load_swat_attack(SWAT_DIR))

    history = CNNDetector().history
    rows = split.test_rows([0, 49, history])      # rows every compared model can score
    is_attack = split.test_is_attack[rows]
    summary = {"config": CNNDetector().config(), "n_test_rows": int(len(rows)),
               "n_attack_rows": int(is_attack.sum()), "seeds": {}}

    train, val = split.anchors(history)
    print(f"CNN history {history}: {len(train):,} training and {len(val):,} validation samples")
    histories = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        seed_dir = OUT_DIR / f"seed{seed}"
        files = [seed_dir / name for name in ("model.pt", "val_scores.npy", "test_scores.npy")]
        if all(f.exists() for f in files):
            # Resume: this seed finished in an earlier, interrupted run.
            print("  already trained; loading saved model and scores")
            det = CNNDetector.load(files[0], split.normal.shape[1])
            train_time = sum(h["seconds"] for h in det.history_)
            val_series, test_series = np.load(files[1]), np.load(files[2])
        else:
            det = CNNDetector(seed=seed)
            t0 = time.time()
            det.fit(split.normal, train, val)
            train_time = time.time() - t0
            seed_dir.mkdir(exist_ok=True)
            val_series = det.score_series(split.normal, val)
            test_series = det.score_series(split.test, split.test_rows([history]))
            np.save(files[1], val_series)
            np.save(files[2], test_series)
            det.save(files[0])            # saved last, so a half-written seed is retrained

        # Step 3's leak check: error on training samples vs validation samples,
        # both with the final weights in evaluation mode. Medians, not means:
        # P301's 11 "on" rows in training (its blip at hour 22.5) scale to
        # about 187 spreads and give a few windows huge errors that swamp a mean.
        train_err = float(np.median(det.score(split.normal, train)))
        val_err = float(np.nanmedian(val_series))
        s = sanity(val_series[val], test_series[rows], is_attack)
        s.update({"epochs_run": len(det.history_), "best_epoch": det.best_epoch_,
                  "train_seconds": train_time,
                  "seconds_per_epoch": float(np.mean([h["seconds"] for h in det.history_])),
                  "train_error": train_err, "val_error": val_err,
                  "val_over_train": val_err / train_err})
        summary["seeds"][str(seed)] = s
        histories.append((seed, det.history_, det.best_epoch_))
        print(f"  kept epoch {det.best_epoch_} of {len(det.history_)}; {train_time / 60:.1f} min; "
              f"median train error {train_err:.5f}, median validation error {val_err:.5f} "
              f"(ratio {val_err / train_err:.2f})")

    print("\nFitting PCA baselines on the same split for comparison...")
    for name, window in (("plain_pca", 1), ("windowed_pca_w50", 50)):
        det = PCADetector(window=window)
        tr, va = split.anchors(det.history)
        det.fit(split.normal, tr)
        summary[name] = sanity(det.score(split.normal, va), det.score(split.test, rows), is_attack)
        summary[name]["k"] = det.k_

    print(f"\nSanity check on {len(rows):,} shared test rows ({is_attack.sum():,} attack rows); "
          f"placeholder threshold, per row:")
    for seed in SEEDS:
        print_row(f"CNN seed {seed}", summary["seeds"][str(seed)])
    print_row(f"plain PCA (k={summary['plain_pca']['k']})", summary["plain_pca"])
    print_row("windowed PCA (w=50)", summary["windowed_pca_w50"])

    plot_curves(histories)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT_DIR.relative_to(REPO_ROOT)}/ (models, scores, training_curves.png, summary.json)")


if __name__ == "__main__":
    main()
