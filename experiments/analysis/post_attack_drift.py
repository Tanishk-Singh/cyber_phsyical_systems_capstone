"""Is the drift in the Attack file's normal rows caused by the attacks?

Why this exists
---------------
Hypothesis (Tanishk and Afjal, 2026-11-17): normal rows in the Attack file
drift because the plant takes time to settle after each attack, so a third
label, "stabilising", could separate that recovery from truly normal rows.
This script measures how much of the drift and of the false alarms really sits
after attacks.

Four measurements
-----------------
  1. Share of normal test rows above each model's 99.9th validation percentile,
     and their median score relative to validation, by time since the last
     attack ended.
  2. Share of normal test rows outside the training range, per drifting
     sensor, by time since the last attack ended; plus whether the same
     sensors already drift inside the attack-free Normal file.
  3. Where the false-alarm rows sit, at the 99.9th and 99.5th validation
     percentiles: "carry-over" (within a model's history after an attack, its
     input window still contains attack rows), "recovery" (from there up to 1
     hour after an attack), or more than 1 hour after any attack.
  4. At the CNN's low threshold (tau calibrated with persistence w = 10 inside
     the calibration, about 0.3), how many false-alarm EVENTS start within
     15 / 60 minutes after an attack.

Result (2026-11-17/23): two separate causes. A real but short aftermath
(minutes to about 1 hour, about 20% of normal test time) holds a large share
of false alarms at the strict 99.9th percentile: windowed PCA 79% (60%
carry-over + 19% recovery), plain PCA 41%, CNN 38-74% depending on the seed
(seed 2: 21% + 53%; seed 0: 12% + 38%; seed 1: 7% + 31%). A slow drift
unrelated to attacks (AIT504 out of the training range 45-64% of the time even
4+ hours after any attack, and drifting inside the attack-free Normal file)
dominates at looser thresholds (78% of CNN false-alarm rows at the 99.5th
percentile, every seed) and at the CNN's low threshold (only 14-16% of
false-alarm events start within an hour of an attack).

Reads saved scores (experiments/pca/run_swat_pca.py, experiments/cnn/
run_swat_cnn.py) and loads the data for measurement 2 (about 2 minutes):

    python experiments/analysis/post_attack_drift.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ics_tsfm.data.swat import load_swat_attack, load_swat_normal
from ics_tsfm.data.swat_corrections import CORRECTED_ATTACK_RANGES, N_ATTACK_ROWS
from ics_tsfm.data.swat_preprocess import prepare_swat
from ics_tsfm.detect.rule import alarms, events, zero_alarm_threshold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "codebase"))
from swat_cycle_analysis import SWAT_DIR  # noqa: E402

RESULTS = REPO_ROOT / "results"
MODELS = {  # name -> (folder, history in rows)
    "Plain PCA": (RESULTS / "pca" / "plain", 0),
    "Windowed PCA": (RESULTS / "pca" / "windowed_w50", 49),
    "CNN seed 0": (RESULTS / "cnn" / "seed0", 200),
    "CNN seed 1": (RESULTS / "cnn" / "seed1", 200),
    "CNN seed 2": (RESULTS / "cnn" / "seed2", 200),
}
TIME_BINS = [(None, "before first attack"), ((0, 300), "0-5 min"), ((300, 900), "5-15 min"),
             ((900, 1800), "15-30 min"), ((1800, 3600), "30-60 min"), ((3600, 7200), "1-2 h"),
             ((7200, 14400), "2-4 h"), ((14400, 10**9), "> 4 h")]
SENSORS = ("AIT504", "FIT504", "FIT401", "AIT202", "LIT101")
NEVER = 10**9


def labels_and_time_since_attack():
    is_attack = np.zeros(N_ATTACK_ROWS, bool)
    for start, end, _ in CORRECTED_ATTACK_RANGES:
        is_attack[start:end] = True
    ends = np.array(sorted(end for _, end, _ in CORRECTED_ATTACK_RANGES))
    idx = np.arange(N_ATTACK_ROWS)
    k = np.searchsorted(ends, idx, side="right") - 1
    since = np.where(k >= 0, idx - ends[np.clip(k, 0, None)], -1)    # -1 = before the first attack
    return is_attack, since


def in_bin(since, bin_):
    if bin_ is None:
        return since == -1
    lo, hi = bin_
    return (since >= lo) & (since < hi)


def main():
    warnings.simplefilter("ignore")
    is_attack, since = labels_and_time_since_attack()
    scores = {n: (np.load(d / "val_scores.npy"), np.load(d / "test_scores.npy"), h) for n, (d, h) in MODELS.items()}
    order = [b[1] for b in TIME_BINS]
    pd.set_option("display.width", 180)

    # 1. false-alarm-like rows and score level by time since the last attack
    rate, level = {}, {}
    for name, (val, test, _) in scores.items():
        tau, med = np.nanpercentile(val, 99.9), np.nanmedian(val)
        normal = ~is_attack & ~np.isnan(test)
        rate[name] = {lbl: (test[normal & in_bin(since, b)] > tau).mean() * 100 for b, lbl in TIME_BINS}
        level[name] = {lbl: np.median(test[normal & in_bin(since, b)]) / med for b, lbl in TIME_BINS}
    hours = {lbl: (~is_attack & in_bin(since, b)).sum() / 3600 for b, lbl in TIME_BINS}
    print("1. Normal test rows above the 99.9th validation percentile (%), by time since the last attack ended")
    print(pd.DataFrame(rate).reindex(order).assign(hours=pd.Series(hours)).round(2).to_string())
    print("\n   Median score of normal test rows / median validation score")
    print(pd.DataFrame(level).reindex(order).round(2).to_string())

    # 2. sensors outside the training range, and drift inside the Normal file itself
    print("\nLoading SWaT for the sensor-level view...")
    split = prepare_swat(load_swat_normal(SWAT_DIR), load_swat_attack(SWAT_DIR))
    train = split.normal[~split.is_val]
    outside = {}
    for col in SENSORS:
        j = split.columns.index(col)
        lo, hi, x = train[:, j].min(), train[:, j].max(), split.test[:, j]
        outside[col] = {lbl: ((x < lo) | (x > hi))[~is_attack & in_bin(since, b)].mean() * 100 for b, lbl in TIME_BINS}
    print("\n2. Normal test rows outside the training range (%), by time since the last attack ended")
    print(pd.DataFrame(outside).reindex(order).round(1).to_string())
    print("\n   Drift inside the attack-free Normal file (last 20% vs first 80%):")
    for col in ("AIT504", "FIT504", "FIT401"):
        x = split.normal[:, split.columns.index(col)]
        cut = int(0.8 * len(x))
        first, last = x[:cut], x[cut:]
        print(f"   {col}: mean shift {(last.mean() - first.mean()) / first.std():+.2f} spreads")

    # 3. where the false-alarm rows sit
    print("\n3. Where the false-alarm rows sit")
    for name, (val, test, history) in scores.items():
        normal = ~is_attack & ~np.isnan(test)
        after = np.where(since >= 0, since, NEVER)
        for q in (99.9, 99.5):
            fa = normal & (test > np.nanpercentile(val, q))
            n = fa.sum()
            carry = (after[fa] < history).sum()
            recovery = ((after[fa] >= history) & (after[fa] < 3600)).sum()
            print(f"   {name:13s} p{q}: {n:6d} rows | carry-over (< {history} s) {carry / n * 100:3.0f}% | "
                  f"recovery (to 1 h) {recovery / n * 100:3.0f}% | > 1 h after an attack "
                  f"{(n - carry - recovery) / n * 100:3.0f}%")

    # 4. false-alarm events at the CNN's low threshold, by when they start
    print("\n4. CNN false-alarm events at the low threshold (tau calibrated with w = 10)")
    scored = np.logical_and.reduce([~np.isnan(t) for _, t, _ in scores.values()])
    for name in ("CNN seed 0", "CNN seed 1", "CNN seed 2"):
        val, test, _ = scores[name]
        tau = zero_alarm_threshold(val, 10)
        runs = events(alarms(np.where(scored, test, np.nan), tau, 10))
        false = np.array([(s, e) for s, e in runs if not is_attack[s:e].any()])
        start_since = np.where(since[false[:, 0]] >= 0, since[false[:, 0]], NEVER)
        duration = false[:, 1] - false[:, 0]
        for minutes in (15, 60):
            w = start_since < minutes * 60
            print(f"   {name} (tau {tau:.3f}): {len(false)} events; starting within {minutes} min after an attack: "
                  f"{w.mean() * 100:.0f}% of events, {duration[w].sum() / duration.sum() * 100:.0f}% of false-alarm time")


if __name__ == "__main__":
    main()
