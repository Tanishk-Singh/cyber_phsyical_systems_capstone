"""Phase A Step 7: apply the detection rule to every model's saved scores.

Rule (2026-11-17, "decoupled"): tau = the highest benign validation score
(zero false alarms on validation without persistence), then alarm persistence
w = 10 s applied to the test alarms; false alarms counted as events. Same rule
for every model. Validation scores set tau; test labels are used only to
score the alarms afterwards. The first version calibrated tau WITH w = 10 and
flooded the CNN with false alarms (see detect/rule.py); it is reported as
"locked rule (w=10 in calibration)" for transparency.

Every model is scored on the same test rows: those every model can score
(the longest history, the CNN's 200 rows, decides; this is the intersection
of the rows where each model's saved test score is not NaN).

Reads (run these first):
  results/pca/{plain,windowed_w50}/   from experiments/pca/run_swat_pca.py
  results/cnn/seed{0,1,2}/            from experiments/cnn/run_swat_cnn.py
Writes (git-ignored):
  results/detection/summary.json      all numbers below
  results/detection/per_attack.csv    detected / delay per attack and model

Needs no dataset loading (labels come from the corrected attack ranges), so it
runs in seconds:

    python experiments/detection/run_swat_detection.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ics_tsfm.data.swat_corrections import CORRECTED_ATTACK_RANGES, N_ATTACK_ROWS
from ics_tsfm.detect.rule import DEFAULT_PERSISTENCE, alarms, zero_alarm_threshold
from ics_tsfm.metrics.alarms import score_alarms

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
OUT_DIR = RESULTS / "detection"
MODELS = {
    "Plain PCA": RESULTS / "pca" / "plain",
    "Windowed PCA (w=50)": RESULTS / "pca" / "windowed_w50",
    "CNN seed 0": RESULTS / "cnn" / "seed0",
    "CNN seed 1": RESULTS / "cnn" / "seed1",
    "CNN seed 2": RESULTS / "cnn" / "seed2",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    is_attack = np.zeros(N_ATTACK_ROWS, bool)
    for start, end, _ in CORRECTED_ATTACK_RANGES:
        is_attack[start:end] = True

    scores = {name: (np.load(d / "val_scores.npy"), np.load(d / "test_scores.npy")) for name, d in MODELS.items()}
    scored = np.logical_and.reduce([~np.isnan(test) for _, test in scores.values()])
    w = DEFAULT_PERSISTENCE
    print(f"Rule: tau = highest validation score, then persistence w = {w} s; events counted.")
    print(f"Shared test rows: {scored.sum():,} ({(is_attack & scored).sum():,} attack rows, "
          f"{(~is_attack & scored).sum() / 3600:.1f} h of normal operation)\n")

    summary, per_attack = {"persistence_w": w, "n_scored_rows": int(scored.sum())}, []
    for name, (val, test) in scores.items():
        tau = zero_alarm_threshold(val, w=1)                             # highest validation score
        # the rule only ever looks backwards, so masking first and alarming after is equivalent
        test_alarm = alarms(np.where(scored, test, np.nan), tau, w)
        r = score_alarms(test_alarm, is_attack, scored, CORRECTED_ATTACK_RANGES)
        r["tau"] = tau
        r["validation_alarms"] = int(alarms(val, tau, 1).sum())          # 0 by construction
        # transparency: the first ("locked") version, tau calibrated with w = 10
        tau_locked = zero_alarm_threshold(val, w)
        locked = score_alarms(alarms(np.where(scored, test, np.nan), tau_locked, w),
                              is_attack, scored, CORRECTED_ATTACK_RANGES)
        r["locked_rule"] = {"tau": tau_locked, **{k: locked[k] for k in
                            ("attacks_detected", "false_alarm_events", "false_alarm_events_per_day", "point_f1")}}
        summary[name] = {k: v for k, v in r.items() if k != "per_attack"}
        per_attack += [{"model": name, **a} for a in r["per_attack"]]
        print(f"{name:22s} tau {tau:9.3f} | attacks {r['attacks_detected']:2d}/{r['attacks_total']} "
              f"(median delay {r['median_delay_s']:.0f} s) | false-alarm events {r['false_alarm_events']:4d} "
              f"({r['false_alarm_events_per_day']:.1f}/day, {r['false_alarm_seconds'] / 60:.0f} min total) | "
              f"point P {r['point_precision']:.3f} R {r['point_recall']:.3f} F1 {r['point_f1']:.3f}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    table = pd.DataFrame(per_attack)
    table.to_csv(OUT_DIR / "per_attack.csv", index=False)
    detected = table.pivot(index="attack", columns="model", values="detected")[list(MODELS)]
    never = detected.index[~detected.any(axis=1)].tolist()
    print(f"\nAttacks no model detects: {never}")
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}/ (summary.json, per_attack.csv)")


if __name__ == "__main__":
    main()
