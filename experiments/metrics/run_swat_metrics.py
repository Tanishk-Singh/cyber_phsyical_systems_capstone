"""Phase A Step 8: the evaluation table for every model on SWaT.

For each model (plain PCA, windowed PCA, CNN seeds 0-2), on the shared test
rows every model can score:
  - threshold-free: VUS-PR (headline), VUS-ROC, ROC-AUC, AUC-PR
  - on the Step 7 alarms (decoupled rule: tau = highest validation score,
    then persistence w = 10 s): range-F1, attacks detected, false-alarm
    events per day, point-F1
overall, per attack type (SSSP / SSMP / MSMP, derived from the attack list)
and per attack. No point-adjust anywhere; no threshold is searched on test
labels.

Per attack type: the series keeps every normal row and only that type's
attacks; the other types' attack rows are left out, so each type is scored
on its own.

Reads saved scores (run experiments/pca/run_swat_pca.py and
experiments/cnn/run_swat_cnn.py first). VUS-PR takes about 2 minutes per
series (4 per model), so the whole run is about 40 minutes; every VUS result
is cached in results/metrics/cache/, so an interrupted run resumes.

    python experiments/metrics/run_swat_metrics.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ics_tsfm.data.swat_attacks import load_attack_types
from ics_tsfm.data.swat_corrections import CORRECTED_ATTACK_RANGES, N_ATTACK_ROWS
from ics_tsfm.detect.rule import DEFAULT_PERSISTENCE, alarms, zero_alarm_threshold
from ics_tsfm.metrics.alarms import score_alarms
from ics_tsfm.metrics.threshold_free import auc_pr, range_f1, roc_auc, vus

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "codebase"))
from swat_cycle_analysis import SWAT_DIR  # noqa: E402

RESULTS = REPO_ROOT / "results"
OUT_DIR = RESULTS / "metrics"
CACHE = OUT_DIR / "cache"
MODELS = {
    "Plain PCA": RESULTS / "pca" / "plain",
    "Windowed PCA (w=50)": RESULTS / "pca" / "windowed_w50",
    "CNN seed 0": RESULTS / "cnn" / "seed0",
    "CNN seed 1": RESULTS / "cnn" / "seed1",
    "CNN seed 2": RESULTS / "cnn" / "seed2",
}


def cached_vus(key: str, scores, labels):
    path = CACHE / f"{key}.json"
    if path.exists():
        return tuple(json.loads(path.read_text()))
    result = vus(scores, labels)
    path.write_text(json.dumps(result))
    return result


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def evaluate(key, scores, labels, alarm):
    vus_pr, vus_roc = cached_vus(key, scores, labels)
    return {"VUS-PR": vus_pr, "VUS-ROC": vus_roc, "ROC-AUC": roc_auc(scores, labels),
            "AUC-PR": auc_pr(scores, labels), "Range-F1": range_f1(labels, alarm)}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    types = load_attack_types(SWAT_DIR)
    attack_of_row = np.zeros(N_ATTACK_ROWS, int)                 # 0 = normal
    for start, end, number in CORRECTED_ATTACK_RANGES:
        attack_of_row[start:end] = number
    is_attack = attack_of_row > 0
    type_of_row = np.where(is_attack, pd.Series(attack_of_row).map(types["type"]).fillna("").to_numpy(), "normal")

    scores = {n: (np.load(d / "val_scores.npy"), np.load(d / "test_scores.npy")) for n, d in MODELS.items()}
    scored = np.logical_and.reduce([~np.isnan(t) for _, t in scores.values()])
    w = DEFAULT_PERSISTENCE
    print(f"{scored.sum():,} shared test rows, {(is_attack & scored).sum():,} attack rows; "
          f"rule: tau = highest validation score, persistence w = {w} s\n")

    overall, per_type, per_attack = [], [], []
    for name, (val, test) in scores.items():
        tau = zero_alarm_threshold(val, w=1)
        alarm = alarms(np.where(scored, test, np.nan), tau, w)
        ev = score_alarms(alarm, is_attack, scored, CORRECTED_ATTACK_RANGES)
        print(f"{name}: overall...", flush=True)
        row = {"model": name, **evaluate(f"{slug(name)}_all", test[scored], is_attack[scored], alarm[scored]),
               "attacks_detected": ev["attacks_detected"], "false_alarm_events_per_day": ev["false_alarm_events_per_day"],
               "point_f1": ev["point_f1"], "tau": tau}
        overall.append(row)
        for kind in ("SSSP", "SSMP", "MSMP"):
            keep = scored & ((type_of_row == "normal") | (type_of_row == kind))
            numbers = [n for n in types.index[types["type"] == kind] if n in set(attack_of_row)]
            detected = sum(a["detected"] for a in ev["per_attack"] if a["attack"] in numbers)
            print(f"{name}: {kind}...", flush=True)
            per_type.append({"model": name, "type": kind, "n_attacks": len(numbers),
                             **evaluate(f"{slug(name)}_{kind}", test[keep], is_attack[keep], alarm[keep]),
                             "attacks_detected": detected})
        per_attack += [{"model": name, "type": types.loc[a["attack"], "type"],
                        "points": types.loc[a["attack"], "points"], **a} for a in ev["per_attack"]]

    overall, per_type, per_attack = pd.DataFrame(overall), pd.DataFrame(per_type), pd.DataFrame(per_attack)
    overall.to_csv(OUT_DIR / "overall.csv", index=False)
    per_type.to_csv(OUT_DIR / "per_type.csv", index=False)
    per_attack.to_csv(OUT_DIR / "per_attack.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== Overall ===")
    print(overall.round(3).to_string(index=False))
    print("\n=== Per attack type (VUS-PR | attacks detected) ===")
    print(per_type.pivot(index="model", columns="type", values="VUS-PR").round(3).to_string())
    print(per_type.pivot(index="model", columns="type", values="attacks_detected").to_string())
    print(f"\nwrote {OUT_DIR.relative_to(REPO_ROOT)}/ (overall.csv, per_type.csv, per_attack.csv)")


if __name__ == "__main__":
    main()
