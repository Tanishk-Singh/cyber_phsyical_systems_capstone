"""Phase A Step 8: PCA sensitivity table (k x window).

The headline PCA settings were fixed in advance (k from 99% of training
variation; window 1 and 50). This table shows how much those choices matter.
It is REPORTED, never used to pick a setting: picking the best cell would be
tuning on test labels.

Grid: window (rows) 1, 10, 30, 50, 120  x  k from 90 / 95 / 99 / 99.9% of
training variation. Each window is fitted once (at 99.9%) and the smaller k
are read off the same fit (PCADetector.score_for_each_k).

Every cell uses the same evaluation as the main table: shared test rows (the
CNN's history of 200 decides), VUS-PR, ROC-AUC, and the Step 7 decoupled
rule (tau = highest validation score, persistence w = 10 s) for range-F1,
attacks detected and false alarms.

About 2 minutes to load the data, a few minutes of fitting (window 120 has
4,320 numbers per sample), then about 2 minutes of VUS-PR per cell (20
cells); VUS results are cached, so an interrupted run resumes.

    python experiments/pca/run_swat_pca_sensitivity.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ics_tsfm.data.swat import load_swat_attack, load_swat_normal
from ics_tsfm.data.swat_corrections import CORRECTED_ATTACK_RANGES
from ics_tsfm.data.swat_preprocess import prepare_swat
from ics_tsfm.detect.rule import DEFAULT_PERSISTENCE, alarms, zero_alarm_threshold
from ics_tsfm.metrics.alarms import score_alarms
from ics_tsfm.metrics.threshold_free import range_f1, roc_auc, vus
from ics_tsfm.models.pca import PCADetector

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "codebase"))
from swat_cycle_analysis import SWAT_DIR  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "pca_sensitivity"
CACHE = OUT_DIR / "cache"
WINDOWS = (1, 10, 30, 50, 120)
VARIANCES = (0.90, 0.95, 0.99, 0.999)
SHARED_HISTORY = 200          # the CNN's, so every cell uses the main table's rows


def main():
    warnings.simplefilter("ignore")
    CACHE.mkdir(parents=True, exist_ok=True)
    print("Loading SWaT and running Step 3 preprocessing...")
    split = prepare_swat(load_swat_normal(SWAT_DIR), load_swat_attack(SWAT_DIR))
    rows = split.test_rows([SHARED_HISTORY])
    scored = np.zeros(len(split.test), bool)
    scored[rows] = True
    is_attack = split.test_is_attack
    w = DEFAULT_PERSISTENCE

    table = []
    for window in WINDOWS:
        det = PCADetector(window=window, variance=max(VARIANCES))
        train, val = split.anchors(det.history)
        t0 = time.time()
        det.fit(split.normal, train)
        cum = np.cumsum(det.explained_variance_ratio_)
        ks = {v: int(np.searchsorted(cum, v) + 1) for v in VARIANCES}
        val_scores = det.score_for_each_k(split.normal, val, ks.values())
        test_scores = det.score_for_each_k(split.test, rows, ks.values())
        print(f"window {window}: fitted in {time.time() - t0:.0f} s; k = {ks}", flush=True)

        for v, k in ks.items():
            series = np.full(len(split.test), np.nan)
            series[rows] = test_scores[k]
            tau = zero_alarm_threshold(val_scores[k], w=1)
            alarm = alarms(series, tau, w)
            ev = score_alarms(alarm, is_attack, scored, CORRECTED_ATTACK_RANGES)
            cache = CACHE / f"w{window}_k{k}.json"
            if cache.exists():
                vus_pr = json.loads(cache.read_text())[0]
            else:
                vus_pr = vus(series[rows], is_attack[rows])[0]
                cache.write_text(json.dumps([vus_pr]))
            table.append({"window": window, "variance": v, "k": k, "VUS-PR": vus_pr,
                          "ROC-AUC": roc_auc(series[rows], is_attack[rows]),
                          "Range-F1": range_f1(is_attack[rows], alarm[rows]),
                          "attacks_detected": ev["attacks_detected"],
                          "false_alarm_events_per_day": ev["false_alarm_events_per_day"],
                          "point_f1": ev["point_f1"]})
            print(f"   {v:.1%} (k = {k}): VUS-PR {vus_pr:.3f}, attacks {ev['attacks_detected']}/32, "
                  f"FA {ev['false_alarm_events_per_day']:.1f}/day", flush=True)

    table = pd.DataFrame(table)
    table.to_csv(OUT_DIR / "table.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nVUS-PR (rows: window, columns: share of variation kept)")
    print(table.pivot(index="window", columns="variance", values="VUS-PR").round(3).to_string())
    print("\nk")
    print(table.pivot(index="window", columns="variance", values="k").to_string())
    print(f"\nwrote {OUT_DIR.relative_to(REPO_ROOT)}/table.csv")


if __name__ == "__main__":
    main()
