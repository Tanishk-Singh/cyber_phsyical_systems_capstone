"""Phase B Step B6: evaluate the Chronos-2 scores exactly like the Phase A baselines.

Same rule and metrics as Phase A Steps 7-8 (experiments/detection and
experiments/metrics), applied to every Chronos-2 setting, with and without
Forecast Fallback, next to plain PCA, windowed PCA and the three CNN seeds:
  - detection rule: tau = highest benign validation score, then an alarm only
    after w = 10 consecutive rows above tau; false alarms counted as events
  - threshold-free: VUS-PR (headline), VUS-ROC, ROC-AUC, AUC-PR
  - on the alarms: range-F1, attacks detected, false-alarm events per day,
    point-F1; overall, per attack type (SSSP / SSMP / MSMP) and per attack
No point-adjust; no threshold is searched on test labels.

Comparison groups
-----------------
Every model in a group is scored on the rows ALL of them can score, so a
model with a longer history decides the rows (Phase A rule). The Chronos-2
settings are grouped by the history that decides their rows:
  history 200   Lowe's grid (contexts 32-128): the CNN's 200 rows decide, so
                these are exactly the Phase A rows and the baselines must
                reproduce the Phase A table (checked and printed below).
  history 4096  our context-4096 setting: it cannot score the first 4,096
                rows of the Attack file (attacks 1 and 2 lie there) nor the
                4,096 rows after the 82 s recording gap (attack 31 loses its
                first 280 rows). The baselines are re-scored on the same rows.
An attack with no scored row is left out of that group (listed in the
output), not counted as missed.

Validation rows: each model's tau comes from all the validation rows it can
score. The CNN's validation rows exclude samples whose history crosses a
training block (the Phase A purge, needed because it was trained); the
zero-shot Chronos-2 learned nothing, so it scores every validation row.

Reads
  results/pca/{plain,windowed_w50}/, results/cnn/seed{0,1,2}/   (Phase A)
  results/tsfm/chronos2/<setting>/                              (Step B4, any found)
  results/tsfm/chronos2_ff/<setting>/<variant>/                 (Step B5, any found)
Writes (git-ignored) results/tsfm_eval/: overall.csv, per_type.csv,
per_attack.csv, groups.json, cache/ (VUS results; reruns are fast and only
compute what is new, e.g. after more per-sensor settings arrive).

    python experiments/tsfm/evaluate_swat_chronos2.py              # 6 parallel VUS workers
    python experiments/tsfm/evaluate_swat_chronos2.py --workers 3
"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
TSFM_RUNS = {   # display label: summary files to look for, under results/tsfm/
    "Chronos-2": "chronos2/*/summary.json",                         # Step B4, no fallback
    "Chronos-2 FF-Lowe": "chronos2_ff/*/lowe/summary.json",         # Step B5, the published rule
    "Chronos-2 FF-bounded": "chronos2_ff/*/bounded/summary.json",   # Step B5, ours (additions A + B)
    "Chronos-2 FF-clamped": "chronos2_ff/*/clamped/summary.json",   # Step B5, ours (additions A + C)
}
OUT_DIR = RESULTS / "tsfm_eval"
CACHE = OUT_DIR / "cache"
BASELINES = {
    "Plain PCA": RESULTS / "pca" / "plain",
    "Windowed PCA (w=50)": RESULTS / "pca" / "windowed_w50",
    "CNN seed 0": RESULTS / "cnn" / "seed0",
    "CNN seed 1": RESULTS / "cnn" / "seed1",
    "CNN seed 2": RESULTS / "cnn" / "seed2",
}
CNN_HISTORY = 200
TYPES = ("SSSP", "SSMP", "MSMP")


def chronos_settings():
    """{display name: (folder, context)} for every finished Chronos-2 setting,
    e.g. "Chronos-2 joint 128x32" and "Chronos-2 FF-bounded joint 128x32"."""
    found = {}
    for label, pattern in TSFM_RUNS.items():
        for summary in sorted((RESULTS / "tsfm").glob(pattern)):
            s = json.loads(summary.read_text())
            found[f"{label} {s['mode']} {s['context']}x{s['horizon']}"] = (summary.parent, s["context"])
    return found


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def vus_job(key, scores, labels):
    return key, vus(scores, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=6, help="parallel VUS computations")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    types = load_attack_types(SWAT_DIR)
    attack_of_row = np.zeros(N_ATTACK_ROWS, int)                 # 0 = normal
    for start, end, number in CORRECTED_ATTACK_RANGES:
        attack_of_row[start:end] = number
    is_attack = attack_of_row > 0
    type_of_row = np.where(is_attack, pd.Series(attack_of_row).map(types["type"]).fillna("").to_numpy(), "normal")

    chronos = chronos_settings()
    groups = {}
    for name, (folder, context) in chronos.items():
        groups.setdefault(max(CNN_HISTORY, context), {}).update({name: folder})
    scores = {name: (np.load(d / "val_scores.npy"), np.load(d / "test_scores.npy"))
              for name, d in {**BASELINES, **{n: f for n, (f, _) in chronos.items()}}.items()}
    w = DEFAULT_PERSISTENCE
    print(f"{len(chronos)} Chronos-2 settings found; rule: tau = highest validation score, "
          f"persistence w = {w} s\n")

    # ---- pass 1: shared rows, alarms and every series that needs a VUS
    plan, jobs, group_info = [], {}, {}
    for history, members in sorted(groups.items()):
        models = {**BASELINES, **members}
        scored = np.logical_and.reduce([~np.isnan(scores[n][1]) for n in models])
        mask_id = hashlib.sha1(np.packbits(scored)).hexdigest()[:10]
        ranges = [r for r in CORRECTED_ATTACK_RANGES if scored[r[0]:r[1]].any()]
        dropped = [int(r[2]) for r in CORRECTED_ATTACK_RANGES if r not in ranges]
        partial = [int(r[2]) for r in ranges if not scored[r[0]:r[1]].all()]
        group_info[history] = {"models": list(models), "rows": int(scored.sum()),
                               "attack_rows": int((is_attack & scored).sum()), "attacks": len(ranges),
                               "attacks_left_out": dropped, "attacks_partly_scored": partial}
        print(f"Group history {history}: {scored.sum():,} shared rows, {(is_attack & scored).sum():,} attack rows, "
              f"{len(ranges)} attacks" + (f"; left out (no scored row): {dropped}" if dropped else "")
              + (f"; partly scored: {partial}" if partial else ""))
        for name in models:
            val, test = scores[name]
            tau = zero_alarm_threshold(val, w=1)
            alarm = alarms(np.where(scored, test, np.nan), tau, w)
            series = {"all": scored}
            series.update({k: scored & ((type_of_row == "normal") | (type_of_row == k)) for k in TYPES})
            for part, keep in series.items():
                key = f"h{history}_{slug(name)}_{part}_{mask_id}"
                if not (CACHE / f"{key}.json").exists():
                    jobs[key] = (test[keep], is_attack[keep])
            plan.append((history, name, tau, alarm, scored, ranges, series, mask_id))

    # ---- pass 2: the slow part, VUS, in parallel and cached
    if jobs:
        print(f"\nComputing {len(jobs)} VUS series with {args.workers} workers "
              f"(about 2 min each; cached in {CACHE.relative_to(REPO_ROOT)})...", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(vus_job, k, *v) for k, v in jobs.items()]
            for i, done in enumerate(as_completed(futures), 1):
                key, result = done.result()
                (CACHE / f"{key}.json").write_text(json.dumps(result))
                print(f"  {i}/{len(jobs)} done", flush=True)

    # ---- pass 3: tables
    overall, per_type, per_attack = [], [], []
    for history, name, tau, alarm, scored, ranges, series, mask_id in plan:
        ev = score_alarms(alarm, is_attack, scored, ranges)

        def metrics(part):
            vus_pr, vus_roc = json.loads((CACHE / f"h{history}_{slug(name)}_{part}_{mask_id}.json").read_text())
            keep = series[part]
            return {"VUS-PR": vus_pr, "VUS-ROC": vus_roc, "ROC-AUC": roc_auc(scores[name][1][keep], is_attack[keep]),
                    "AUC-PR": auc_pr(scores[name][1][keep], is_attack[keep]),
                    "Range-F1": range_f1(is_attack[keep], alarm[keep])}

        overall.append({"group": f"history {history}", "model": name, **metrics("all"),
                        "attacks_detected": ev["attacks_detected"], "attacks_total": ev["attacks_total"],
                        "false_alarm_events_per_day": ev["false_alarm_events_per_day"],
                        "median_delay_s": ev["median_delay_s"], "point_f1": ev["point_f1"], "tau": tau})
        in_group = {r[2] for r in ranges}
        for kind in TYPES:
            numbers = [n for n in types.index[types["type"] == kind] if n in in_group]
            per_type.append({"group": f"history {history}", "model": name, "type": kind,
                             "n_attacks": len(numbers), **metrics(kind),
                             "attacks_detected": sum(a["detected"] for a in ev["per_attack"] if a["attack"] in numbers)})
        per_attack += [{"group": f"history {history}", "model": name, "type": types.loc[a["attack"], "type"], **a}
                       for a in ev["per_attack"]]

    overall, per_type, per_attack = pd.DataFrame(overall), pd.DataFrame(per_type), pd.DataFrame(per_attack)
    overall.to_csv(OUT_DIR / "overall.csv", index=False)
    per_type.to_csv(OUT_DIR / "per_type.csv", index=False)
    per_attack.to_csv(OUT_DIR / "per_attack.csv", index=False)
    (OUT_DIR / "groups.json").write_text(json.dumps({f"history {h}": g for h, g in group_info.items()}, indent=2))

    pd.set_option("display.width", 250)
    cols = ["model", "VUS-PR", "ROC-AUC", "Range-F1", "attacks_detected", "attacks_total",
            "false_alarm_events_per_day", "point_f1", "tau"]
    for group, table in overall.groupby("group", sort=False):
        print(f"\n=== {group} ===")
        print(table[cols].round(3).to_string(index=False))

    # Phase A check: on the history-200 rows the baselines must match the Phase A table
    phase_a = RESULTS / "metrics" / "overall.csv"
    if phase_a.exists() and CNN_HISTORY in groups:
        ref = pd.read_csv(phase_a).set_index("model")
        mine = overall[overall["group"] == f"history {CNN_HISTORY}"].set_index("model")
        diff = max(abs(mine.loc[m, c] - ref.loc[m, c]) for m in BASELINES
                   for c in ("VUS-PR", "ROC-AUC", "Range-F1", "point_f1"))
        print(f"\nPhase A check (baselines, history 200 vs results/metrics/overall.csv): "
              f"largest difference {diff:.2e} -> {'MATCHES' if diff < 1e-9 else 'DIFFERS, investigate'}")
    print(f"\nwrote {OUT_DIR.relative_to(REPO_ROOT)}/ (overall.csv, per_type.csv, per_attack.csv, groups.json)")


if __name__ == "__main__":
    main()
