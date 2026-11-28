"""Phase B Step B5: score SWaT with zero-shot Chronos-2 WITH Forecast Fallback.

Same settings, modes, rows and outputs as Step B4 (run_swat_chronos2.py),
plus Lowe et al.'s Forecast Fallback (src/ics_tsfm/models/tsfm/fallback.py):
a sensor whose forecast error exceeds its threshold on 3 rows in a row stops
feeding its real values to the model, which reads its own forecasts instead,
so an attack held for a long time stays visible.

Per setting
-----------
  1. Fallback thresholds: plain Chronos-2 forecasts (no fallback) on 30,000
     Normal-file TRAINING rows, whole forecast blocks spread evenly over all
     training blocks; threshold of each sensor = 99.5th percentile of its
     absolute error. Training rows, so the validation rows stay untouched for
     the detection threshold tau (Step B6), as for every other model.
  2. Validation rows of the Normal file, with fallback (tau comes from these).
  3. Every clean Attack-file row, with fallback.
Nothing is tuned on test labels.

What it writes (results/ is git-ignored)
----------------------------------------
  results/tsfm/chronos2_ff/<mode>_I<context>_H<horizon>_<dtype>/
    thresholds.json          per-sensor fallback thresholds and how they were made
    val_scores.npy           one score per Normal-file row (NaN = not scored)
    test_scores.npy          one score per Attack-file row (NaN = not scored)
    val_fallback.npy, test_fallback.npy
                             sensors in fallback at each row (-1 = not scored)
    cache_train/             plain forecasts for the thresholds (resumable chunks)
    state_val/, state_test/  fallback state (resumable checkpoints)
    summary.json             rows, time, sanity numbers, how often fallback was on
  results/tsfm/chronos2_ff/run_log.txt   everything printed
A setting whose summary.json exists is skipped.

Cost
----
Fallback forecasts must be made in order (each one depends on which earlier
rows were replaced), so the Attack file needs one forward pass per origin of
its longer run: about 298,000 passes at horizon 1, about 9,300 at horizon 32.
The two runs of the Attack file, and the ~20 runs of validation rows, share
each pass. The smoke test measures the cost of every setting on this machine.

Running in sessions (shared lab PC)
-----------------------------------
  --max-hours 12 or --stop-at 08:00 stops cleanly: the fallback state is
  saved at the limit (and every 5 minutes), and the same command run later
  continues from it with the same result as an uninterrupted run. A hard stop
  (Ctrl+C, shutdown) loses at most 5 minutes of work. Keep the same command
  (in particular --batch-series) for every session of a setting.

Smoke test (run this first on a new machine)
--------------------------------------------
  --smoke times every setting on a few dozen fallback steps around the onset
  of the longest attack plus the end of the file (two runs, as in the full
  run) and on the first steps of every validation run, then prints the
  estimated time of each full run. It also prints, around the onset, the
  share of sensors in fallback and the score with and without fallback, and
  checks that scoring the two runs together gives the same forecasts as
  scoring one alone. Writes results/tsfm_smoke/ff_smoke_report.json. About
  5-15 minutes on the RTX 5080.

Run from the repo root:

    python experiments/tsfm/run_swat_chronos2_ff.py --smoke
    python experiments/tsfm/run_swat_chronos2_ff.py --settings lowe --max-hours 12
    python experiments/tsfm/run_swat_chronos2_ff.py --settings ours --max-hours 12
    python experiments/tsfm/run_swat_chronos2_ff.py --settings 128x32           # a single setting
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ics_tsfm.data.swat import is_attack_mask, load_swat_attack, load_swat_normal
from ics_tsfm.data.swat_preprocess import prepare_swat
from ics_tsfm.data.windows import contiguous_anchors
from ics_tsfm.models.tsfm import (FF_PERCENTILE, FF_TRIGGER, ForecastDetector, ForecastFallbackDetector,
                                  TimeLimitReached, fallback_thresholds, origins, spread_blocks)
from ics_tsfm.models.tsfm.chronos2 import Chronos2Forecaster
from ics_tsfm.models.tsfm.fallback import MIN_THRESHOLD

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_swat_chronos2 import (REPO_ROOT, SMOKE_DIR, SWAT_DIR, Tee, chunk_origins, n_origins,  # noqa: E402
                               parse_settings, sanity, scored_rows, setting_name, smoke_rows, stop_time)

OUT_DIR = REPO_ROOT / "results" / "tsfm" / "chronos2_ff"
TRAIN_ROWS = 30_000


def train_sample(split, context, horizon, n_rows=TRAIN_ROWS):
    """About n_rows Normal-file training rows with a clean context, as whole
    forecast blocks spread over all training blocks (spread_blocks)."""
    rows = np.flatnonzero(contiguous_anchors(split.normal_timestamps, context) & ~split.is_val)
    return spread_blocks(rows, horizon, n_rows)


def runs_of(rows):
    """Number of runs of consecutive rows."""
    return int(np.r_[True, np.diff(rows) != 1].sum()) if len(rows) else 0


def lockstep_steps(rows, horizon):
    """Forward passes a fallback run over `rows` makes: the most origins in any one run."""
    _, lead = origins(rows, horizon)
    run = np.cumsum(np.r_[True, np.diff(rows) != 1]) - 1
    return int(np.bincount(run[lead == 0]).max())


def fallback_counts(n, rows, fallback):
    """Sensors in fallback at each of n rows; -1 where the row was not scored."""
    counts = np.full(n, -1, np.int8)
    counts[rows] = fallback.sum(axis=1)
    return counts


def thresholds_for(split, forecaster, mode, context, horizon, out, deadline):
    """Load thresholds.json, or make it from plain forecasts on training rows."""
    path = out / "thresholds.json"
    sample = train_sample(split, context, horizon)
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved["columns"] != list(split.columns) or saved["percentile"] != FF_PERCENTILE
                or saved["floor"] != MIN_THRESHOLD or saved["train_rows"] != len(sample)):
            raise ValueError(f"{path} was made differently; delete {out} to start this setting again")
        print(f"  1. fallback thresholds: loaded {path.name}")
        return np.array(saved["thresholds"])
    print(f"  1. fallback thresholds: plain forecasts on {len(sample):,} training rows "
          f"({n_origins(sample, horizon):,} forecasts)")
    det = ForecastDetector(forecaster, context, horizon, joint=(mode == "joint"),
                           chunk_origins=chunk_origins(context), deadline=deadline)
    errors = det.errors(split.normal, sample, cache_dir=out / "cache_train")
    thresholds = fallback_thresholds(errors)
    path.write_text(json.dumps({
        "columns": list(split.columns), "thresholds": thresholds.tolist(),
        "percentile": FF_PERCENTILE, "floor": MIN_THRESHOLD,
        "train_rows": len(sample), "train_forecasts": n_origins(sample, horizon),
        "share_of_train_errors_above": float((errors > thresholds).mean())}, indent=2))
    return thresholds


# --------------------------------------------------------------------- full runs
def run_setting(split, raw_labels, forecaster, mode, context, horizon, deadline):
    name = setting_name(mode, context, horizon, forecaster.dtype)
    out = OUT_DIR / name
    if (out / "summary.json").exists():
        print(f"\n{name}: already done, skipped")
        return
    if deadline is not None and time.time() >= deadline:
        raise TimeLimitReached(f"time limit reached before starting {name}")
    out.mkdir(parents=True, exist_ok=True)
    val_rows, test_rows = scored_rows(split, context)
    print(f"\n{name} with Forecast Fallback: {len(val_rows):,} validation rows "
          f"({runs_of(val_rows)} runs, {lockstep_steps(val_rows, horizon):,} steps), "
          f"{len(test_rows):,} test rows ({runs_of(test_rows)} runs, {lockstep_steps(test_rows, horizon):,} steps)")

    start = time.perf_counter()
    thresholds = thresholds_for(split, forecaster, mode, context, horizon, out, deadline)
    det = ForecastFallbackDetector(forecaster, context, thresholds, horizon, joint=(mode == "joint"),
                                   deadline=deadline)
    print("  2. validation rows (Normal file), with fallback")
    val_pred, val_fb = det.run(split.normal, val_rows, cache_dir=out / "state_val")
    print("  3. test rows (Attack file), with fallback")
    test_pred, test_fb = det.run(split.test, test_rows, cache_dir=out / "state_test")
    seconds = time.perf_counter() - start

    val = np.full(len(split.normal), np.nan)
    val[val_rows] = np.abs(val_pred - split.normal[val_rows]).mean(axis=1)
    test = np.full(len(split.test), np.nan)
    test[test_rows] = np.abs(test_pred - split.test[test_rows]).mean(axis=1)
    np.save(out / "val_scores.npy", val)
    np.save(out / "test_scores.npy", test)
    np.save(out / "val_fallback.npy", fallback_counts(len(split.normal), val_rows, val_fb))
    np.save(out / "test_fallback.npy", fallback_counts(len(split.test), test_rows, test_fb))

    attack = split.test_is_attack[test_rows]
    summary = {"setting": name, "mode": mode, "context": context, "horizon": horizon,
               **det.config(), "fallback_percentile": FF_PERCENTILE,
               "fallback_train_rows": json.loads((out / "thresholds.json").read_text())["train_rows"],
               "batch_series": forecaster.batch_size,
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "val_rows": int(len(val_rows)), "test_rows": int(len(test_rows)),
               "minutes_this_run": seconds / 60,
               "val_score_p99_9": float(np.percentile(val[val_rows], 99.9)),
               "val_score_max": float(val[val_rows].max()),
               # share of (row, sensor) pairs in fallback, and of rows with any sensor in fallback
               "fallback_share": {"val": float(val_fb.mean()), "test_normal": float(test_fb[~attack].mean()),
                                  "test_attack": float(test_fb[attack].mean())},
               "rows_with_fallback": {"val": float(val_fb.any(1).mean()),
                                      "test_normal": float(test_fb[~attack].any(1).mean()),
                                      "test_attack": float(test_fb[attack].any(1).mean())},
               "sanity": sanity(test, test_rows, split.test_is_attack, raw_labels)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    s, r = summary["sanity"], summary["rows_with_fallback"]
    print(f"  done in {seconds / 60:.1f} min | sanity ROC-AUC {s['roc_auc']:.3f} "
          f"(original labels {s['roc_auc_original_labels']:.3f}) | mean score attack "
          f"{s['mean_score_attack_rows']:.3f} vs normal {s['mean_score_normal_rows']:.3f} | rows with a "
          f"sensor in fallback: validation {r['val']:.1%}, test normal {r['test_normal']:.1%}, "
          f"test attack {r['test_attack']:.1%}")


# ------------------------------------------------------------------------ smoke
def first_rows_of_each_run(rows, n):
    """The first n rows of every run of consecutive rows."""
    starts_run = np.r_[True, np.diff(rows) != 1]
    run_start = np.maximum.accumulate(np.where(starts_run, np.arange(len(rows)), 0))
    return rows[np.arange(len(rows)) - run_start < n]


def clock(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def run_smoke(split, args):
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    n = args.smoke_steps
    report = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "batch_series": args.batch_series, "dtype": args.dtype, "settings": []}
    forecaster = Chronos2Forecaster(batch_size=args.batch_series, dtype=args.dtype)
    print(f"\nEvery setting: thresholds from {args.smoke_train_rows} training rows (the full run uses "
          f"{TRAIN_ROWS:,}), {2 * n} fallback steps on two test runs, {args.smoke_val_steps} steps on "
          "every validation run")
    for mode in args.modes:
        for context, horizon in parse_settings(args.settings):
            joint = mode == "joint"
            plain = ForecastDetector(forecaster, context, horizon, joint=joint, verbose=False)
            sample = train_sample(split, context, horizon, args.smoke_train_rows)
            plain.errors(split.normal, sample[:horizon])   # warm-up
            errors, sec_train = clock(lambda: plain.errors(split.normal, sample))
            ff = ForecastFallbackDetector(forecaster, context, fallback_thresholds(errors), horizon,
                                          joint=joint, verbose=False)

            val_all, test_all = scored_rows(split, context)
            _, around_onset, onset = smoke_rows(split, context, horizon, n)
            file_end = test_all[-len(around_onset):]   # the second run of the Attack file
            if np.any(np.diff(file_end) != 1) or file_end[0] - context <= around_onset[-1]:
                raise RuntimeError("smoke rows at the end of the Attack file are not a separate clean run")
            two_runs = np.r_[around_onset, file_end]
            (pred, fb), sec_test = clock(lambda: ff.run(split.test, two_runs))
            val_part = first_rows_of_each_run(val_all, args.smoke_val_steps * horizon)
            _, sec_val = clock(lambda: ff.run(split.normal, val_part))

            per_forecast = sec_train / n_origins(sample, horizon)
            per_step_test = sec_test / lockstep_steps(two_runs, horizon)
            per_step_val = sec_val / lockstep_steps(val_part, horizon)
            parts_h = {"thresholds": n_origins(train_sample(split, context, horizon), horizon) * per_forecast / 3600,
                       "validation": lockstep_steps(val_all, horizon) * per_step_val / 3600,
                       "test": lockstep_steps(test_all, horizon) * per_step_test / 3600}

            # around the onset of the longest attack: fallback on, score kept up?
            k = len(around_onset)
            before = around_onset < onset
            ff_score = np.abs(pred[:k] - split.test[around_onset]).mean(axis=1)
            plain_score = plain.score(split.test, around_onset)
            alone_pred, alone_fb = ff.run(split.test, around_onset)   # one run on its own
            row = {"setting": setting_name(mode, context, horizon, args.dtype),
                   "ms_per_plain_forecast": 1000 * per_forecast,
                   "ms_per_step_test": 1000 * per_step_test, "ms_per_step_val": 1000 * per_step_val,
                   "hours": {**parts_h, "total": sum(parts_h.values())},
                   "fallback_share_before_onset": float(fb[:k][before].mean()),
                   "fallback_share_after_onset": float(fb[:k][~before].mean()),
                   "score_after_onset_plain": float(plain_score[~before].mean()),
                   "score_after_onset_ff": float(ff_score[~before].mean()),
                   "score_before_onset_plain": float(plain_score[before].mean()),
                   "score_before_onset_ff": float(ff_score[before].mean()),
                   "two_runs_vs_one_max_forecast_diff": float(np.abs(alone_pred - pred[:k]).max()),
                   "two_runs_vs_one_fallback_agreement": float((alone_fb == fb[:k]).mean()),
                   "finite": bool(np.isfinite(pred).all())}
            report["settings"].append(row)
            print(f"  {row['setting']:24s} full run ~{row['hours']['total']:6.2f} h (thresholds "
                  f"{parts_h['thresholds']:.2f}, validation {parts_h['validation']:.2f}, test {parts_h['test']:.2f}) | "
                  f"after onset: fallback {row['fallback_share_after_onset']:.0%} of sensors "
                  f"(before {row['fallback_share_before_onset']:.0%}), score plain "
                  f"{row['score_after_onset_plain']:.3f} -> fallback {row['score_after_onset_ff']:.3f} | "
                  f"one run vs two: max diff {row['two_runs_vs_one_max_forecast_diff']:.1e}")

    path = SMOKE_DIR / "ff_smoke_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nSmoke report saved to {path.relative_to(REPO_ROOT)}")


# ------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", nargs="+", default=["all"],
                        help="lowe, ours, all, and/or CONTEXTxHORIZON items such as 128x32")
    parser.add_argument("--modes", nargs="+", choices=["joint", "alone"], default=["joint"])
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--batch-series", type=int, default=144,
                        help="series per forward pass (36 per origin in joint mode)")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop cleanly after this many hours (rerun the same command to continue)")
    parser.add_argument("--stop-at", default=None,
                        help="stop cleanly at this clock time, HH:MM, e.g. 08:00 (the next time it comes round)")
    parser.add_argument("--smoke", action="store_true", help="quick timing and sanity run, see above")
    parser.add_argument("--smoke-steps", type=int, default=12,
                        help="fallback steps on EACH side of the attack onset per setting in --smoke")
    parser.add_argument("--smoke-val-steps", type=int, default=4,
                        help="fallback steps on every validation run per setting in --smoke")
    parser.add_argument("--smoke-train-rows", type=int, default=480,
                        help="training rows for the quick thresholds in --smoke")
    args = parser.parse_args()

    log_dir = SMOKE_DIR if args.smoke else OUT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(log_dir / ("ff_smoke_log.txt" if args.smoke else "run_log.txt"))
    print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv[1:]) or '(defaults)'}")

    print(f"Loading and preprocessing SWaT from {SWAT_DIR} (about 2 minutes)...")
    attack_df = load_swat_attack(SWAT_DIR)
    split = prepare_swat(load_swat_normal(SWAT_DIR), attack_df)
    raw_labels = is_attack_mask(attack_df).to_numpy()
    print(f"  {split.normal.shape[0]:,} Normal rows ({split.is_val.sum():,} validation), "
          f"{split.test.shape[0]:,} Attack rows, {len(split.columns)} columns; "
          f"fallback: {FF_PERCENTILE}th percentile, {FF_TRIGGER} rows in a row")

    if args.smoke:
        run_smoke(split, args)
        return
    deadline = stop_time(args.max_hours, args.stop_at)
    if deadline is not None:
        print(f"Will stop cleanly by about {time.strftime('%Y-%m-%d %H:%M', time.localtime(deadline))} "
              "(the fallback state is saved at the limit)")
    forecaster = Chronos2Forecaster(batch_size=args.batch_series, dtype=args.dtype)
    # cheapest settings first, so early results arrive quickly
    todo = sorted(parse_settings(args.settings), key=lambda ch: (ch[0] / 16 + 2) / ch[1])
    try:
        for mode in args.modes:
            for context, horizon in todo:
                run_setting(split, raw_labels, forecaster, mode, context, horizon, deadline)
    except TimeLimitReached as stop:
        print(f"\nSTOPPED: {stop}. Everything finished so far is saved.\n"
              f"To continue later, run exactly the same command again:\n"
              f"    python experiments/tsfm/run_swat_chronos2_ff.py {' '.join(sys.argv[1:])}")
        return
    print("\nAll requested settings done.")


if __name__ == "__main__":
    main()
