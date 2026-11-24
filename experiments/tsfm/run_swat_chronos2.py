"""Phase B Step B4: score SWaT with zero-shot Chronos-2, no Forecast Fallback.

Settings (decided 2026-11-23, before any test result)
-----------------------------------------------------
  lowe   Lowe et al.'s grid, context 32/64/128 x horizon 1/16/32, so our
         numbers can be set next to their Table 2 (reproduction).
  ours   context 4096 rows (about one 70-minute plant cycle, measured on the
         Normal file) and horizon 1 (every second scored as it arrives, like
         the one-step CNN). Our headline setting.
The whole grid is reported; no setting is picked from test results.

Modes
-----
  joint  all 36 sensors in one forecast (Chronos-2's group attention)
  alone  every sensor forecast on its own (Direction 2 ablation)

What gets scored
----------------
  Normal file, validation rows: the detection threshold comes from these
  (Step B6). Training rows are not needed yet (the Forecast Fallback
  thresholds of Step B5 will use them).
  Attack file: every row whose context has no recording gap.
Preprocessing is the Phase A pipeline (prepare_swat, seed 0, 36 columns),
run at the start; nothing preprocessed is read from disk.

What it writes (results/ is git-ignored)
----------------------------------------
  results/tsfm/chronos2/<mode>_I<context>_H<horizon>_<dtype>/
    val_scores.npy    one score per Normal-file row (NaN = not scored)
    test_scores.npy   one score per Attack-file row (NaN = not scored)
    cache_val/, cache_test/   forecast chunks; an interrupted run resumes here
    summary.json      rows scored, time taken, sanity numbers
  results/tsfm/chronos2/run_log.txt   everything printed
A setting whose summary.json exists is skipped, so rerunning the same
command after a shutdown only does the unfinished work.

Running in sessions (shared lab PC)
-----------------------------------
  --max-hours 12 or --stop-at 08:00 stops cleanly: no new chunk is started
  after the limit, everything finished is saved, and the same command run
  later continues where it stopped. Chunks are about 4 minutes of work at
  context 4096, so the run ends at most a few minutes after the limit, and
  even a hard stop (Ctrl+C, shutdown) loses at most one chunk.

The sanity numbers (ROC-AUC, attack vs normal score) are NOT the official
evaluation; that is Step B6, with the same detection rule and metrics as
PCA and the CNN. ROC-AUC is also given against the dataset's ORIGINAL labels,
which is what Lowe et al. most likely used.

Smoke test (run this first on a new machine)
--------------------------------------------
  --smoke scores a few dozen forecasts per setting around the onset of the
  longest attack, then prints: time per forecast and the estimated time of
  the full run for every setting, a batch-size sweep (fastest batch on this
  GPU), and float32 vs bfloat16 (speed and how much the scores move). Writes
  results/tsfm_smoke/smoke_report.json. About 5-15 minutes.

Run from the repo root:

    python experiments/tsfm/run_swat_chronos2.py --smoke
    python experiments/tsfm/run_swat_chronos2.py                       # everything, joint mode
    python experiments/tsfm/run_swat_chronos2.py --settings lowe
    python experiments/tsfm/run_swat_chronos2.py --settings 128x1 1024x1      # single settings
    python experiments/tsfm/run_swat_chronos2.py --settings ours --modes joint alone
    python experiments/tsfm/run_swat_chronos2.py --dtype bfloat16 --batch-series 576
    python experiments/tsfm/run_swat_chronos2.py --settings all --max-hours 12   # a 12-hour session
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ics_tsfm.data.swat import is_attack_mask, load_swat_attack, load_swat_normal
from ics_tsfm.data.swat_preprocess import prepare_swat
from ics_tsfm.data.windows import contiguous_anchors
from ics_tsfm.models.tsfm import ForecastDetector, TimeLimitReached, origins
from ics_tsfm.models.tsfm.chronos2 import Chronos2Forecaster

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "codebase"))
from swat_cycle_analysis import SWAT_DIR  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "tsfm" / "chronos2"
SMOKE_DIR = REPO_ROOT / "results" / "tsfm_smoke"
LOWE = [(c, h) for c in (32, 64, 128) for h in (1, 16, 32)]
OURS = [(4096, 1)]
SETTINGS = {"lowe": LOWE, "ours": OURS, "all": LOWE + OURS}


def parse_settings(items):
    """Names from SETTINGS and/or single settings written context x horizon,
    e.g. ["lowe", "1024x1"]. Duplicates are dropped, order kept."""
    out = []
    for item in items:
        if item in SETTINGS:
            out += SETTINGS[item]
        else:
            try:
                context, horizon = (int(v) for v in item.lower().split("x"))
            except ValueError:
                raise SystemExit(f"--settings: {item!r} is neither {list(SETTINGS)} nor CONTEXTxHORIZON")
            out.append((context, horizon))
    return list(dict.fromkeys(out))


class Tee:
    """Print to the screen and to a log file at the same time."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()


def scored_rows(split, context):
    """(validation rows of the Normal file, Attack-file rows) with a clean context."""
    val = np.flatnonzero(contiguous_anchors(split.normal_timestamps, context) & split.is_val)
    test = np.flatnonzero(contiguous_anchors(split.test_timestamps, context))
    return val, test


def n_origins(rows, horizon):
    return int((origins(rows, horizon)[1] == 0).sum())


def setting_name(mode, context, horizon, dtype):
    return f"{mode}_I{context}_H{horizon}_{dtype}"


def sanity(scores, rows, labels, raw_labels):
    s, y, y_raw = scores[rows], labels[rows], raw_labels[rows]
    return {"roc_auc": float(roc_auc_score(y, s)),
            "roc_auc_original_labels": float(roc_auc_score(y_raw, s)),
            "mean_score_attack_rows": float(s[y].mean()),
            "mean_score_normal_rows": float(s[~y].mean())}


# --------------------------------------------------------------------- full runs
def chunk_origins(context):
    """Forecasts per saved chunk: about 4 minutes of work on the RTX 5080 at
    context 4096, so a stop (time limit or Ctrl+C) loses at most that much.
    Short contexts keep the original 8192 so runs started before this rule
    resume with their existing chunks."""
    return 8192 if context < 1024 else 2048


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
    det = ForecastDetector(forecaster, context, horizon, joint=(mode == "joint"),
                           chunk_origins=chunk_origins(context), deadline=deadline)
    print(f"\n{name}: {len(val_rows):,} validation + {len(test_rows):,} test rows, "
          f"{n_origins(val_rows, horizon) + n_origins(test_rows, horizon):,} forecasts")

    start = time.perf_counter()
    print("  validation rows (Normal file)")
    val = det.score_series(split.normal, val_rows, cache_dir=out / "cache_val")
    print("  test rows (Attack file)")
    test = det.score_series(split.test, test_rows, cache_dir=out / "cache_test")
    seconds = time.perf_counter() - start
    np.save(out / "val_scores.npy", val)
    np.save(out / "test_scores.npy", test)

    summary = {"setting": name, "mode": mode, "context": context, "horizon": horizon,
               **det.config(), "batch_series": forecaster.batch_size,
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "val_rows": int(len(val_rows)), "test_rows": int(len(test_rows)),
               "minutes_this_run": seconds / 60,
               "val_score_p99_9": float(np.percentile(val[val_rows], 99.9)),
               "val_score_max": float(val[val_rows].max()),
               "sanity": sanity(test, test_rows, split.test_is_attack, raw_labels)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    s = summary["sanity"]
    print(f"  done in {seconds / 60:.1f} min | sanity ROC-AUC {s['roc_auc']:.3f} "
          f"(original labels {s['roc_auc_original_labels']:.3f}) | mean score attack "
          f"{s['mean_score_attack_rows']:.3f} vs normal {s['mean_score_normal_rows']:.3f}")


# ------------------------------------------------------------------------ smoke
def smoke_rows(split, context, horizon, n):
    """n origins on each side of the longest attack's onset (test file), and
    n origins from the first validation stretch (Normal file)."""
    y = split.test_is_attack
    starts = np.flatnonzero(np.diff(y.astype(int)) == 1) + 1
    ends = np.flatnonzero(np.diff(y.astype(int)) == -1) + 1
    onset = starts[np.argmax(ends - starts)]
    val_all, test_all = scored_rows(split, context)
    test = np.arange(onset - n * horizon, onset + n * horizon)
    if not np.isin(test, test_all).all():
        raise RuntimeError("smoke rows around the attack onset lack a clean context")
    return val_all[:2 * n * horizon], test, onset


def timed(det, x, rows):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    scores = det.score(x, rows)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return scores, time.perf_counter() - start


def run_smoke(split, args):
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    n = args.smoke_origins
    report = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "batch_series": args.batch_series, "dtype": args.dtype, "settings": [],
              "batch_sweep": [], "precision": []}
    forecaster = Chronos2Forecaster(batch_size=args.batch_series, dtype=args.dtype)

    print(f"\n1. Every setting: {2 * n} forecasts each around the onset of the longest attack")
    for mode in args.modes:
        for context, horizon in parse_settings(args.settings):
            val_rows, test_rows, onset = smoke_rows(split, context, horizon, n)
            det = ForecastDetector(forecaster, context, horizon, joint=(mode == "joint"), verbose=False)
            det.score(split.test, test_rows[:horizon])   # warm-up
            scores, sec = timed(det, split.test, test_rows)
            val_scores = det.score(split.normal, val_rows)   # the Normal-file path works too
            per_forecast = sec / (2 * n)
            full_val, full_test = scored_rows(split, context)
            full = n_origins(full_val, horizon) + n_origins(full_test, horizon)
            before, after = scores[test_rows < onset].mean(), scores[test_rows >= onset].mean()
            row = {"setting": setting_name(mode, context, horizon, args.dtype),
                   "ms_per_forecast": 1000 * per_forecast, "forecasts_full_run": full,
                   "hours_full_run": full * per_forecast / 3600,
                   "mean_score_before_onset": float(before), "mean_score_after_onset": float(after),
                   "mean_score_validation": float(val_scores.mean()),
                   "finite": bool(np.isfinite(scores).all() and np.isfinite(val_scores).all())}
            report["settings"].append(row)
            print(f"  {row['setting']:24s} {row['ms_per_forecast']:8.1f} ms/forecast -> full run "
                  f"{row['hours_full_run']:6.2f} h ({full:,} forecasts) | score before onset "
                  f"{before:.3f}, after {after:.3f}")

    batches = (72, 144, 288, 576, 1152, 2304)
    # enough forecasts to fill the largest batch three times (2304 series = 64 origins)
    n_sweep = max(n, 3 * max(batches) // 36 // 2)
    print(f"\n2. Batch-size sweep (series per forward pass), joint mode, horizon 1, {2 * n_sweep} forecasts")
    for context in (128, 4096):
        _, test_rows, _ = smoke_rows(split, context, 1, n_sweep)
        for batch in batches:
            forecaster.batch_size = batch
            det = ForecastDetector(forecaster, context, 1, verbose=False)
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                det.score(split.test, test_rows[:2])
                _, sec = timed(det, split.test, test_rows)
                peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                row = {"context": context, "batch_series": batch,
                       "forecasts_per_s": 2 * n_sweep / sec, "peak_gb": peak}
                print(f"  context {context:5d}, batch {batch:5d}: {row['forecasts_per_s']:7.1f} "
                      f"forecasts/s, peak GPU memory {peak:.1f} GB")
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = {"context": context, "batch_series": batch, "forecasts_per_s": None}
                print(f"  context {context:5d}, batch {batch:5d}: out of GPU memory")
            report["batch_sweep"].append(row)
    forecaster.batch_size = args.batch_series

    print("\n3. float32 vs bfloat16 (same rows, joint mode, horizon 1)")
    for context in (128, 4096):
        _, test_rows, _ = smoke_rows(split, context, 1, n)
        result = {}
        for dtype in ("float32", "bfloat16"):
            forecaster.pipeline.model.to(getattr(torch, dtype))
            det = ForecastDetector(forecaster, context, 1, verbose=False)
            det.score(split.test, test_rows[:2])
            result[dtype] = timed(det, split.test, test_rows)
        (s32, t32), (s16, t16) = result["float32"], result["bfloat16"]
        diff = np.abs(s32 - s16)
        row = {"context": context, "speedup": t32 / t16, "max_abs_score_diff": float(diff.max()),
               "median_score": float(np.median(s32)), "max_diff_over_median_score": float(diff.max() / np.median(s32))}
        report["precision"].append(row)
        print(f"  context {context:5d}: bfloat16 is {row['speedup']:.1f}x the speed of float32; "
              f"scores move by at most {row['max_abs_score_diff']:.4f} "
              f"(median score {row['median_score']:.3f})")
    forecaster.pipeline.model.to(getattr(torch, args.dtype))

    path = SMOKE_DIR / "smoke_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nSmoke report saved to {path.relative_to(REPO_ROOT)}")


# ------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", nargs="+", default=["all"],
                        help="lowe, ours, all, and/or CONTEXTxHORIZON items such as 1024x1")
    parser.add_argument("--modes", nargs="+", choices=["joint", "alone"], default=["joint"])
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--batch-series", type=int, default=144,
                        help="series per forward pass (36 per origin in joint mode)")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop cleanly after this many hours (rerun the same command to continue)")
    parser.add_argument("--stop-at", default=None,
                        help="stop cleanly at this clock time, HH:MM, e.g. 08:00 (the next time it comes round)")
    parser.add_argument("--smoke", action="store_true", help="quick timing and sanity run, see above")
    parser.add_argument("--smoke-origins", type=int, default=24,
                        help="forecasts on EACH side of the attack onset per setting in --smoke")
    args = parser.parse_args()

    log_dir = SMOKE_DIR if args.smoke else OUT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(log_dir / ("smoke_log.txt" if args.smoke else "run_log.txt"))
    print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv[1:]) or '(defaults)'}")

    print(f"Loading and preprocessing SWaT from {SWAT_DIR} (about 2 minutes)...")
    attack_df = load_swat_attack(SWAT_DIR)
    split = prepare_swat(load_swat_normal(SWAT_DIR), attack_df)
    raw_labels = is_attack_mask(attack_df).to_numpy()
    print(f"  {split.normal.shape[0]:,} Normal rows ({split.is_val.sum():,} validation), "
          f"{split.test.shape[0]:,} Attack rows, {len(split.columns)} columns")

    if args.smoke:
        run_smoke(split, args)
        return
    deadline = stop_time(args.max_hours, args.stop_at)
    if deadline is not None:
        print(f"Will stop cleanly by about {time.strftime('%Y-%m-%d %H:%M', time.localtime(deadline))} "
              "(the chunk in progress finishes first: at most a few minutes)")
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
              f"    python experiments/tsfm/run_swat_chronos2.py {' '.join(sys.argv[1:])}")
        return
    print("\nAll requested settings done.")


def stop_time(max_hours, stop_at):
    """time.time() value at which to stop, from --max-hours and/or --stop-at
    (whichever comes first); None for no limit. Counted from now, i.e. after
    the 2-minute data load."""
    limits = []
    if max_hours is not None:
        limits.append(time.time() + 3600 * max_hours)
    if stop_at is not None:
        try:
            hour, minute = (int(v) for v in stop_at.split(":"))
        except ValueError:
            raise SystemExit(f"--stop-at must look like 08:00, got {stop_at!r}")
        now = time.localtime()
        target = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hour, minute, 0, 0, 0, -1))
        if target <= time.time():
            target += 24 * 3600   # that time has passed today: tomorrow
        limits.append(target)
    return min(limits) if limits else None


if __name__ == "__main__":
    main()
