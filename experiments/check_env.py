"""Phase B Step B1: check that a machine is ready to run the experiments.

Run this first on any new machine (the lab PC after `git pull`, a Colab
session, the laptop after an upgrade). It checks, in order:

  1. code version   which git commit is checked out, and whether every
                    package pinned in requirements.txt is installed at the
                    pinned version
  2. GPU            name, memory, and a short matrix-multiply benchmark in
                    float32 / float16 / bfloat16, so the faster precision is
                    measured on THIS card rather than assumed (on the
                    laptop's GTX 1660 Ti float16 is about 7x SLOWER than
                    float32; on an RTX 20-series or newer it is usually
                    2-4x faster)
  3. repo package   `ics_tsfm` imports, and from this clone
  4. SWaT data      the files are where SWAT_DIR (or the default folder
                    inside the repo) says
  5. model cache    where Hugging Face stores downloaded models, and the
                    free disk space there
  6. Chronos-2      downloads the pinned checkpoint (about 480 MB, once),
                    checks its weights hash, and forecasts a small synthetic
                    signal, all sensors together and one sensor at a time;
                    both must beat "repeat the last value"

Each check prints PASS, WARN or FAIL. The script exits with code 1 if any
check fails. A JSON copy of the report goes to results/env_check/<host>.json
(git-ignored).

The latency printed in check 6 is only a sign of life. Official cost numbers
come from the cost harness (Step B9), measured on one fixed device.

Run from the repo root:

    python experiments/check_env.py
    python experiments/check_env.py --skip-model       # offline: skip check 6
    python experiments/check_env.py --skip-benchmark   # skip the precision test
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SWAT_DIR = Path(os.environ.get("SWAT_DIR", REPO_ROOT / "$$SWAT Dataset A2-Dec 2015"))
SWAT_FILES = ("SWaT_Dataset_Normal_v1.xlsx", "SWaT_Dataset_Attack_v0.xlsx",
              "List_of_attacks_Final(Sheet1).csv")
# The Chronos-2 that Lowe et al. (ICML 2026 FMSD workshop) evaluate. The paper
# names no size or checkpoint; "Chronos-2" is Amazon's 120M flagship (the only
# other sizes are named chronos-2-small and chronos-2-synth). Its weights were
# uploaded 2025-10-30 and never changed (later commits touch only the README),
# so the paper used these same weights; autogluon/chronos-2 is a byte-identical
# mirror. Pinning the commit and the weight hash keeps every machine on them.
CHRONOS2_ID = "amazon/chronos-2"
CHRONOS2_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
CHRONOS2_SHA256 = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
# Installed with --no-deps (see requirements.txt), so it is a comment there
# and has to be checked by hand here.
EXTRA_PINS = {"TSB-AD": "1.5"}
# Model packages added in later steps; reported, never required yet.
LATER_PACKAGES = {"uni2ts": "B7 (Moirai-2.0)", "granite-tsfm": "B8 (TSPulse)"}

report = []


def record(check, status, message, **details):
    report.append({"check": check, "status": status, "message": message, **details})
    print(f"  [{status}] {message}")


# ------------------------------------------------------------------ checks
def check_code_version():
    print("\n1. Code version and packages")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        record("git", "PASS" if not dirty else "WARN",
               f"commit {commit}" + (" (with uncommitted changes)" if dirty else ""), commit=commit)
    except (OSError, subprocess.CalledProcessError):
        record("git", "WARN", "could not read the git commit")

    pins = dict(EXTRA_PINS)
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)", line.strip())
        if m:
            pins[m.group(1)] = m.group(2)
    missing, mismatched = [], []
    for name, pinned in pins.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if installed != pinned:
            mismatched.append(f"{name} {installed} (pinned {pinned})")
    if missing:
        record("packages", "FAIL", "missing: " + ", ".join(missing))
    if mismatched:
        record("packages", "WARN", "different version: " + "; ".join(mismatched))
    if not missing and not mismatched:
        record("packages", "PASS", f"all {len(pins)} pinned packages at the pinned version")

    import torch
    record("torch", "PASS", f"torch {torch.__version__}, CUDA build {torch.version.cuda}, "
                            f"Python {platform.python_version()}")
    for name, step in LATER_PACKAGES.items():
        try:
            print(f"  [INFO] {name} {metadata.version(name)} installed (needed from {step})")
        except metadata.PackageNotFoundError:
            print(f"  [INFO] {name} not installed yet (added in {step})")


def matmul_tflops(dtype, n=4096, reps=20):
    import torch
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    return 2 * n ** 3 * reps / (time.perf_counter() - start) / 1e12


def check_gpu(benchmark):
    print("\n2. GPU")
    import torch
    if not torch.cuda.is_available():
        record("gpu", "FAIL", "no CUDA GPU visible to torch (check the driver and the torch CUDA build)")
        return
    props = torch.cuda.get_device_properties(0)
    record("gpu", "PASS", f"{props.name}, {props.total_memory / 2**30:.1f} GB, "
                          f"compute capability {props.major}.{props.minor}",
           name=props.name, memory_gb=round(props.total_memory / 2**30, 1),
           capability=f"{props.major}.{props.minor}")
    if not benchmark:
        return
    speeds = {}
    for label, dtype in (("float32", torch.float32), ("float16", torch.float16),
                         ("bfloat16", torch.bfloat16)):
        try:
            speeds[label] = matmul_tflops(dtype)
        except RuntimeError as err:
            print(f"  [INFO] {label}: not usable ({err.__class__.__name__})")
    torch.cuda.empty_cache()
    for label, tflops in speeds.items():
        print(f"  [INFO] {label:8s} matmul {tflops:6.2f} TFLOPS ({tflops / speeds['float32']:.1f}x float32)")
    best = max(speeds, key=speeds.get)
    if best != "float32" and speeds[best] > 1.2 * speeds["float32"]:
        msg = f"{best} is {speeds[best] / speeds['float32']:.1f}x faster than float32 on this card"
    else:
        best, msg = "float32", "float32 is the fastest precision on this card; keep it"
    record("precision", "PASS", msg, tflops={k: round(v, 2) for k, v in speeds.items()}, fastest=best)


def check_repo_package():
    print("\n3. Repo package")
    try:
        import ics_tsfm
    except ImportError:
        record("ics_tsfm", "FAIL", "ics_tsfm not importable; run `pip install -e . --no-deps` in the repo")
        return
    path = Path(ics_tsfm.__file__).resolve()
    if REPO_ROOT in path.parents:
        record("ics_tsfm", "PASS", f"imported from this clone ({path.parent})")
    else:
        record("ics_tsfm", "WARN", f"imported from {path.parent}, not from this clone; "
                                   "rerun `pip install -e . --no-deps` here")


def check_data():
    print("\n4. SWaT data")
    source = "SWAT_DIR" if "SWAT_DIR" in os.environ else "default folder"
    missing = [f for f in SWAT_FILES if not (SWAT_DIR / f).is_file()]
    if missing:
        record("swat", "FAIL", f"missing in {SWAT_DIR} ({source}): {', '.join(missing)}")
    else:
        size = sum((SWAT_DIR / f).stat().st_size for f in SWAT_FILES) / 2**20
        record("swat", "PASS", f"all {len(SWAT_FILES)} files in {SWAT_DIR} ({source}), {size:.0f} MB")


def check_cache():
    print("\n5. Model cache")
    from huggingface_hub import constants
    cache = Path(constants.HF_HUB_CACHE)
    probe = cache if cache.exists() else next(p for p in cache.parents if p.exists())
    free_gb = shutil.disk_usage(probe).free / 2**30
    source = "HF_HOME" if "HF_HOME" in os.environ else "default"
    status = "PASS" if free_gb > 5 else "WARN"
    record("hf_cache", status, f"{cache} ({source}), {free_gb:.0f} GB free"
                               + ("" if status == "PASS" else "; set HF_HOME to a bigger drive"),
           path=str(cache), free_gb=round(free_gb, 1))


def synthetic_signal(context=512, horizon=16, seed=0):
    """Three sensors: two phase-shifted sines and their product, so the third
    depends non-linearly on the other two. Returns (context, future) arrays
    of shape (3, context) and (3, horizon)."""
    rng = np.random.default_rng(seed)
    t = np.arange(context + horizon)
    a = np.sin(2 * np.pi * t / 64)
    b = np.sin(2 * np.pi * t / 64 + 1.0)
    x = np.stack([a, b, a * b]) + 0.05 * rng.standard_normal((3, t.size))
    return x[:, :context].astype(np.float32), x[:, context:].astype(np.float32)


def check_chronos(model_id, device):
    print("\n6. Chronos-2")
    import torch
    from chronos import Chronos2Pipeline
    from huggingface_hub import hf_hub_download

    start = time.perf_counter()
    try:
        pipe = Chronos2Pipeline.from_pretrained(model_id, revision=CHRONOS2_REVISION, device_map=device)
        weights = Path(hf_hub_download(model_id, "model.safetensors", revision=CHRONOS2_REVISION))
    except Exception as err:  # network, auth, disk: report it, don't crash the report
        record("chronos2_load", "FAIL", f"could not load {model_id}: {err}")
        return
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest == CHRONOS2_SHA256:
        record("chronos2_weights", "PASS", f"weights match the pinned checkpoint "
                                           f"(revision {CHRONOS2_REVISION[:10]}, sha256 {digest[:12]})")
    else:
        record("chronos2_weights", "FAIL", f"weights hash {digest[:12]} != pinned {CHRONOS2_SHA256[:12]}")
    n_params = sum(p.numel() for p in pipe.model.parameters())
    record("chronos2_load", "PASS",
           f"{model_id} loaded in {time.perf_counter() - start:.0f} s: {n_params / 1e6:.1f}M parameters, "
           f"{pipe.model.dtype}, {pipe.model.device}, context up to {pipe.model_context_length}, "
           f"output patch {pipe.model_output_patch_size}",
           params_m=round(n_params / 1e6, 1), context_length=pipe.model_context_length)

    context, future = synthetic_signal()
    horizon = future.shape[1]
    median = pipe.quantiles.index(0.5)
    # One row with 3 sensors = forecast them together (group attention);
    # three rows with 1 sensor each = forecast each on its own.
    joint = pipe.predict(context[None], prediction_length=horizon)[0][:, median].cpu().numpy()
    alone = np.concatenate([p[:, median].cpu().numpy()
                            for p in pipe.predict(context[:, None, :], prediction_length=horizon)])
    if joint.shape != future.shape or alone.shape != future.shape:
        record("chronos2_forecast", "FAIL", f"unexpected output shapes {joint.shape}, {alone.shape}")
        return
    naive = np.abs(future - context[:, -1:]).mean()
    mae_joint, mae_alone = np.abs(future - joint).mean(), np.abs(future - alone).mean()
    ok = mae_joint < naive and mae_alone < naive
    record("chronos2_forecast", "PASS" if ok else "FAIL",
           f"forecast error (MAE) together {mae_joint:.3f}, one at a time {mae_alone:.3f}, "
           f"repeat-last-value {naive:.3f}",
           mae_joint=float(mae_joint), mae_alone=float(mae_alone), mae_naive=float(naive))

    times = []
    for _ in range(12):
        if pipe.model.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipe.predict(context[None], prediction_length=horizon)
        if pipe.model.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print(f"  [INFO] one 3-sensor forecast (context 512, horizon 16): "
          f"median {1000 * np.median(times[2:]):.0f} ms (sign of life only, not a cost number)")


# -------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-model", action="store_true", help="skip the Chronos-2 download and forecast")
    parser.add_argument("--skip-benchmark", action="store_true", help="skip the precision benchmark")
    parser.add_argument("--model-id", default=CHRONOS2_ID)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Environment check on {platform.node()} ({platform.system()} {platform.release()})")

    check_code_version()
    check_gpu(benchmark=not args.skip_benchmark)
    check_repo_package()
    check_data()
    check_cache()
    if args.skip_model:
        print("\n6. Chronos-2: skipped (--skip-model)")
    else:
        check_chronos(args.model_id, device)

    failed = [r["check"] for r in report if r["status"] == "FAIL"]
    warned = [r["check"] for r in report if r["status"] == "WARN"]
    print("\nSummary: " + ("FAIL in " + ", ".join(failed) if failed else "no failures")
          + (f"; warnings in {', '.join(warned)}" if warned else ""))

    out = REPO_ROOT / "results" / "env_check" / f"{platform.node()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"host": platform.node(), "system": platform.platform(),
                               "time": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": report}, indent=2))
    print(f"Report saved to {out.relative_to(REPO_ROOT)}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
