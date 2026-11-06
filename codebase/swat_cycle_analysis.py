"""How long is one operating cycle of the SWaT plant?

Why this exists
---------------
Step 3 splits the Normal file into contiguous blocks and assigns whole blocks
to training or validation at random. A block should be long enough to hold at
least one full plant cycle; otherwise a block can capture only one phase of the
process (a tank filling but never draining, say). This script measures that
cycle from the data instead of trusting published figures.

Two independent measurements, so each can check the other
---------------------------------------------------------
  A. Switch-on intervals (on/off actuators only). Find every moment a pump or
     valve turns on and measure the time until it next turns on. One gap is
     one cycle. Easy to verify by hand.
  B. Autocorrelation (every column that changes). Compare a signal with a copy
     of itself shifted later in time. The shift at which the two line up best
     is the cycle length. Works for continuous sensors too (levels, flows,
     pressures), where there is no on/off event to count.

Columns that never change are skipped, results are grouped by plant stage (the
first digit of the tag: LIT101 is stage 1, DPIT301 is stage 3), and the slowest
clear cycle sets the lower bound on block size.

Run from the repo root (takes about 2 minutes the first time, while the xlsx
loads; later runs read a cached copy):

    python codebase/swat_cycle_analysis.py

Writes 8 figures and a CSV table to results/swat_cycles/ (git-ignored).
"""
import os
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

from ics_tsfm.data.swat import load_swat_normal, sensor_columns

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SWAT_DIR = Path(os.environ.get("SWAT_DIR", REPO_ROOT / "$$SWAT Dataset A2-Dec 2015"))
OUT_DIR = REPO_ROOT / "results" / "swat_cycles"
CACHE_FILE = OUT_DIR / "normal_cache.parquet"

# The first hours of the Normal file are the plant starting up: tanks are
# still filling, so the signals do not cycle yet. LIT101, LIT301 and LIT401
# all reach their operating band by hour 4; skipping 6 adds a margin. This is
# an analysis-only choice. The startup cutoff for model training is a separate
# Step 3 decision.
SKIP_HOURS = 6

MAX_LAG_MIN = 240        # look for cycles up to 4 hours long
CLEAR_PEAK = 0.3         # autocorrelation peak at least this high = clear cycle
WEAK_PEAK = 0.1          # between WEAK and CLEAR = weak (usually short pulses)
DRIFT_ACF_AT_1H = 0.5    # no peak, but still this similar 1 h later = slow drift
MIN_SWITCH_ONS = 10      # fewer switch-ons than this is too few to call a cycle
BLOCK_CANDIDATES_MIN = (30, 60, 120)

# 3-hour window used for the zoomed-in figures (hours from the file start).
ZOOM_START_HOUR = 30
ZOOM_HOURS = 3

# SWaT actuator encoding: 2 = on/open, 1 = off/closed, 0 = valve moving.
ON = 2

STAGE_NAMES = {
    1: "Raw water intake",
    2: "Chemical dosing",
    3: "Ultrafiltration (UF)",
    4: "Dechlorination",
    5: "Reverse osmosis",
    6: "Backwash and product storage",
}

# Tag prefix -> what kind of instrument it is. Order matters: DPIT and PIT
# must be checked before the bare P that marks a pump.
KINDS = [
    ("LIT", "tank level"),
    ("DPIT", "pressure"),
    ("PIT", "pressure"),
    ("FIT", "flow"),
    ("AIT", "chemical analyzer"),
    ("MV", "valve"),
    ("UV", "UV lamp"),
    ("P", "pump"),
]

# --------------------------------------------------------------------------
# Chart style (light theme; one hue per meaning, recessive grid and axes)
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # the signal being shown
ORANGE = "#eb6834"   # a contrasting event or comparison series


def apply_style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
    })


def new_figure(width, height, title, subtitle, **subplot_kw):
    """Figure with a left-aligned headline (the takeaway) and a subtitle
    (how to read it) in a reserved band above the plots."""
    fig, axes = plt.subplots(figsize=(width, height), layout="constrained", **subplot_kw)
    header_in = 1.05
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - header_in / height))
    fig.text(0.01, 1 - 0.12 / height, title, ha="left", va="top",
             fontsize=13.5, weight="semibold", color=INK)
    fig.text(0.01, 1 - 0.46 / height, subtitle, ha="left", va="top",
             fontsize=10, color=INK2, linespacing=1.4)
    return fig, axes


def block_size_lines(ax, top_labels=True):
    """Dashed reference lines at the candidate block sizes."""
    for b in BLOCK_CANDIDATES_MIN:
        ax.axvline(b, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        if top_labels:
            label = f"{b // 60}-hour block" if b % 60 == 0 and b >= 120 else f"{b}-min block"
            ax.text(b, 1.01, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color=INK2)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------
# Step 1: load the Normal file
# --------------------------------------------------------------------------
def load_normal() -> pd.DataFrame:
    """The xlsx takes over a minute to parse, so the first run caches it as
    parquet. Delete the cache file to force a fresh load."""
    if CACHE_FILE.exists():
        print(f"Reading cached copy {CACHE_FILE.relative_to(REPO_ROOT)}")
        return pd.read_parquet(CACHE_FILE)
    print(f"Loading SWaT Normal file from {SWAT_DIR} (about a minute)...")
    df = load_swat_normal(SWAT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE)
    return df


# --------------------------------------------------------------------------
# Step 2: column bookkeeping
# --------------------------------------------------------------------------
def plant_stage(col: str) -> int:
    """SWaT tags end in a 3-digit number whose first digit is the stage:
    LIT101 -> 1, DPIT301 -> 3, P602 -> 6."""
    return int(re.search(r"(\d)\d\d$", col).group(1))


def instrument_kind(col: str) -> str:
    for prefix, kind in KINDS:
        if col.startswith(prefix):
            return kind
    return "other"


def is_actuator(col: str) -> bool:
    """Pumps (P101), valves (MV101) and the UV lamp (UV401). The pattern needs
    digits right after the letters, so pressure sensors like PIT501 don't match."""
    return re.fullmatch(r"(P|MV|UV)\d{3}", col) is not None


# --------------------------------------------------------------------------
# Step 3: method A, switch-on intervals
# --------------------------------------------------------------------------
def switch_on_rows(values: np.ndarray) -> np.ndarray:
    """Row positions where an actuator turns on: off (or moving) in the row
    before, on in this row.

    Example: values = [1, 1, 2, 2, 1, 2]  ->  switches on at rows [2, 5]
    """
    on = values == ON
    return np.flatnonzero(on[1:] & ~on[:-1]) + 1


def switch_on_intervals_min(values: np.ndarray) -> np.ndarray:
    """Minutes between consecutive switch-ons. Rows are 1 second apart, so a
    gap of 4200 rows is 70 minutes."""
    return np.diff(switch_on_rows(values)) / 60


def on_spans(values: np.ndarray, x: np.ndarray):
    """(start, end) x-positions of every stretch where an actuator is on, for
    shading those stretches in the zoomed-in figures."""
    on = np.r_[False, values == ON, False].astype(int)
    edges = np.diff(on)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    x_ext = np.r_[x, x[-1] + (x[1] - x[0])]
    return list(zip(x_ext[starts], x_ext[ends]))


# --------------------------------------------------------------------------
# Step 4: method B, autocorrelation
# --------------------------------------------------------------------------
def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """acf[k] = how similar the signal is to itself k seconds later.
    1 means identical, 0 means unrelated, -1 means mirror image.

    The plain definition, for lag k:
        acf[k] = sum( x[t] * x[t+k] ) / sum( x[t] * x[t] )     (x mean-removed)

    Computing that directly for every lag up to 4 hours over ~470,000 rows is
    billions of multiplications. The FFT gives the same numbers for all lags
    at once in about a second. Zero-padding to at least twice the length stops
    the end of the signal wrapping around onto its start.

    main() checks the result against np.corrcoef at one lag so you can see the
    two agree. (They differ slightly at long lags because this version divides
    every lag by the full length; at 4 h out of 131 h that is a ~3% shrink.)
    """
    x = x - x.mean()
    size = 1 << (2 * len(x) - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), size)[: max_lag + 1]
    return acf / acf[0]


def dominant_period(acf: np.ndarray):
    """Lag (minutes) and height of the highest autocorrelation peak.

    Peaks closer than 2 minutes to zero lag are ignored (that's sensor noise,
    not a cycle), as are peaks below zero (a bump inside a trough is not a
    match). Returns (nan, nan) when there is no peak at all.
    """
    peaks, _ = find_peaks(acf, prominence=0.05, distance=60)
    peaks = peaks[(peaks >= 120) & (acf[peaks] > 0)]
    if len(peaks) == 0:
        return np.nan, np.nan
    best = peaks[np.argmax(acf[peaks])]
    return best / 60, acf[best]


def classify(peak_height: float, acf_at_1h: float) -> str:
    if peak_height >= CLEAR_PEAK:
        return "clear cycle"
    if peak_height >= WEAK_PEAK:
        return "weak cycle"
    if acf_at_1h >= DRIFT_ACF_AT_1H:
        return "slow drift"      # barely changes within an hour, never repeats
    return "irregular"


# --------------------------------------------------------------------------
# Step 5: measure every column
# --------------------------------------------------------------------------
def measure_all(steady: pd.DataFrame, columns):
    """One row per column with both measurements. Also returns the raw
    switch-on intervals and autocorrelation curves for the figures."""
    rows, intervals, acfs = [], {}, {}
    max_lag = MAX_LAG_MIN * 60
    for col in columns:
        row = {
            "column": col,
            "stage": plant_stage(col),
            "kind": instrument_kind(col),
            "acf_period_min": np.nan,
            "acf_peak": np.nan,
            "pattern": "",
            "n_switch_ons": np.nan,
            "switch_median_min": np.nan,
            "switch_q25_min": np.nan,
            "switch_q75_min": np.nan,
        }
        # A few pumps switch on once during startup and then never change
        # (P402, P501, UV401 stay on; P301 is a standby pump that stays off).
        # A flat signal has no autocorrelation to compute, so record that and
        # move on.
        if steady[col].nunique() == 1:
            state = "on" if steady[col].iloc[0] == ON else "off"
            row["pattern"] = f"always {state} after startup"
            row["n_switch_ons"] = 0
            rows.append(row)
            continue

        x = steady[col].to_numpy(float)
        acf = autocorrelation(x, max_lag)
        acfs[col] = acf
        period, height = dominant_period(acf)
        row["acf_period_min"] = round(period, 1)
        row["acf_peak"] = round(height, 2)
        row["pattern"] = classify(height, acf[3600])
        if is_actuator(col):
            values = steady[col].to_numpy()
            gaps = switch_on_intervals_min(values)
            intervals[col] = gaps
            row["n_switch_ons"] = len(switch_on_rows(values))
            if row["n_switch_ons"] >= MIN_SWITCH_ONS:
                row["switch_median_min"] = round(np.median(gaps), 1)
                row["switch_q25_min"] = round(np.percentile(gaps, 25), 1)
                row["switch_q75_min"] = round(np.percentile(gaps, 75), 1)
        rows.append(row)
    return pd.DataFrame(rows), intervals, acfs


def uf_runs_and_backwash(steady: pd.DataFrame) -> pd.DataFrame:
    """Every uninterrupted UF run (P302 on), its length, and whether a
    backwash (P602 switching on) started within 3 minutes of the run ending.

    Published descriptions say the backwash is "programmed to start every 30
    minutes", yet P602 switches on only about every 70 minutes. This table
    tests the explanation that the 30 minutes counts filtering time, not
    clock time.
    """
    p302 = steady["P302"].to_numpy() == ON
    edges = np.diff(np.r_[0, p302.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    backwash = switch_on_rows(steady["P602"].to_numpy())
    followed = [bool(((backwash >= e) & (backwash <= e + 180)).any()) for e in ends]
    return pd.DataFrame({"run_min": (ends - starts) / 60, "backwash_after": followed})


# --------------------------------------------------------------------------
# Step 6: figures
# --------------------------------------------------------------------------
def fig1_overview(df):
    """The whole file for one sensor: startup, then a steady repeating band."""
    t = df["Timestamp"]
    skip_end = t.iloc[SKIP_HOURS * 3600]
    z0 = t.iloc[ZOOM_START_HOUR * 3600]
    z1 = t.iloc[(ZOOM_START_HOUR + ZOOM_HOURS) * 3600]

    fig, ax = new_figure(
        11, 4.6,
        "Once startup ends, tank T101's level repeats the same pattern for the rest of the file",
        f"LIT101 (tank T101 level) across the whole Normal file, one reading per second.\n"
        f"The first {SKIP_HOURS} hours are startup (tanks filling) and are left out of the cycle "
        f"measurement. The outlined window is the 3 hours shown up close in Figures 2 and 3.",
    )
    ax.plot(t, df["LIT101"], color=BLUE, lw=0.5, rasterized=True)
    ax.axvspan(t.iloc[0], skip_end, color=MUTED, alpha=0.18, lw=0)
    ax.text(t.iloc[SKIP_HOURS * 1800], 1040, "Startup\n(skipped)", ha="center",
            va="center", fontsize=9, color=INK2)
    ax.axvspan(z0, z1, facecolor="none", edgecolor=INK, lw=1.2, zorder=4)
    ax.annotate("3-hour window\n(Figures 2 and 3)", xy=(z1, 1000), xytext=(30, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_ylim(0, 1100)
    ax.set_ylabel("Tank level (mm)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="x", visible=False)
    save(fig, "fig1_overview_LIT101.png")


def _zoom_window(df):
    r0 = ZOOM_START_HOUR * 3600
    w = df.iloc[r0: r0 + ZOOM_HOURS * 3600]
    x = np.arange(len(w)) / 60  # minutes from the window start
    return w, x


def _discrete_axis(ax, ticks, labels):
    ax.set_yticks(ticks, labels)
    ax.set_ylim(min(ticks) - 0.35, max(ticks) + 0.35)
    ax.grid(axis="y", visible=False)


def _cycle_arrow(ax, x0, x1, y, text):
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y, text, ha="center", va="bottom", fontsize=9.5,
            color=INK, weight="semibold",
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))


def fig2_stage1_cycle(df):
    """Pump on -> tank level climbs; pump off -> it drains. Shows one cycle."""
    w, x = _zoom_window(df)
    p101 = w["P101"].to_numpy()

    fig, axes = new_figure(
        11, 7.2,
        "Stage 1: one fill-and-drain round of tank T101 takes about 70 minutes",
        "Three signals over the same 3 hours. Shaded = pump P101 running. The level climbs "
        "fast while the inlet valve MV101 is open and P101 is off,\nfalls fast while P101 runs "
        "with the valve closed, and creeps up while both run. One cycle is measured pump-start "
        "to pump-start.",
        nrows=3, sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]},
    )
    spans = on_spans(p101, x)
    for ax in axes:
        for a, b in spans:
            ax.axvspan(a, b, color=BLUE, alpha=0.08, lw=0, zorder=0)

    axes[0].plot(x, w["LIT101"], color=BLUE)
    axes[0].set_ylabel("LIT101\ntank level (mm)")
    axes[0].set_title("Tank T101 level", pad=6)
    starts = switch_on_rows(p101) / 60
    if len(starts) >= 2:
        top = w["LIT101"].max()
        axes[0].set_ylim(w["LIT101"].min() - 30, top + 90)
        _cycle_arrow(axes[0], starts[0], starts[1], top + 35,
                     f"one cycle = {starts[1] - starts[0]:.0f} min")

    axes[1].step(x, p101, where="post", color=BLUE)
    _discrete_axis(axes[1], [1, 2], ["off", "on"])
    axes[1].set_title("Pump P101 (sends water on to stage 2)", pad=6)

    axes[2].step(x, w["MV101"], where="post", color=BLUE)
    _discrete_axis(axes[2], [0, 1, 2], ["moving", "closed", "open"])
    axes[2].set_title("Inlet valve MV101 (lets raw water into the tank)", pad=6)
    axes[2].set_xlabel("Minutes into the 3-hour window")
    axes[2].set_xlim(0, x[-1])
    save(fig, "fig2_stage1_one_cycle.png")


def fig3_stage3_backwash(df):
    """Stage 3 has a faster rhythm (UF runs) nested inside the 70-min cycle."""
    w, x = _zoom_window(df)
    p302 = w["P302"].to_numpy()
    p602 = w["P602"].to_numpy()
    uf_starts = switch_on_rows(p302) / 60
    bw_starts = switch_on_rows(p602) / 60

    fig, axes = new_figure(
        11, 8.6,
        "Stage 3 runs a faster ~32-minute rhythm inside the 70-minute cycle",
        "Same 3 hours. Shaded = UF feed pump P302 running. The filter (UF) runs up to 30 "
        "minutes at a time. Runs that reach 30 minutes end with\na 30-second backwash "
        "(orange lines) that cleans the filter; Figure 4 checks this across the whole file. "
        "DPIT301 falls whenever the UF stops.",
        nrows=4, sharex=True, gridspec_kw={"height_ratios": [1.5, 0.9, 1.5, 0.9]},
    )
    for ax in axes:
        for a, b in on_spans(p302, x):
            ax.axvspan(a, b, color=BLUE, alpha=0.08, lw=0, zorder=0)
        for s in bw_starts:
            ax.axvline(s, color=ORANGE, lw=1.6, zorder=3)

    axes[0].plot(x, w["LIT301"], color=BLUE)
    axes[0].set_ylabel("LIT301\nlevel (mm)")
    axes[0].set_title("Tank T301 level (feeds the UF filter)", pad=6)
    for s in bw_starts:
        axes[0].text(s + 1.2, 1.0, "backwash", transform=axes[0].get_xaxis_transform(),
                     ha="left", va="top", fontsize=8.5, color=INK2)

    axes[1].step(x, p302, where="post", color=BLUE)
    _discrete_axis(axes[1], [1, 2], ["off", "on"])
    axes[1].set_title("UF feed pump P302", pad=6)
    if len(uf_starts) >= 2:
        axes[1].set_ylim(0.65, 2.75)
        _cycle_arrow(axes[1], uf_starts[0], uf_starts[1], 2.35,
                     f"UF restarts after {uf_starts[1] - uf_starts[0]:.0f} min")

    axes[2].plot(x, w["DPIT301"], color=BLUE)
    axes[2].set_ylabel("DPIT301\npressure drop")
    axes[2].set_title("Pressure drop across the UF filter", pad=6)

    axes[3].step(x, p602, where="post", color=BLUE)
    _discrete_axis(axes[3], [1, 2], ["off", "on"])
    axes[3].set_title("Backwash pump P602", pad=6)
    if len(bw_starts) >= 2:
        axes[3].set_ylim(0.65, 2.75)
        _cycle_arrow(axes[3], bw_starts[0], bw_starts[1], 2.35,
                     f"backwash every {bw_starts[1] - bw_starts[0]:.0f} min")
    axes[3].set_xlabel("Minutes into the 3-hour window")
    axes[3].set_xlim(0, x[-1])
    save(fig, "fig3_stage3_ultrafiltration_and_backwash.png")


def fig4_backwash_rule(runs):
    """Why a published "every 30 minutes" and our "every 70 minutes" are both right."""
    with_bw = runs.loc[runs["backwash_after"], "run_min"]
    without = runs.loc[~runs["backwash_after"], "run_min"]
    fig, axes = new_figure(
        11, 6.6,
        "The backwash fires after 30 minutes of filtering, not every 30 minutes of clock time",
        "Every uninterrupted UF run (P302 on), split by whether a backwash (P602) started "
        "within 3 minutes of it ending. Runs that end in a backwash\nlast about 30 minutes; "
        "runs that stop sooner do not trigger one. Because the UF also pauses while tanks "
        "refill, a backwash comes about every 70\nminutes of clock time. So the published "
        "\"every 30 minutes\" and our measured 70 minutes describe the same rule.",
        nrows=2, sharex=True,
    )
    bins = np.arange(0, 32.25, 0.25)
    for ax, data, color, label in [
        (axes[0], with_bw, ORANGE, "followed by a backwash"),
        (axes[1], without, BLUE, "not followed by a backwash"),
    ]:
        ax.hist(data, bins=bins, color=color, edgecolor=SURFACE, linewidth=0.6, zorder=2)
        ax.axvline(30, color=INK, lw=1, ls=(0, (4, 3)), zorder=3)
        ax.set_title(f"UF runs {label}: {len(data)} runs, median {np.median(data):.1f} min", pad=6)
        ax.set_ylabel("Number of runs")
        ax.grid(axis="x", visible=False)
    axes[0].text(29.5, 0.95, "30 min of\nfiltering", transform=axes[0].get_xaxis_transform(),
                 ha="right", va="top", fontsize=9, color=INK2)
    axes[1].set_xlabel("Length of the UF run (minutes)")
    axes[1].set_xlim(0, 32)
    save(fig, "fig4_backwash_rule.png")


def fig5_autocorrelation_explained(steady, acfs):
    """What autocorrelation does, shown on LIT101, plus a signal with no cycle."""
    seg_min = 150
    x = steady["LIT101"].to_numpy()
    start = (ZOOM_START_HOUR - SKIP_HOURS) * 3600
    t = np.arange(seg_min * 60) / 60
    acf = acfs["LIT101"]
    lags = np.arange(len(acf)) / 60

    fig, axes = new_figure(
        12, 8.4,
        "Autocorrelation: slide a copy of the signal along itself until it lines up",
        "Top: LIT101 (blue) against a copy of itself taken later in time (orange). After 35 "
        "minutes the copy is out of step; after 70 minutes it lines up again.\nBottom left: "
        "the similarity score for every shift from 0 to 4 hours. The highest peak marks the "
        "cycle length. Bottom right: a sensor with no cycle has no peak.",
        nrows=2, ncols=2,
    )
    for ax, shift, verdict in [(axes[0, 0], 35, "out of step"), (axes[0, 1], 70, "lined up")]:
        k = shift * 60
        ax.plot(t, x[start: start + len(t)], color=BLUE, label="LIT101")
        ax.plot(t, x[start + k: start + k + len(t)], color=ORANGE,
                label=f"LIT101, {shift} min later")
        ax.set_title(f"Shifted by {shift} min: {verdict} (score {acf[k]:+.2f})", pad=6)
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Tank level (mm)")
        ax.set_xlim(0, seg_min)
        ax.set_ylim(470, 900)
        ax.legend(loc="upper left", fontsize=8.5, ncols=2)

    ax = axes[1, 0]
    ax.plot(lags, acf, color=BLUE)
    ax.axhline(0, color=AXIS, lw=0.8)
    for shift, color in [(35, ORANGE), (70, INK)]:
        k = shift * 60
        ax.plot(lags[k], acf[k], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"{shift} min: {acf[k]:+.2f}", (lags[k], acf[k]),
                    xytext=(8, 6 if acf[k] > 0 else -14), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_title("LIT101: peak at 70 min = the cycle length", pad=6)
    ax.set_xlabel("Shift (minutes)")
    ax.set_ylabel("Similarity score")
    ax.set_xlim(0, MAX_LAG_MIN)
    ax.set_ylim(-1.05, 1.05)

    ax = axes[1, 1]
    ax.plot(lags, acfs["AIT201"], color=BLUE)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title("AIT201 (conductivity): no peak, it only drifts", pad=6)
    ax.set_xlabel("Shift (minutes)")
    ax.set_xlim(0, MAX_LAG_MIN)
    ax.set_ylim(-1.05, 1.05)
    ax.text(120, 0.55, "Still almost identical hours later,\nbut never dips and comes back.\n"
            "That is slow drift, not a cycle.", ha="center", va="center", fontsize=9, color=INK2)
    save(fig, "fig5_autocorrelation_explained.png")


def fig6_periods_by_column(table, constant_cols, slowest):
    """The summary: every changing column, grouped by stage, at its cycle length."""
    rows = table.sort_values(["stage", "column"])
    ypos, ylabels, headers = [], [], []
    y = 0.0
    for stage, grp in rows.groupby("stage"):
        headers.append((y, f"Stage {stage}: {STAGE_NAMES[stage]}"))
        y += 1
        for _, r in grp.iterrows():
            ypos.append(y)
            ylabels.append(r["column"])
            y += 1
        y += 0.5

    fig, ax = new_figure(
        10.5, 13,
        "Almost every signal that cycles repeats about every 70 minutes",
        "Each row is one sensor or actuator, grouped by plant stage, placed at its measured cycle "
        "length (autocorrelation peak).\nFilled dot = clear cycle; hollow = weak cycle (short "
        "pulses). Dashed lines are the block sizes under consideration.\n"
        f"Not shown: {len(constant_cols)} columns that never change ({', '.join(constant_cols)}).",
    )
    ax.axvspan(66, 74, color=BLUE, alpha=0.07, lw=0, zorder=0)
    block_size_lines(ax)
    for yy, (_, r) in zip(ypos, rows.iterrows()):
        if r["pattern"] == "clear cycle":
            ax.scatter(r["acf_period_min"], yy, s=55, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=4)
        elif r["pattern"] == "weak cycle":
            ax.scatter(r["acf_period_min"], yy, s=55, facecolor=SURFACE, edgecolor=BLUE, lw=1.6, zorder=4)
        else:
            text = {
                "slow drift": "no repeating cycle (slow drift)",
                "always on after startup": "switched on at startup, then never changes",
                "always off after startup": "standby pump, stays off after startup",
            }.get(r["pattern"], "no regular pattern")
            ax.text(3, yy, text, va="center", fontsize=8.5, color=MUTED)
    for yy, text in headers:
        ax.text(1, yy, text, va="center", fontsize=10, weight="semibold", color=INK)

    col, minutes = slowest
    ys = ypos[ylabels.index(col)]
    ax.annotate(f"slowest clear cycle: {col}, {minutes:.1f} min", (minutes, ys),
                xytext=(18, 0), textcoords="offset points", va="center", fontsize=9,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

    ax.set_yticks(ypos, ylabels, fontsize=8.5)
    ax.set_ylim(y - 0.3, -0.8)
    ax.set_xlim(0, 135)
    ax.set_xlabel("Cycle length (minutes)")
    ax.grid(axis="y", visible=False)
    handles = [
        Line2D([], [], marker="o", ls="", ms=8, color=BLUE, mec=SURFACE, label="clear cycle"),
        Line2D([], [], marker="o", ls="", ms=8, mfc=SURFACE, mec=BLUE, mew=1.6, label="weak cycle (short pulses)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    save(fig, "fig6_cycle_length_by_column.png")


def fig7_switch_on_intervals(table, intervals):
    """Method A for every actuator that switches often enough."""
    acts = table[table["n_switch_ons"] >= MIN_SWITCH_ONS].sort_values(["stage", "column"])
    names = list(acts["column"])
    data = [intervals[c] for c in names]
    xmax = 160
    hidden = sum(int((d > xmax).sum()) for d in data)

    fig, ax = new_figure(
        11, 6.4,
        "Counting switch-ons gives the same answer: two rhythms, ~32 and ~70 minutes",
        "Each row is one pump or valve. Every dot is the time between two consecutive "
        "switch-ons; the box holds the middle half of them.\nStage 3's UF valves restart "
        "about every 32 minutes; everything else follows the 70-minute plant cycle."
        + (f" {hidden} intervals longer than {xmax} min are off the right edge." if hidden else ""),
    )
    block_size_lines(ax)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        jitter = rng.uniform(-0.18, 0.18, len(d))
        ax.scatter(d, i + jitter, s=7, color=BLUE, alpha=0.35, lw=0, zorder=2)
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True,
                    showfliers=False, tick_labels=names, zorder=3)
    for box in bp["boxes"]:
        box.set(facecolor=(0, 0, 0, 0), edgecolor=INK2, linewidth=1)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color=INK2, linewidth=0.9)
    for line in bp["medians"]:
        line.set(color=INK, linewidth=2)
    stage_of = [plant_stage(c) for c in names]
    for i in range(1, len(names)):
        if stage_of[i] != stage_of[i - 1]:
            ax.axhline(i + 0.5, color=GRID, lw=1)
    ax.set_xlim(0, xmax)
    ax.invert_yaxis()
    ax.set_xlabel("Minutes between consecutive switch-ons")
    ax.grid(axis="y", visible=False)
    save(fig, "fig7_switch_on_intervals.png")


def fig8_cycle_stability(steady):
    """Is the 70-minute cycle the same every day? One dot per P101 cycle."""
    rows = switch_on_rows(steady["P101"].to_numpy())
    gaps = np.diff(rows) / 60
    when = steady["Timestamp"].to_numpy()[rows[1:]]
    med = np.median(gaps)
    short = int((gaps < 0.75 * med).sum())
    # Median cycle length per calendar day, for days with enough cycles to count.
    by_day = pd.Series(gaps, index=pd.DatetimeIndex(when)).groupby(lambda d: d.date())
    daily = by_day.median()[by_day.size() >= 10]

    fig, ax = new_figure(
        11, 4.8,
        f"The cycle length stays close to {med:.0f} minutes every day, creeping up slightly",
        "One dot per stage 1 cycle (time between consecutive P101 switch-ons), plotted at the "
        "time the cycle ended. Daily medians run from\n"
        f"{daily.min():.1f} to {daily.max():.1f} minutes, slightly longer on the last days. "
        f"{short} cycles were unusually short (the dots well below the line).",
    )
    ax.scatter(when, gaps, s=18, color=BLUE, edgecolor=SURFACE, lw=0.8, zorder=3)
    ax.axhline(med, color=INK2, lw=1, zorder=2)
    ax.text(when[0], med + 12, f"Line = overall median, {med:.1f} min", ha="left",
            va="bottom", fontsize=9, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Cycle length (minutes)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    save(fig, "fig8_cycle_length_over_time.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_normal()
    cols = sensor_columns(df)
    steady = df.iloc[SKIP_HOURS * 3600:].reset_index(drop=True)
    print(f"Normal file: {len(df):,} rows ({len(df) / 3600:.1f} h). "
          f"Analysing {len(steady):,} rows after skipping the first {SKIP_HOURS} h.")

    constant_cols = [c for c in cols if df[c].nunique() == 1]
    changing = [c for c in cols if c not in constant_cols]
    print(f"{len(constant_cols)} columns never change, skipped: {', '.join(constant_cols)}")
    print(f"{len(changing)} columns change and are measured.\n")

    table, intervals, acfs = measure_all(steady, changing)

    # Self-check: the FFT autocorrelation should agree with a plain correlation.
    k = 70 * 60
    x = steady["LIT101"].to_numpy(float)
    direct = np.corrcoef(x[:-k], x[k:])[0, 1]
    print(f"Check, LIT101 at a 70-min shift: FFT autocorrelation {acfs['LIT101'][k]:.3f}, "
          f"np.corrcoef {direct:.3f} (should be close)\n")

    # The slowest clear cycle, from either method.
    candidates = {}
    for _, r in table.iterrows():
        if r["pattern"] == "clear cycle":
            candidates[r["column"]] = r["acf_period_min"]
        if not np.isnan(r["switch_median_min"]):
            candidates[r["column"]] = max(candidates.get(r["column"], 0), r["switch_median_min"])
    slowest_col = max(candidates, key=candidates.get)
    slowest = (slowest_col, candidates[slowest_col])

    print("Per-column results:")
    print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "cycle_periods.csv", index=False)

    runs = uf_runs_and_backwash(steady)
    bw = runs[runs["backwash_after"]]
    print(f"\nUF runs: {len(runs)}, of which {len(bw)} end in a backwash "
          f"(median run {bw['run_min'].median():.1f} min, vs "
          f"{runs.loc[~runs['backwash_after'], 'run_min'].median():.1f} min for the rest)")

    print("\nPattern counts:", table["pattern"].value_counts().to_dict())
    print(f"Slowest clear cycle: {slowest[0]} at {slowest[1]:.1f} min")
    print("Cycles that fit in each candidate block size:")
    for b in BLOCK_CANDIDATES_MIN:
        print(f"  {b:>4}-min block holds {b / slowest[1]:.2f} cycles")

    print("\nFigures:")
    fig1_overview(df)
    fig2_stage1_cycle(df)
    fig3_stage3_backwash(df)
    fig4_backwash_rule(runs)
    fig5_autocorrelation_explained(steady, acfs)
    fig6_periods_by_column(table, constant_cols, slowest)
    fig7_switch_on_intervals(table, intervals)
    fig8_cycle_stability(steady)
    print(f"  wrote {(OUT_DIR / 'cycle_periods.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
