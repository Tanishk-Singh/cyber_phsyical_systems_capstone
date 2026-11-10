"""Does the SWaT plant behave differently on different days?

Why this exists
---------------
Fung et al. (Appendix A, Finding #3) say the benign data must be split at
random rather than at a fixed time, because "system behavior can differ
between days". Our Step 3 split follows them: random 75-minute blocks.
This script checks the claim on our own copy of the Normal file, so the
choice rests on evidence instead of on a citation.

The comparison that matters
---------------------------
We need to know which kind of split gives a validation set that looks like
the training set:

  Tail split   first 70% of the file for training, last 30% for validation
               (Fung's own example of what goes wrong).
  Block split  random 75-minute blocks, 80% training and 20% validation
               (our Step 3 design). Repeated with 20 different random draws,
               because one draw could be lucky or unlucky.

For each sensor, "how different do the two sets look?" is measured with the
KS statistic: the largest gap between the two sets' cumulative distributions,
from 0 (identical) to 1 (no overlap). Figure 1 shows exactly what that gap is.

A note on p-values: they are deliberately not used. The file has 470,000
readings, one per second, and neighbouring readings are almost copies of each
other. With that much (heavily repeated) data a significance test calls even
trivial differences "significant". The size of the gap is what matters, and
the block split gives us a yardstick for how big a gap is just noise.

Run from the repo root (reuses the cached Normal file from
codebase/swat_cycle_analysis.py, so run that script once first, or this one
will load the xlsx itself, about 1.5 minutes):

    python codebase/swat_drift_analysis.py

Writes 4 figures and a CSV table to results/swat_drift/ (git-ignored).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ks_2samp

from ics_tsfm.data.swat import sensor_columns
# Shared chart style and the cached Normal-file loader.
from swat_cycle_analysis import (
    AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, REPO_ROOT, SKIP_HOURS, STAGE_NAMES,
    SURFACE, apply_style, load_normal, new_figure, plant_stage,
)

OUT_DIR = REPO_ROOT / "results" / "swat_drift"

BLOCK_ROWS = 75 * 60      # the block size chosen for Step 3: 75 minutes
TRAIN_SHARE = 0.8         # block split: 80% of blocks train, 20% validate
TAIL_TRAIN_SHARE = 0.7    # tail split: first 70% train, last 30% validate
N_DRAWS = 20              # random block assignments to try
N_PARTS = 5               # equal slices of the file for the part-by-part view
SEED = 0


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------
def ks_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Largest vertical gap between the two samples' cumulative distributions.

    The cumulative distribution F(x) is "the share of readings at or below x".
    Walk along every value that appears in either sample and record
    |F_a(x) - F_b(x)|; the largest value is the KS statistic.

    Written out by hand (instead of calling scipy) so you can read it; main()
    checks it against scipy.stats.ks_2samp.
    """
    a, b = np.sort(a), np.sort(b)
    values = np.concatenate([a, b])
    share_a = np.searchsorted(a, values, side="right") / len(a)
    share_b = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(share_a - share_b)))


def tail_split_mask(n_rows: int) -> np.ndarray:
    """True for validation rows: the last 30% of the file."""
    mask = np.zeros(n_rows, bool)
    mask[int(n_rows * TAIL_TRAIN_SHARE):] = True
    return mask


def block_split_mask(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """True for validation rows: whole 75-minute blocks chosen at random.

    Every block gets its own coin flip (20% validation), and all rows in a
    block share that block's outcome. This is the Step 3 design, minus the
    seam trimming, which removes samples but does not change which blocks
    are train or validation.
    """
    n_blocks = int(np.ceil(n_rows / BLOCK_ROWS))
    block_is_val = rng.random(n_blocks) >= TRAIN_SHARE
    return np.repeat(block_is_val, BLOCK_ROWS)[:n_rows]


def measure(steady: pd.DataFrame, columns):
    """One row per sensor: tail-split gap, and the spread of block-split gaps."""
    n = len(steady)
    tail_val = tail_split_mask(n)
    rng = np.random.default_rng(SEED)
    block_masks = [block_split_mask(n, rng) for _ in range(N_DRAWS)]

    rows = []
    for col in columns:
        x = steady[col].to_numpy(float)
        tail = ks_gap(x[~tail_val], x[tail_val])
        blocks = np.array([ks_gap(x[~m], x[m]) for m in block_masks])
        rows.append({
            "column": col,
            "stage": plant_stage(col),
            "tail_split_gap": round(tail, 3),
            "block_split_median": round(float(np.median(blocks)), 3),
            "block_split_max": round(float(blocks.max()), 3),
            "block_split_min": round(float(blocks.min()), 3),
        })
    table = pd.DataFrame(rows)
    # Flagged when the tail split is worse than ALL 20 block-split draws, and
    # the gap is big enough to see (not just a hair above a tiny number).
    table["drifts"] = (table["tail_split_gap"] > table["block_split_max"]) & (
        table["tail_split_gap"] >= 0.10)
    return table


def part_vs_rest(steady: pd.DataFrame, columns) -> pd.DataFrame:
    """For each sensor and each of 5 equal slices of the file: how different
    does that slice look from the other four slices combined?"""
    n = len(steady)
    edges = np.linspace(0, n, N_PARTS + 1).astype(int)
    out = {}
    for col in columns:
        x = steady[col].to_numpy(float)
        gaps = []
        for p in range(N_PARTS):
            inside = np.zeros(n, bool)
            inside[edges[p]:edges[p + 1]] = True
            gaps.append(ks_gap(x[~inside], x[inside]))
        out[col] = gaps
    return pd.DataFrame(out, index=[f"Part {p + 1}" for p in range(N_PARTS)]).T


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def headline(table):
    """The takeaway, built from the numbers so it can't contradict them."""
    n_sens = len(table)
    n_drift = int(table["drifts"].sum())
    tail_med = table["tail_split_gap"].median()
    block_med = table["block_split_median"].median()
    return n_sens, n_drift, tail_med, block_med


def fig1_what_the_gap_is(steady, table):
    """One sensor, two cumulative curves, and the gap between them."""
    col = table.sort_values("tail_split_gap", ascending=False).iloc[0]["column"]
    x = steady[col].to_numpy(float)
    val = tail_split_mask(len(x))
    a, b = np.sort(x[~val]), np.sort(x[val])
    grid = np.unique(np.concatenate([a, b]))
    grid = grid[:: max(1, len(grid) // 4000)]            # thin for plotting only
    share_a = np.searchsorted(a, grid, side="right") / len(a)
    share_b = np.searchsorted(b, grid, side="right") / len(b)
    i = int(np.argmax(np.abs(share_a - share_b)))
    gap = abs(share_a[i] - share_b[i])

    fig, ax = new_figure(
        9, 5.6,
        f"The KS gap measures how far apart two sets of readings are: {col} gives {gap:.2f}",
        f"For {col}, the share of readings at or below each value, for the first 70% of the file "
        f"(training) and the last 30%\n(validation). The vertical line is the biggest gap between "
        f"the curves. 0 means identical readings; 1 means no overlap at all.",
    )
    ax.plot(grid, share_a, color=BLUE, label="first 70% (training)")
    ax.plot(grid, share_b, color=ORANGE, label="last 30% (validation)")
    mid = (share_a[i] + share_b[i]) / 2
    ax.plot([grid[i], grid[i]], [share_a[i], share_b[i]], color=INK, lw=2, zorder=4)
    ax.annotate(f"biggest gap = {gap:.2f}", (grid[i], mid), xytext=(14, 0),
                textcoords="offset points", va="center", fontsize=10, weight="semibold")
    ax.set_xlabel(f"{col} reading")
    ax.set_ylabel("Share of readings at or below this value")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    save(fig, "fig1_what_the_gap_means.png")


def fig2_tail_vs_blocks(table):
    """The main result: gap for the tail split vs the random block split."""
    n_sens, n_drift, tail_med, block_med = headline(table)
    t = table.sort_values("tail_split_gap").reset_index(drop=True)
    y = np.arange(len(t))

    flagged_rows = t[t["drifts"]]
    shrink = (flagged_rows["tail_split_gap"] / flagged_rows["block_split_median"]).median()
    if n_drift >= n_sens / 4:
        title = (f"A fixed tail split leaves a large gap on {n_drift} of {n_sens} sensors; "
                 f"random blocks shrink it about {shrink:.0f}-fold")
    elif n_drift > 0:
        title = (f"A fixed tail split fails on only {n_drift} of {n_sens} sensors, "
                 f"but random blocks match better on all of them")
    else:
        title = (f"No sensor shows drift: the tail split is no worse than random blocks "
                 f"on all {n_sens}")

    fig, ax = new_figure(
        10, max(7.5, 0.30 * len(t) + 2.8),
        title,
        f"Each row is a sensor. Orange dot = gap between training and validation when the file is cut at "
        f"70/30 in time.\nBlue bar = range of the same gap over {N_DRAWS} random draws of 75-minute "
        f"blocks (80/20). Filled dot = tail split worse than every block draw.\n"
        f"Median gap over all sensors: tail {tail_med:.2f}, blocks {block_med:.2f}. Blocks help a lot but "
        f"not completely: slow drift still leaves some gap.",
    )
    ax.hlines(y, t["block_split_min"], t["block_split_max"], color=BLUE, lw=6,
              alpha=0.85, capstyle="round", zorder=2)
    flagged = t["drifts"].to_numpy()
    ax.scatter(t.loc[flagged, "tail_split_gap"], y[flagged], s=62, color=ORANGE,
               edgecolor=SURFACE, lw=1.5, zorder=4)
    ax.scatter(t.loc[~flagged, "tail_split_gap"], y[~flagged], s=62, facecolor=SURFACE,
               edgecolor=ORANGE, lw=1.7, zorder=4)
    ax.set_yticks(y, t["column"], fontsize=8.5)
    ax.set_xlim(0, max(0.5, t["tail_split_gap"].max() * 1.08))
    ax.set_xlabel("KS gap between training and validation (0 = identical)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.8, len(t) - 0.2)
    save(fig, "fig2_tail_split_vs_random_blocks.png")


def fig3_part_by_part(parts, steady):
    """Which sensors move over the file, and when."""
    stage_order = parts.index.to_series().map(plant_stage)
    parts = parts.loc[stage_order.sort_values(kind="stable").index]
    hours = len(steady) / 3600
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#3987e5", "#0d366b"])

    n_move = int((parts.max(axis=1) >= 0.3).sum())
    fig, ax = new_figure(
        7.5, max(8.5, 0.27 * len(parts) + 2.6),
        f"{n_move} of {len(parts)} sensors have at least one part of the file that looks clearly different",
        f"The file after startup ({hours:.0f} hours) cut into {N_PARTS} equal parts of about "
        f"{hours / N_PARTS:.0f} hours each.\nColour = how different that part looks from the other four "
        f"combined (KS gap; darker = more different).",
    )
    im = ax.imshow(parts.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=max(0.4, parts.to_numpy().max()),
                   interpolation="nearest")
    ax.set_xticks(range(N_PARTS), parts.columns, fontsize=9)
    ax.set_yticks(range(len(parts)), parts.index, fontsize=8.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    stages = parts.index.map(plant_stage).to_numpy()
    for i in range(1, len(parts)):
        if stages[i] != stages[i - 1]:
            ax.axhline(i - 0.5, color=SURFACE, lw=3)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.03)
    cbar.set_label("KS gap vs the other four parts", color=INK2)
    cbar.outline.set_visible(False)
    save(fig, "fig3_part_by_part_heatmap.png")


def fig4_trends(steady, table):
    """What drift looks like: the biggest movers, averaged per 75-minute block
    (averaging over a block removes the 70-minute cycle so the trend shows)."""
    top = table.sort_values("tail_split_gap", ascending=False).head(4)["column"].tolist()
    n_blocks = len(steady) // BLOCK_ROWS
    cut = n_blocks * BLOCK_ROWS
    t = steady["Timestamp"].to_numpy()[:cut:BLOCK_ROWS]
    split_t = steady["Timestamp"].iloc[int(len(steady) * TAIL_TRAIN_SHARE)]

    fig, axes = new_figure(
        11, 7.2,
        "The sensors that move most change gradually over the days, so the end of the file looks unlike the start",
        "The four sensors with the biggest tail-split gap. Each dot is the average of one 75-minute block, "
        "which smooths out the 70-minute cycle.\nThe shaded area is the last 30% of the file, the part a "
        "tail split would use for validation.",
        nrows=2, ncols=2, sharex=True,
    )
    for ax, col in zip(axes.ravel(), top):
        means = steady[col].to_numpy(float)[:cut].reshape(n_blocks, BLOCK_ROWS).mean(axis=1)
        ax.axvspan(split_t, t[-1], color=MUTED, alpha=0.16, lw=0)
        ax.plot(t, means, color=BLUE, lw=1.3)
        ax.set_title(f"{col}  (tail-split gap {table.set_index('column').loc[col, 'tail_split_gap']:.2f})", pad=6)
        ax.set_ylabel("Block average")
    for ax in axes[1]:
        ax.tick_params(axis="x", labelrotation=0)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    save(fig, "fig4_trends_of_biggest_movers.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_normal()
    steady = df.iloc[SKIP_HOURS * 3600:].reset_index(drop=True)
    columns = [c for c in sensor_columns(df) if steady[c].nunique() > 1]
    print(f"Analysing {len(steady):,} rows after skipping {SKIP_HOURS} h of startup.")
    print(f"{len(columns)} sensors change after startup (the rest are constant and cannot drift).")

    # Self-check: my hand-written KS gap should match scipy's.
    x = steady["LIT301"].to_numpy(float)
    val = tail_split_mask(len(x))
    print(f"Check, LIT301 tail split: hand-written {ks_gap(x[~val], x[val]):.4f}, "
          f"scipy {ks_2samp(x[~val], x[val]).statistic:.4f} (should match)\n")

    table = measure(steady, columns)
    parts = part_vs_rest(steady, columns)

    n_sens, n_drift, tail_med, block_med = headline(table)
    print(f"Median gap: tail split {tail_med:.3f}, random blocks {block_med:.3f}")
    print(f"{n_drift} of {n_sens} sensors have a tail-split gap above every block draw "
          f"and at least 0.10:")
    print(table.sort_values("tail_split_gap", ascending=False).to_string(index=False))
    table.to_csv(OUT_DIR / "drift_by_sensor.csv", index=False)

    print("\nFigures:")
    fig1_what_the_gap_is(steady, table)
    fig2_tail_vs_blocks(table)
    fig3_part_by_part(parts, steady)
    fig4_trends(steady, table)
    print(f"  wrote {(OUT_DIR / 'drift_by_sensor.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
