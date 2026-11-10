"""When has the SWaT plant finished starting up?

Why this exists
---------------
The Normal file begins with the plant being switched on: tanks nearly empty,
pumps running for the first time. The Attack file, by contrast, continues
second-for-second from the end of the Normal file, so it contains no startup
at all. Training on startup rows teaches the model a state it is never tested
on, and a startup block that lands in validation inflates the alarm threshold.
Step 3 therefore cuts the start of the Normal file. This script measures how
much to cut instead of guessing (the cycle analysis used a rough 6 hours).

The measurement
---------------
  1. Cut the file into 75-minute pieces, the same size as the Step 3 blocks.
  2. For every piece and every column, record two numbers: the piece's
     average (where the signal sits) and its spread (how much it moves inside
     the piece; this is what separates a tank that fills and drains from one
     that only fills).
  3. Take hours 24 to 48 as the "settled" reference. For each column, the
     settled band of each number is the range seen across the reference
     pieces, widened by its own width on both sides. Example: reference
     averages from 500 to 560 (width 60) give a band of 440 to 620.
     The widening is needed because 19 reference pieces is a small sample:
     without it, ordinary pieces later in the file fall outside the band
     10-20% of the time just by chance.
  4. A piece is "in band" when both its average and its spread are in band.
     A column has settled at the first piece from which every piece up to
     the reference window is in band.
  5. The plant has settled when its slowest column has.

What we decided (Step 3): the slowest column settles at 22.5 h, but most of
that is a stage-2 dosing mode (P205 switching every 24 min until about hour
20), not startup. Cutting it would drop 16% of the training data and tailor
training to what the test set happens to contain; the models should learn
that mode as normal. So the cutoff is 3.75 h: the point where most columns
settle (the tanks have filled), exactly 3 blocks of 75 minutes. The 22.5 h
figure is still printed, as the strict reading of the rule.

Why compare only up to hour 24, not to the end of the file: several sensors
drift slowly over the days, so pieces late in the file leave the hour 24-48
band for reasons that have nothing to do with startup.

Only the 36 columns kept in Step 3 are measured (Fung's 34 plus AIT202 and
P205): those are the columns the models will see.

Run from the repo root (reuses the cached Normal file from
codebase/swat_cycle_analysis.py; the Attack file is loaded for the last
check, about a minute):

    python codebase/swat_startup_analysis.py

Writes 2 figures and a CSV table to results/swat_startup/ (git-ignored).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ics_tsfm.data.swat import corrected_attack_mask, load_swat_attack, sensor_columns
# Shared chart style and the cached Normal-file loader.
from swat_cycle_analysis import (
    BLUE, INK, INK2, MUTED, ORANGE, REPO_ROOT, SWAT_DIR,
    apply_style, load_normal, new_figure,
)

OUT_DIR = REPO_ROOT / "results" / "swat_startup"

PIECE_ROWS = 75 * 60        # 75-minute pieces, the Step 3 block size
PIECE_HOURS = PIECE_ROWS / 3600
REF_START_H, REF_END_H = 24, 48
WIDEN = 1.0                 # widen the reference range by this many widths per side
OUTSIDE_LIMIT_PCT = 10      # Step 3 column rule: more than this = large drift
CUTOFF_HOURS = 3.75         # the Step 3 decision (see the docstring)
PLOT_HOURS = 32

# Fung's 17 removed columns minus the two we keep (AIT202, P205).
DROPPED = {
    "P202", "P401", "P404", "P502", "P601", "P603",
    "AIT201", "AIT203", "AIT401", "AIT402", "AIT501", "AIT502", "AIT503",
    "PIT502", "P201",
}
# Columns for Figure 1: the three tanks, a pump, and the two kept-by-exception columns.
KEY_COLUMNS = ["LIT101", "LIT301", "LIT401", "P101", "P205", "AIT202"]


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------
def piece_stats(x: np.ndarray):
    """Average and spread of every full 75-minute piece.

    Example with 3-row pieces: x = [1, 2, 3, 10, 10, 10, 7]
      -> pieces [1, 2, 3] and [10, 10, 10] (the leftover 7 is dropped)
      -> averages [2, 10], spreads [0.82, 0]
    """
    n_pieces = len(x) // PIECE_ROWS
    pieces = x[:n_pieces * PIECE_ROWS].reshape(n_pieces, PIECE_ROWS)
    return pieces.mean(axis=1), pieces.std(axis=1)


def reference_pieces():
    """Piece numbers inside the hour 24-48 reference window."""
    return range(int(REF_START_H / PIECE_HOURS), int(REF_END_H / PIECE_HOURS))


def settled_band(stat: np.ndarray):
    """The reference range of one number, widened by its own width per side."""
    ref = stat[list(reference_pieces())]
    width = ref.max() - ref.min()
    return ref.min() - WIDEN * width, ref.max() + WIDEN * width


def in_band(x: np.ndarray) -> np.ndarray:
    """True for every piece whose average AND spread are inside the settled band."""
    ok = np.ones(len(x) // PIECE_ROWS, bool)
    for stat in piece_stats(x):
        lo, hi = settled_band(stat)
        ok &= (stat >= lo) & (stat <= hi)
    return ok


def settle_piece(ok: np.ndarray) -> int:
    """First piece from which every piece up to the reference window is in band.

    Walk backwards from the reference start and stop at the first piece that
    is out of band. Example: ok = [F, F, T, F, T, T | reference...]
    -> walking back: T, T, then F at piece 3 -> settled at piece 4.
    """
    k = reference_pieces()[0]
    while k > 0 and ok[k - 1]:
        k -= 1
    return k


def measure(df: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for col in columns:
        ok = in_band(df[col].to_numpy(float))
        k = settle_piece(ok)
        rows.append({
            "column": col,
            "settle_hours": k * PIECE_HOURS,
            # How many pieces before settling were already in band: a column
            # that is in band almost everywhere and fails only once is an
            # isolated event, not a slow startup.
            "out_of_band_pieces_before_settling": int((~ok[:k]).sum()),
            "pieces_in_band_first_32h": "".join("#" if v else "." for v in ok[:int(PLOT_HOURS / PIECE_HOURS)]),
        })
    return pd.DataFrame(rows).sort_values("settle_hours", ascending=False).reset_index(drop=True)


def outside_range_pct(train: np.ndarray, test: np.ndarray) -> float:
    """Share of test readings outside the training min..max (the Step 3 column rule)."""
    return float(((test < train.min()) | (test > train.max())).mean() * 100)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig1_first_hours(df, table):
    """Raw signal, piece averages coloured by in/out of band, and the settling point."""
    settle = table.set_index("column")["settle_hours"]
    rows = int(PLOT_HOURS * 3600)
    t = np.arange(rows) / 3600
    n_pieces = int(PLOT_HOURS / PIECE_HOURS)
    centers = (np.arange(n_pieces) + 0.5) * PIECE_HOURS

    fig, axes = new_figure(
        12, 9.6,
        "The tanks settle within about 4 hours, but the dosing pump P205 and AIT202 keep a startup pattern for much longer",
        "First 32 hours of the Normal file. Thin line = the raw signal. Dots = average of each 75-minute piece: blue when the "
        "piece's average and spread\nare both in the settled band, orange when not. Blue band = settled band for the "
        "average. Grey area = the hour 24-48 reference.",
        nrows=3, ncols=2, sharex=True,
    )
    for ax, col in zip(axes.ravel(), KEY_COLUMNS):
        x = df[col].to_numpy(float)
        means, _ = piece_stats(x)
        lo, hi = settled_band(means)
        ok = in_band(x)[:n_pieces]
        ax.axvspan(REF_START_H, PLOT_HOURS, color=MUTED, alpha=0.14, lw=0)
        ax.axhspan(lo, hi, color=BLUE, alpha=0.10, lw=0)
        ax.plot(t[::10], x[:rows:10], color=INK2, lw=0.5, alpha=0.6)
        ax.scatter(centers[ok], means[:n_pieces][ok], s=26, color=BLUE, zorder=4)
        ax.scatter(centers[~ok], means[:n_pieces][~ok], s=26, color=ORANGE, zorder=4)
        s = settle[col]
        ax.axvline(s, color=INK, lw=1.1, ls=(0, (4, 3)))
        ax.text(s, 1.01, f" settled at {s:g} h", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=9, color=INK)
        ax.set_title(col, pad=16)
    for ax in axes[-1]:
        ax.set_xlabel("Hours from the start of the Normal file")
        ax.set_xlim(0, PLOT_HOURS)
    save(fig, "fig1_first_hours_key_columns.png")


def fig2_settle_by_column(table):
    """One bar per column; orange bars settle after the chosen cutoff."""
    t = table.sort_values("settle_hours").reset_index(drop=True)
    y = np.arange(len(t))
    late = (t["settle_hours"] > CUTOFF_HOURS).to_numpy()
    n_by_cutoff = int((~late).sum())

    fig, ax = new_figure(
        9, max(8, 0.27 * len(t) + 2.6),
        f"{n_by_cutoff} of {len(t)} columns have settled by {CUTOFF_HOURS:g} hours; most late ones trace back "
        f"to the dosing change, which we keep",
        "Each bar is one of the 36 kept columns: the hour from which every 75-minute piece stays in the settled band "
        "up to the reference window.\nOrange = settles after the cutoff (mostly the stage-2 dosing change around "
        f"hour 19-20). Dashed line = the chosen cutoff, {CUTOFF_HOURS:g} h.",
    )
    ax.barh(y[~late], t.loc[~late, "settle_hours"], color=BLUE, height=0.65)
    ax.barh(y[late], t.loc[late, "settle_hours"], color=ORANGE, height=0.65)
    ax.axvline(CUTOFF_HOURS, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(CUTOFF_HOURS, 1.0, f" cutoff {CUTOFF_HOURS:g} h", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=9, color=INK)
    ax.set_yticks(y, t["column"], fontsize=8.5)
    ax.set_xlabel("Settling time (hours from the start of the Normal file)")
    ax.set_xlim(0, REF_START_H)
    ax.set_ylim(-0.7, len(t) - 0.3)
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_settling_time_by_column.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_normal()
    columns = [c for c in sensor_columns(df) if c not in DROPPED]
    print(f"Measuring {len(columns)} kept columns in {len(df) // PIECE_ROWS} pieces of 75 minutes; "
          f"reference = hours {REF_START_H}-{REF_END_H}.\n")

    table = measure(df, columns)
    slowest_h = float(table["settle_hours"].max())
    slowest_rows = int(slowest_h * 3600)
    cutoff_rows = int(CUTOFF_HOURS * 3600)
    print("Settling time per column (# = piece in band, . = out of band):")
    print(table.to_string(index=False))
    print(f"\nStrict reading (slowest column): {slowest_h:g} h, {slowest_rows / len(df) * 100:.1f}% of the file.")
    print(f"Chosen cutoff: {CUTOFF_HOURS:g} h (row {cutoff_rows:,}, {cutoff_rows / len(df) * 100:.1f}% of the file; "
          f"{(len(df) - cutoff_rows) / 3600:.1f} h remain). "
          f"{int((table['settle_hours'] <= CUTOFF_HOURS).sum())} of {len(table)} columns have settled by then.")
    table.to_csv(OUT_DIR / "settling_by_column.csv", index=False)

    # Why the strict reading is so late: the dosing pump P205 changes how it
    # runs. Switch-on rhythm before and after the slowest column settles.
    p205 = df["P205"].to_numpy()
    starts = np.flatnonzero((p205[1:] == 2) & (p205[:-1] != 2)) + 1
    gaps = np.diff(starts) / 60
    before, after = starts[1:] < slowest_rows, starts[1:] >= slowest_rows
    print(f"\nP205 before {slowest_h:g} h: on {(p205[:slowest_rows] == 2).mean() * 100:.0f}% of the time, "
          f"median {np.median(gaps[before]):.0f} min between switch-ons")
    print(f"P205 after {slowest_h:g} h:  on {(p205[slowest_rows:] == 2).mean() * 100:.0f}% of the time, "
          f"median {np.median(gaps[after]):.0f} min between switch-ons")

    # How the cutoff changes the column rule: share of the Attack file's normal
    # rows outside the training range, with the chosen cutoff and the strict one.
    # (With the strict one, AIT202 loses its early pulses and looks drifted.)
    print("\nLoading the Attack file to compare the column rule under both cutoffs...")
    att = load_swat_attack(SWAT_DIR)
    benign = ~corrected_attack_mask(att).to_numpy()
    rows = []
    for col in columns:
        test = att[col].to_numpy(float)[benign]
        x = df[col].to_numpy(float)
        rows.append({
            "column": col,
            f"outside_pct_cut_{CUTOFF_HOURS:g}h": round(outside_range_pct(x[cutoff_rows:], test), 1),
            f"outside_pct_cut_{slowest_h:g}h": round(outside_range_pct(x[slowest_rows:], test), 1),
        })
    recheck = pd.DataFrame(rows)
    changed = recheck[(recheck.iloc[:, 1:] > OUTSIDE_LIMIT_PCT).any(axis=1)]
    print(f"Kept columns with more than {OUTSIDE_LIMIT_PCT}% of Attack-file normal rows outside the "
          f"training range, under either cutoff:")
    print(changed.to_string(index=False) if len(changed) else "  none")

    print("\nFigures:")
    fig1_first_hours(df, table)
    fig2_settle_by_column(table)
    print(f"  wrote {(OUT_DIR / 'settling_by_column.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
