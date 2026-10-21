"""Hand-corrected SWaT attack labels from Fung et al. (ESORICS 2022).

`SWaT_Dataset_Attack_v0.xlsx`'s own `Normal/Attack` column is not usable as
ground truth. Four defects, all verified directly against the sensor traces:

1. Attack onsets are offset from when the manipulation actually appears in the
   data: 16-40s LATE on 2015-12-28..31, then 35-131s EARLY on 2016-01-01..02.
   E.g. attack 3 spoofs LIT101 to climb exactly +1.000mm/s starting 11:21:41,
   but the label does not start until 11:22:00.
2. Rows 73800-74520 are labeled Attack but match no documented attack. They sit
   exactly 12 hours before attack 21 and run for exactly the same 721s; the
   sensors there show ordinary noise. An AM/PM transcription error.
3. Attacks 27 and 28 are run together into one 35,900s segment across 1157s of
   genuinely normal data. Attack 27's own CSV text says the LIT401 spoof ran
   "till 1:26:01", and LIT401 is indeed pinned at a constant 600.00 until then.
4. Attacks 4, 14 and 29 are labeled Attack but left no trace in the 51 recorded
   columns -- see BENIGN_ATTACK_RANGES below.

Source: github.com/pwwl/ics-anomaly-detection -- `data/SWAT/process_SWaT.py`
for the row ranges, `swat-corrections.md` for the wall-clock table they were
derived from. The rationale is in the paper's Appendix A, Finding #2 ("Attack
Cleaning"). Zizzo et al. (arXiv:1911.04278) first reported the timing problem.

Row indices are 0-based over the attack file's data rows, so row N is
10:00:00 + N seconds. Verified against this package's loader on attacks
1, 3, 33, 37 and 41 -- each lands exactly on the physical onset.
All ranges are half-open, [start, end).
"""

# Number of data rows in SWaT_Dataset_Attack_v0.xlsx. The ranges below are
# row-indexed, so they are only meaningful for a file of exactly this length.
N_ATTACK_ROWS = 449919

# (start, end, attack number as documented in List_of_attacks_Final(Sheet1).csv)
# 32 windows. Attacks 10 and 11 are genuinely continuous and share one window.
CORRECTED_ATTACK_RANGES = [
    (1738, 2673, 1),
    (3046, 3491, 2),
    (4901, 5283, 3),
    (7233, 7432, 6),
    (7685, 8113, 7),
    (11385, 12355, 8),
    (15361, 16084, 10),  # covers documented attacks 10 and 11
    (90662, 90917, 13),
    (93424, 93705, 16),
    (103092, 103797, 17),
    (115822, 116080, 19),
    (116123, 116515, 20),
    (116999, 117701, 21),
    (132896, 133362, 22),
    (142927, 143611, 23),
    (172268, 172588, 24),
    (172892, 173499, 25),
    (198273, 199716, 26),
    (227828, 228362, 27),  # ends 01:26:01, where the LIT401 spoof stops
    (229519, 263727, 28),  # the 1157s between 27 and 28 is genuinely normal
    (280023, 281185, 30),
    (302653, 303020, 31),
    (347718, 348315, 32),
    (361243, 361674, 33),
    (371519, 371618, 34),
    (371893, 372374, 35),
    (389746, 390262, 36),
    (436672, 437046, 37),
    (437455, 437734, 38),
    (438184, 438584, 39),
    (438659, 438955, 40),
    (443540, 445191, 41),
]

# Documented attacks that the xlsx labels as Attack but which left no trace in
# the 51 recorded columns. Scored as Normal (Fung does the same), but kept here
# so model behaviour inside them can be reported as an unscored diagnostic.
# Ranges are the RAW label segments, not corrected ones -- Fung publishes no
# corrected times for these three.
BENIGN_ATTACK_RANGES = [
    # "Open MV-504". The valve did open, but MV-504 is not one of the 51
    # recorded columns, and 0 of 51 leave their baseline. CSV: "No impact".
    (6459, 6849, 4),
    # "Do not let MV-303 open". Tank 301 was already full so the UF startup
    # sequence never ran and the valve was never going to open. MV303 holds a
    # constant 1 (closed), its normal resting state. CSV: "Attack Failed".
    (92140, 92571, 14),
    # "Turn on P-201, P-203, P-205". A mechanical interlock refused the
    # command; all three hold a constant 1 (off) and never reach 2 (on).
    (279120, 279241, 29),
]

# The AM/PM duplicate of attack 21: labeled Attack, no documented attack, and
# LIT101 shows ordinary noise (504-519, std 1.85) rather than attack 21's
# pinned 700.0. Scored as Normal.
PHANTOM_RANGE = (73800, 74521)
