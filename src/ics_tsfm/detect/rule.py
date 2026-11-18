"""Detection rule (Phase A, Step 7): turning anomaly scores into alarms.

Every model gives one score per row (NaN where it cannot score). The rule:

  1. A row is "suspicious" when its score is above the threshold tau.
  2. An alarm is raised at row t only when the last w scored rows, t included,
     are all suspicious ("alarm persistence", w = 10 seconds by default).
     A NaN row is never suspicious, so a run never bridges rows a model could
     not score (the gap in the Attack file, the seams between validation
     blocks).
  3. tau is set from benign validation scores only: the highest validation
     score (`zero_alarm_threshold(val, w=1)`, i.e. zero false alarms on
     validation even without persistence). The persistence w is applied
     AFTER that, only to suppress brief blips on test data.

Worked example, w = 3:
    scores      1  5  6  7  2  8  9  4
    tau = 4:    .  S  S  S  .  S  S  .        S = suspicious
    alarm:      .  .  .  A  .  .  .  .        only row 3 ends 3 suspicious rows

Why tau is calibrated with w = 1 ("decoupled rule", adopted 2026-11-17 after
the first run, same rule for every model). Calibrating with the persistence
itself (`zero_alarm_threshold(val, w=10)`) only has to stay above the highest
score that LASTS 10 rows. On validation high scores are brief (the CNN's
longest run above its 99.9th percentile is 3-4 s), so that lowered the CNN's
threshold about 20-fold (7.0 -> 0.3), to the level of normal fluctuation. The
Attack file's drifted sensors keep errors elevated for hours, so that
threshold produced about 100 false-alarm events a day. With tau above every
validation score, persistence trims blips instead of lowering the threshold.
The warning sign is visible on validation alone: tau(w=10) far below tau(w=1).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DEFAULT_PERSISTENCE = 10      # seconds (rows at 1 Hz)


def run_lengths(flag: np.ndarray) -> np.ndarray:
    """How many consecutive True values end at each position.

    Example: [F, T, T, F, T] -> [0, 1, 2, 0, 1]
    """
    flag = np.asarray(flag, bool)
    idx = np.arange(len(flag))
    last_false = np.maximum.accumulate(np.where(~flag, idx, -1))
    return np.where(flag, idx - last_false, 0)


def alarms(scores: np.ndarray, tau: float, w: int = DEFAULT_PERSISTENCE) -> np.ndarray:
    """True at every row where the last w scored rows were all above tau."""
    suspicious = np.nan_to_num(np.asarray(scores, float), nan=-np.inf) > tau
    return run_lengths(suspicious) >= w


def zero_alarm_threshold(val_scores: np.ndarray, w: int = DEFAULT_PERSISTENCE) -> float:
    """The lowest tau that raises no alarm on the validation scores.

    An alarm needs w consecutive rows above tau, i.e. a stretch of w rows whose
    LOWEST score is above tau. So take the lowest score of every stretch of w
    consecutive rows; tau is the highest of those. At exactly that tau the
    worst stretch is not strictly above it, so nothing fires. Stretches that
    contain a NaN (unscored row) cannot alarm and are skipped.
    With w = 1 this is simply the highest validation score.
    """
    s = np.asarray(val_scores, float)
    if len(s) < w:
        raise ValueError("fewer validation rows than the alarm persistence")
    stretch_min = sliding_window_view(s, w).min(axis=1)      # NaN if the stretch has a NaN
    if np.all(np.isnan(stretch_min)):
        raise ValueError(f"no stretch of {w} consecutive scored validation rows")
    return float(np.nanmax(stretch_min))


def events(flag: np.ndarray) -> np.ndarray:
    """(start, end) of every unbroken run of True, end exclusive.

    Example: [F, T, T, F, T] -> [[1, 3], [4, 5]]
    """
    padded = np.r_[False, np.asarray(flag, bool), False].astype(int)
    edges = np.diff(padded)
    return np.column_stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)])
