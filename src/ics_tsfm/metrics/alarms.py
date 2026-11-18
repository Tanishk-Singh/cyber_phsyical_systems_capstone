"""Scoring a set of alarms against the true attack windows (Phase A, Step 7).

Three views, because each answers a different question:
  - Rows (point-wise): precision, recall and F1 over individual rows. Long
    attacks dominate (attack 28 is two thirds of all attack rows).
  - Attacks: how many of the attack windows got at least one alarm, and how
    many seconds after the attack started the first alarm came.
  - False-alarm events: unbroken alarm runs that touch no attack window. This
    is what an operator experiences; one 10-minute false alarm is one event.

An alarm run that overlaps an attack counts as a true detection even if it
runs on past the attack's end (windowed models keep scores high for up to
`history` rows after an attack, because their window still contains it).

Official threshold-free metrics (VUS-PR, range-F1) come in Step 8.
"""
import numpy as np

from ics_tsfm.detect.rule import events


def score_alarms(alarm: np.ndarray, is_attack: np.ndarray, scored: np.ndarray,
                 attack_ranges) -> dict:
    """alarm, is_attack, scored: one bool per test row. `scored` marks the rows
    every compared model can score; only those are counted. attack_ranges:
    (start, end, attack_number) per attack window, end exclusive."""
    alarm = np.asarray(alarm, bool) & scored
    y = np.asarray(is_attack, bool)

    tp = int((alarm & y & scored).sum())
    fp = int((alarm & ~y & scored).sum())
    fn = int((~alarm & y & scored).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_attack = []
    for start, end, number in attack_ranges:
        hits = np.flatnonzero(alarm[start:end])
        per_attack.append({"attack": int(number), "length_s": int(end - start),
                           "detected": bool(len(hits)),
                           "delay_s": int(hits[0]) if len(hits) else None})

    runs = events(alarm)
    touches_attack = np.array([y[s:e].any() for s, e in runs], bool) if len(runs) else np.array([], bool)
    false_runs = runs[~touches_attack] if len(runs) else runs
    normal_hours = float((~y & scored).sum()) / 3600

    detected = [a for a in per_attack if a["detected"]]
    return {
        "point_precision": precision, "point_recall": recall, "point_f1": f1,
        "attacks_detected": len(detected), "attacks_total": len(per_attack),
        "median_delay_s": float(np.median([a["delay_s"] for a in detected])) if detected else None,
        "false_alarm_events": int(len(false_runs)),
        "false_alarm_events_per_day": len(false_runs) / normal_hours * 24 if normal_hours else 0.0,
        "false_alarm_seconds": int(sum(e - s for s, e in false_runs)),
        "per_attack": per_attack,
    }
