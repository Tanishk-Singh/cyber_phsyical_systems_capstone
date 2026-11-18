"""Step 8 metrics, through TSB-AD (Liu & Paparrizos, NeurIPS 2024).

  VUS-PR    Volume under the precision-recall surface: the area under the
            precision-recall curve (all thresholds), averaged over tolerance
            buffers of 0..`sliding_window` rows around each attack, so a
            detection a few seconds early or late still earns credit. No
            threshold needed; our headline metric. sliding_window = 100 rows
            (100 s), TSB-AD's default.
  ROC-AUC   Area under the ROC curve, all thresholds, no tolerance.
  AUC-PR    Area under the precision-recall curve, no tolerance.
  Range-F1  Tatbul et al. (2018) range-based F1 as implemented in TSB-AD
            (existence reward weight alpha = 0.2). Computed on OUR alarms
            (Step 7 rule), never on TSB-AD's default, which searches 100
            thresholds against the test labels (an oracle).

TSB-AD 1.5 is installed with --no-deps: only its evaluation code is used.
"""
import numpy as np
from TSB_AD.evaluation.basic_metrics import basic_metricor, generate_curve

SLIDING_WINDOW = 100


def vus(scores: np.ndarray, labels: np.ndarray, sliding_window: int = SLIDING_WINDOW) -> tuple[float, float]:
    """(VUS-PR, VUS-ROC). Slow on SWaT-sized data: about 2 minutes."""
    *_, vus_roc, vus_pr = generate_curve(np.asarray(labels, int), np.asarray(scores, float),
                                         sliding_window, "opt", 250)
    return float(vus_pr), float(vus_roc)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(basic_metricor().metric_ROC(np.asarray(labels, int), np.asarray(scores, float)))


def auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(basic_metricor().metric_PR(np.asarray(labels, int), np.asarray(scores, float)))


def range_f1(labels: np.ndarray, alarms: np.ndarray) -> float:
    """Range-based F1 of a given set of alarms (no threshold search)."""
    labels = np.asarray(labels, int)
    preds = np.asarray(alarms, int)
    # the score argument is ignored when preds is given
    return float(basic_metricor().metric_RF1(labels, np.zeros(len(labels)), preds=preds))
