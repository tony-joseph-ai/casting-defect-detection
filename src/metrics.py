"""Metrics, with defect (class 1) as the positive class.

Deliberately hand-rolled rather than a one-line sklearn call, so that the
confusion matrix terms are visible and you can explain them under questioning.
"""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tn, fp, fn, tp) for the binary case."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Full metric set at a given operating threshold.

    fn (false negative) = a defective part we labelled OK = a defect shipped
                          to the customer. This is the number that matters.
    fp (false positive) = a good part flagged = a few seconds of operator time.
    """
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion(y_true, y_pred)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)          # defect recall == 1 - miss rate
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "accuracy": accuracy,
        "precision_defect": precision,
        "recall_defect": recall,
        "specificity_ok": specificity,
        "f1_defect": f1,
        "roc_auc": roc_auc(y_true, probs),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": float(threshold),
    }


def roc_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """AUC via the rank (Mann-Whitney U) identity. Threshold-independent."""
    pos = probs[y_true == 1]
    neg = probs[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    order = np.argsort(probs)
    ranks = np.empty(len(probs), dtype=float)
    ranks[order] = np.arange(1, len(probs) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(probs, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_sum_pos = ranks[y_true == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_cost(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    """Total business cost at a threshold. Lower is better."""
    y_pred = (probs >= threshold).astype(int)
    _, fp, fn, _ = confusion(y_true, y_pred)
    return cost_fn * fn + cost_fp * fp


def sweep_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    cost_fn: float,
    cost_fp: float,
    n: int = 199,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sweep the operating threshold and return (thresholds, costs, best).

    This is the core argument of the project: 0.5 is an arbitrary default that
    silently assumes a missed defect and a false alarm cost the same. They do
    not. We choose the threshold that minimises expected cost instead.
    """
    ths = np.linspace(0.01, 0.99, n)
    costs = np.array([expected_cost(y_true, probs, t, cost_fn, cost_fp) for t in ths])
    return ths, costs, float(ths[int(np.argmin(costs))])
