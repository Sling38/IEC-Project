"""Validation metrics for MarketFit predictions vs. ground truth.

Two families of metric, matching the two things the scorer emits:

* **Score metrics** treat the 1-5 entry-viability score as an ordinal prediction —
  MAE, RMSE, exact-match and within-1 accuracy, signed bias, and Spearman rank
  correlation (does the model *order* markets like reality does?).
* **Bucket metrics** treat the success/struggle flag as binary classification —
  accuracy, precision, recall, F1, and the confusion counts, with *success* as the
  positive class.

Everything is implemented from the standard library (no scikit-learn / scipy) so it
stays dependency-light and fully offline-testable, consistent with the rest of the
pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class ScoreMetrics:
    """Ordinal-regression metrics over the 1-5 entry-viability score."""

    n: int
    mae: float                  # mean absolute error
    rmse: float                 # root mean squared error
    exact_accuracy: float       # fraction with predicted == actual
    within_one_accuracy: float  # fraction with |predicted - actual| <= 1
    bias: float                 # mean(pred - actual); >0 over-predicts, <0 under
    rank_correlation: float     # Spearman rho of predicted vs actual scores


@dataclass(frozen=True)
class BucketMetrics:
    """Binary-classification metrics for success (positive) vs. struggle."""

    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def _rank(values: Sequence[float]) -> List[float]:
    """Return 1-based ranks, averaging ties (as Spearman requires)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # positions i..j share the average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx <= 0 or vy <= 0:  # a constant vector has no defined correlation
        return 0.0
    return cov / math.sqrt(vx * vy)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation in [-1, 1]; 0.0 when undefined (n<2 or constant)."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    return _pearson(_rank(a), _rank(b))


def score_metrics(
    predicted: Sequence[int], actual: Sequence[int]
) -> ScoreMetrics:
    """Compute ordinal metrics between predicted and actual 1-5 scores."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must be the same length")
    n = len(predicted)
    if n == 0:
        return ScoreMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    errors = [p - a for p, a in zip(predicted, actual)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    exact = sum(e == 0 for e in errors) / n
    within_one = sum(abs(e) <= 1 for e in errors) / n
    bias = sum(errors) / n
    rho = spearman(predicted, actual)

    return ScoreMetrics(
        n=n,
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        exact_accuracy=round(exact, 4),
        within_one_accuracy=round(within_one, 4),
        bias=round(bias, 4),
        rank_correlation=round(rho, 4),
    )


def bucket_metrics(
    predicted: Sequence[bool], actual: Sequence[bool]
) -> BucketMetrics:
    """Compute binary metrics with ``True`` (success) as the positive class."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must be the same length")
    n = len(predicted)
    tp = sum(bool(p) and bool(a) for p, a in zip(predicted, actual))
    fp = sum(bool(p) and not bool(a) for p, a in zip(predicted, actual))
    fn = sum(not bool(p) and bool(a) for p, a in zip(predicted, actual))
    tn = sum(not bool(p) and not bool(a) for p, a in zip(predicted, actual))

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return BucketMetrics(
        n=n,
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )
