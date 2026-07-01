"""Ground-truth market-entry cases for MarketFit validation.

Checkpoint 1 (Samuel) curates the labeled dataset that later checkpoints score
against. It has two parts:

* :mod:`marketfit.groundtruth.labels` — the outcome-label taxonomy and the 1-5
  entry-viability score each label maps to (the project's success metric).
* :mod:`marketfit.groundtruth.cases`  — a loader for the curated case CSV in
  ``data/ground_truth/``.
"""

from marketfit.groundtruth.labels import (
    OUTCOME_LABELS,
    OutcomeLabel,
    label_for,
    score_for,
)
from marketfit.groundtruth.cases import GroundTruthLoader, DEFAULT_CASES_PATH

__all__ = [
    "OUTCOME_LABELS",
    "OutcomeLabel",
    "label_for",
    "score_for",
    "GroundTruthLoader",
    "DEFAULT_CASES_PATH",
]
