"""Outcome-label taxonomy and the MarketFit success metric.

Per the Checkpoint 1 action items, we fix the success metric *in writing* before
building the model: every historical validation case is assigned one of four
simplified outcome labels, and each label maps to a point on a **1-5 entry-viability
score**. The prediction agent (Checkpoint 2) emits the same 1-5 score, so
"the prediction matched reality" has a concrete, comparable definition:

============= ================================================= =====
Label         Meaning                                           Score
============= ================================================= =====
Strong Success  Entered and became a large, durable market        5
Moderate Success Entered and sustained a viable presence          4
Struggled       Entered but under-performed / grew painfully      2
Withdrew        Entered then materially retreated or exited       1
============= ================================================= =====

Scores are deliberately non-contiguous (5/4/2/1) to leave a gap between "worked"
and "did not work" — a predicted 3 sits on the fence and counts as neither a hit
nor a clear miss when we bucket into success vs. struggle at Checkpoint 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class OutcomeLabel:
    """One rung of the outcome taxonomy.

    Attributes
    ----------
    name:
        Human-readable label as it appears in the ground-truth CSV.
    score:
        1-5 entry-viability score this label maps to.
    is_success:
        Coarse binary bucket (success vs. struggle) for classification metrics.
    description:
        What the label means, for report and UI display.
    """

    name: str
    score: int
    is_success: bool
    description: str


STRONG_SUCCESS = OutcomeLabel(
    "Strong Success", 5, True,
    "Entered and became a large, durable, profitable market.",
)
MODERATE_SUCCESS = OutcomeLabel(
    "Moderate Success", 4, True,
    "Entered and sustained a viable presence, if slower or smaller than the best markets.",
)
STRUGGLED = OutcomeLabel(
    "Struggled", 2, False,
    "Entered but materially under-performed or grew painfully against local competition.",
)
WITHDREW = OutcomeLabel(
    "Withdrew", 1, False,
    "Entered then retreated or exited (mass closures / market exit).",
)

# Canonical registry, keyed by label name.
OUTCOME_LABELS: Dict[str, OutcomeLabel] = {
    label.name: label
    for label in (STRONG_SUCCESS, MODERATE_SUCCESS, STRUGGLED, WITHDREW)
}


def label_for(name: str) -> OutcomeLabel:
    """Return the :class:`OutcomeLabel` for ``name`` (raises ``KeyError`` if unknown)."""
    try:
        return OUTCOME_LABELS[name]
    except KeyError as exc:
        valid = ", ".join(OUTCOME_LABELS)
        raise KeyError(f"Unknown outcome label {name!r}. Expected one of: {valid}.") from exc


def score_for(name: str) -> int:
    """Return the 1-5 entry-viability score for outcome label ``name``."""
    return label_for(name).score
