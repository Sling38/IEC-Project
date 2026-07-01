"""Error analysis over a :class:`~marketfit.validation.harness.ValidationReport`.

Metrics say *how well* the model did; error analysis says *where and why* it went
wrong, which is what drives the "iterate on features" step in the Checkpoint-2 plan.

For each mis-predicted case we record:

* whether it is a **bucket** error (success/struggle flipped) and/or a **magnitude**
  error (|score error| >= a threshold),
* the **direction** (the model over- or under-predicted), and
* the features most implicated — for over-predictions, the top *drivers* that
  inflated the score; for under-predictions, the top *gaps* (weight the model
  couldn't fill). Aggregating those points at the signals to revisit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from marketfit.validation.harness import CaseResult, ValidationReport


@dataclass(frozen=True)
class CaseError:
    """One mis-predicted case with the features that best explain the miss."""

    case_id: str
    outcome_label: str
    predicted_score: int
    actual_score: int
    score_error: int          # predicted - actual
    direction: str            # "over" | "under"
    is_bucket_error: bool
    is_magnitude_error: bool
    implicated: List[Tuple[str, float]]  # (feature, contribution|gap), largest first
    note: str


@dataclass(frozen=True)
class ErrorAnalysis:
    """Aggregate diagnosis of a validation run's mistakes."""

    errors: List[CaseError]
    n_bucket_errors: int
    n_magnitude_errors: int
    over_predictions: int
    under_predictions: int
    implicated_features: List[Tuple[str, int]]  # feature -> times implicated, ranked
    worst_case: Optional[CaseError] = field(default=None)

    def summary(self) -> str:
        """Human-readable diagnosis for reports / the demo."""
        if not self.errors:
            return "No prediction errors: every case's bucket and score(+/-1) matched."
        lines = [
            f"{len(self.errors)} case(s) with errors "
            f"({self.n_bucket_errors} bucket, {self.n_magnitude_errors} magnitude>=2); "
            f"{self.over_predictions} over-, {self.under_predictions} under-predicted.",
        ]
        if self.implicated_features:
            impl = ", ".join(f"{f}({n})" for f, n in self.implicated_features[:3])
            lines.append(f"Most implicated features: {impl}")
        if self.worst_case is not None:
            w = self.worst_case
            lines.append(
                f"Worst miss: {w.case_id} predicted {w.predicted_score} vs "
                f"actual {w.actual_score} ({w.outcome_label}) — {w.note}"
            )
        return "\n".join(lines)


def _implicated(case: CaseResult, direction: str, n: int) -> List[Tuple[str, float]]:
    """Features most responsible for the miss, given its direction."""
    result = case.score_result
    if direction == "over":
        # Over-predicted: the features that contributed most pushed the score up.
        return result.top_drivers(n)
    # Under-predicted: the biggest unfilled weight (gaps) held the score down.
    return result.top_gaps(n)


def analyze_errors(
    report: ValidationReport,
    magnitude_threshold: int = 2,
    top_features: int = 2,
) -> ErrorAnalysis:
    """Diagnose the mistakes in ``report``.

    A case is an error if its success/struggle bucket is wrong *or* its score is off
    by at least ``magnitude_threshold``. ``top_features`` controls how many
    implicated features are recorded per case.
    """
    errors: List[CaseError] = []
    implicated_counter: Counter = Counter()

    for case in report.cases:
        is_bucket = not case.bucket_correct
        is_magnitude = abs(case.score_error) >= magnitude_threshold
        if not (is_bucket or is_magnitude):
            continue

        # Direction from the (possibly zero) score error; ties broken toward the
        # bucket error's sign so a flagged case always has a direction.
        if case.score_error > 0:
            direction = "over"
        elif case.score_error < 0:
            direction = "under"
        else:
            direction = "over" if case.predicted_success and not case.actual_success else "under"

        implicated = _implicated(case, direction, top_features)
        implicated_counter.update(f for f, _ in implicated)

        reasons = []
        if is_bucket:
            reasons.append(
                f"bucket flipped ({'pred success' if case.predicted_success else 'pred struggle'}"
                f" vs {'actual success' if case.actual_success else 'actual struggle'})"
            )
        if is_magnitude:
            reasons.append(f"score off by {case.score_error:+d}")
        lead = implicated[0][0] if implicated else "n/a"
        note = f"{'; '.join(reasons)}; {direction}-predicted, led by {lead}"

        errors.append(
            CaseError(
                case_id=case.case_id,
                outcome_label=case.outcome_label,
                predicted_score=case.predicted_score,
                actual_score=case.actual_score,
                score_error=case.score_error,
                direction=direction,
                is_bucket_error=is_bucket,
                is_magnitude_error=is_magnitude,
                implicated=implicated,
                note=note,
            )
        )

    worst = max(errors, key=lambda e: abs(e.score_error), default=None)
    return ErrorAnalysis(
        errors=errors,
        n_bucket_errors=sum(e.is_bucket_error for e in errors),
        n_magnitude_errors=sum(e.is_magnitude_error for e in errors),
        over_predictions=sum(e.direction == "over" for e in errors),
        under_predictions=sum(e.direction == "under" for e in errors),
        implicated_features=implicated_counter.most_common(),
        worst_case=worst,
    )
