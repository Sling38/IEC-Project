"""Validation harness, metrics, and error analysis for MarketFit (Checkpoint 2).

Runs the entry-viability scorer over the curated ground-truth market-entry cases
and measures how well predictions match documented outcomes:

* :mod:`marketfit.validation.metrics`        — score and bucket metrics.
* :mod:`marketfit.validation.harness`        — the harness + validation report.
* :mod:`marketfit.validation.error_analysis` — where/why the model is wrong.
"""

from marketfit.validation.metrics import (
    BucketMetrics,
    ScoreMetrics,
    bucket_metrics,
    score_metrics,
    spearman,
)
from marketfit.validation.harness import (
    CaseResult,
    ValidationHarness,
    ValidationReport,
    load_signal_fixtures,
)
from marketfit.validation.error_analysis import (
    CaseError,
    ErrorAnalysis,
    analyze_errors,
)

__all__ = [
    "BucketMetrics",
    "ScoreMetrics",
    "bucket_metrics",
    "score_metrics",
    "spearman",
    "CaseResult",
    "ValidationHarness",
    "ValidationReport",
    "load_signal_fixtures",
    "CaseError",
    "ErrorAnalysis",
    "analyze_errors",
]
